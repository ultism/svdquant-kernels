"""Triton kernels for `quantize_w4a4_act_fuse_lora`.

Two callable entry points:

- `lora_down(x, w)` — isolated LoRA-down matmul, the prototype / scaffold.
- `quantize_w4a4_act_fuse_lora(...)` — the full fused op: shares one HBM
  read of `x` between the LoRA-down tl.dot and the NVFP4 pack.

INT4 branch is stubbed — it exists so that the Ascend path has the same
call site once `triton-ascend` ships. NVFP4 is the CUDA path.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


# NVFP4 max finite magnitude = 6.0 (E2M1 with exp bias 1, mantissa 1 → {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}).
_NVFP4_AMAX = tl.constexpr(6.0)
_NVFP4_AMAX_INV = tl.constexpr(1.0 / 6.0)
# FP8-E4M3 max finite value — scales are clamped to this before cast to avoid saturation.
# Matches nunchaku's `MSCALE_MAX` at `gemm_w4a4.cuh:93`.
_FP8_E4M3_MAX = tl.constexpr(448.0)


@triton.jit
def _lora_down_kernel(
    input_ptr, lora_down_ptr, out_ptr,
    M, N, R,
    stride_im, stride_in,
    stride_ln, stride_lr,
    stride_om, stride_or,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)

    mask_m = offs_m < M
    mask_r = offs_r < R

    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

    input_base = input_ptr + offs_m[:, None] * stride_im
    lora_base = lora_down_ptr + offs_r[None, :] * stride_lr

    for k in range(0, N, BLOCK_N):
        k_offs = k + offs_n
        mask_n = k_offs < N

        a = tl.load(
            input_base + k_offs[None, :] * stride_in,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        b = tl.load(
            lora_base + k_offs[:, None] * stride_ln,
            mask=mask_n[:, None] & mask_r[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_r[None, :] * stride_or,
        acc,
        mask=mask_m[:, None] & mask_r[None, :],
    )


def lora_down(
    input: torch.Tensor,       # [M, N] fp16/bf16
    weight: torch.Tensor,      # [N, R] fp16/bf16, same dtype as input
    *,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:             # [M, R] fp32
    """Standalone LoRA-down projection. Prototype for the fused op."""
    assert input.dim() == 2 and weight.dim() == 2
    assert input.is_cuda and weight.is_cuda
    assert input.dtype == weight.dtype
    assert input.dtype in (torch.float16, torch.bfloat16)

    M, N = input.shape
    N2, R = weight.shape
    assert N == N2, f"K mismatch: input K={N}, weight K={N2}"

    out = torch.empty((M, R), dtype=torch.float32, device=input.device)

    # R is a small LoRA rank (32/128/256 in ZImage Turbo); round to next
    # power of two so the whole R axis fits in one block, and enforce the
    # 16-min dim that `tl.dot` needs on sm_80+.
    block_r = max(16, triton.next_power_of_2(R))

    grid = (triton.cdiv(M, block_m),)
    _lora_down_kernel[grid](
        input, weight, out,
        M, N, R,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
    )
    return out


@triton.jit
def _fused_step(
    k0,
    offs_m, offs_n, mask_m, mask_r,
    x_row_base, ld_col_base,
    smooth_ptr, qout_ptr, oscales_ptr,
    K,
    stride_xk, stride_lk, stride_sm_k,
    stride_qm, stride_qk, stride_og, stride_om,
    HAS_SMOOTH: tl.constexpr,
    FP4: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """One (BLOCK_M × BLOCK_N) step: load x, LoRA-down dot, smooth+quantize,
    store qout / oscales. Returns the fp32 dot output — caller decides
    whether to accumulate into a persistent reg tile or atomic_add into HBM.
    """
    k_offs = k0 + offs_n
    mask_n = k_offs < K

    # --- Load x tile once; feeds both branches. ---
    x_tile = tl.load(
        x_row_base + k_offs[None, :] * stride_xk,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    # --- LoRA-down (tensor cores). ---
    ld_tile = tl.load(
        ld_col_base + k_offs[:, None] * stride_lk,
        mask=mask_n[:, None] & mask_r[None, :],
        other=0.0,
    )
    dot_out = tl.dot(x_tile, ld_tile, out_dtype=tl.float32)

    # --- smooth + NVFP4 pack (vector units). ---
    x_hat = x_tile.to(tl.float32)
    if HAS_SMOOTH:
        smooth = tl.load(
            smooth_ptr + k_offs * stride_sm_k, mask=mask_n, other=1.0
        ).to(tl.float32)
        x_hat = x_hat / smooth[None, :]

    if FP4:
        x_blocks = tl.reshape(x_hat, (BLOCK_M, BLOCK_N // 16, 16))
        amax = tl.max(tl.abs(x_blocks), axis=2)
        scale_f32 = tl.maximum(amax * _NVFP4_AMAX_INV, 1e-12)
        scale_f32 = tl.minimum(scale_f32, _FP8_E4M3_MAX)
        scale_fp8 = scale_f32.to(tl.float8e4nv, fp_downcast_rounding="rtne")
        scale_back = scale_fp8.to(tl.float32)

        scaled = x_blocks / scale_back[:, :, None]
        scaled = tl.reshape(scaled, (BLOCK_M, BLOCK_N))

        ax = tl.abs(scaled)
        mag = (
            (ax >= 0.25).to(tl.int32)
            + (ax >= 0.75).to(tl.int32)
            + (ax >= 1.25).to(tl.int32)
            + (ax >= 1.75).to(tl.int32)
            + (ax >= 2.5).to(tl.int32)
            + (ax >= 3.5).to(tl.int32)
            + (ax >= 5.0).to(tl.int32)
        )
        sign_bit = (scaled < 0).to(tl.int32) << 3
        nib = (mag | sign_bit).to(tl.uint8)

        nib_pairs = tl.reshape(nib, (BLOCK_M, BLOCK_N // 2, 2))
        nib_lo, nib_hi = tl.split(nib_pairs)
        qbyte = nib_lo | (nib_hi << 4)

        offs_k_half = (k0 // 2) + tl.arange(0, BLOCK_N // 2)
        mask_k_half = offs_k_half < (K // 2)
        q_ptrs = qout_ptr + offs_m[:, None] * stride_qm + offs_k_half[None, :] * stride_qk
        tl.store(q_ptrs, qbyte, mask=mask_m[:, None] & mask_k_half[None, :])

        offs_g = (k0 // 16) + tl.arange(0, BLOCK_N // 16)
        mask_g = offs_g < (K // 16)
        os_ptrs = oscales_ptr + offs_g[:, None] * stride_og + offs_m[None, :] * stride_om
        tl.store(os_ptrs, tl.trans(scale_fp8), mask=mask_g[:, None] & mask_m[None, :])
    else:
        # INT4 (Ascend) placeholder; gated off at JIT time.
        tl.static_assert(False, "INT4 branch pending; pass fp4=True.")

    return dot_out


@triton.jit
def _quantize_w4a4_act_fuse_lora_kernel(
    x_ptr, lora_down_ptr, smooth_ptr,
    qout_ptr, oscales_ptr, lora_act_ptr,
    M, K, R,                        # M is the real (un-padded) row count
    stride_xm, stride_xk,
    stride_lk, stride_lr,
    stride_sm_k,                    # smooth is 1D [K]; one stride
    stride_qm, stride_qk,
    stride_og, stride_om,           # oscales is [K/16, M_pad]
    stride_am, stride_ar,
    HAS_SMOOTH: tl.constexpr,
    FP4: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,          # must be multiple of 16
    BLOCK_R: tl.constexpr,
    K_SPLITS: tl.constexpr,         # # of CTAs along K
    ACC_AT_END: tl.constexpr,       # False = straight 1-iter kernel, atomic per CTA
):
    # 2D grid `(pid_m, pid_k)`. Two JIT specialisations:
    #
    #   ACC_AT_END=True — fewer, longer CTAs. Persistent fp32 acc[BM, BR]
    #       accumulates over K/K_SPLITS tiles, atomic_add once at the end.
    #       Good for R≥64 where the acc cost pays off (one atomic vs many).
    #
    #   ACC_AT_END=False — many short CTAs (K_SPLITS = total BLOCK_N tiles).
    #       No persistent acc, no inner loop at all — straight-through to a
    #       single atomic_add. Preserves the R=32 latency of the original
    #       2D-grid kernel, which Triton's range() + cdiv machinery can't
    #       otherwise match (it can't constant-fold `cdiv(K, BLOCK_N)` since
    #       K is a runtime arg).
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)

    mask_m = offs_m < M
    mask_r = offs_r < R

    la_ptrs = lora_act_ptr + offs_m[:, None] * stride_am + offs_r[None, :] * stride_ar
    la_mask = mask_m[:, None] & mask_r[None, :]

    x_row_base = x_ptr + offs_m[:, None] * stride_xm
    ld_col_base = lora_down_ptr + offs_r[None, :] * stride_lr

    if ACC_AT_END:
        # Split K into BLOCK_N-aligned ranges — 16-K oscales groups never
        # straddle a split, so no race on the scale slot.
        n_tiles_total = tl.cdiv(K, BLOCK_N)
        n_tiles_per_split = tl.cdiv(n_tiles_total, K_SPLITS)
        start_tile = pid_k * n_tiles_per_split
        end_tile = tl.minimum(start_tile + n_tiles_per_split, n_tiles_total)
        k_start = start_tile * BLOCK_N
        k_end = end_tile * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for k0 in range(k_start, k_end, BLOCK_N):
            acc += _fused_step(
                k0, offs_m, offs_n, mask_m, mask_r,
                x_row_base, ld_col_base,
                smooth_ptr, qout_ptr, oscales_ptr,
                K,
                stride_xk, stride_lk, stride_sm_k,
                stride_qm, stride_qk, stride_og, stride_om,
                HAS_SMOOTH=HAS_SMOOTH, FP4=FP4,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_R=BLOCK_R,
            )
        tl.atomic_add(la_ptrs, acc, mask=la_mask)
    else:
        k0 = pid_k * BLOCK_N
        dot_out = _fused_step(
            k0, offs_m, offs_n, mask_m, mask_r,
            x_row_base, ld_col_base,
            smooth_ptr, qout_ptr, oscales_ptr,
            K,
            stride_xk, stride_lk, stride_sm_k,
            stride_qm, stride_qk, stride_og, stride_om,
            HAS_SMOOTH=HAS_SMOOTH, FP4=FP4,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_R=BLOCK_R,
        )
        tl.atomic_add(la_ptrs, dot_out, mask=la_mask)


def quantize_w4a4_act_fuse_lora(
    input: torch.Tensor,             # [M, K]   fp16/bf16
    lora_down: torch.Tensor,         # [K, R]   fp16/bf16 (diag(1/smooth) pre-absorbed offline)
    smooth: Optional[torch.Tensor],  # [K]      fp16/bf16 or None
    *,
    fp4: bool,                       # True = NVFP4 (CUDA), False = INT4 (Ascend, not impl)
    pad_size: int = 256,
    block_m: int = 64,
    block_n: int = 64,
    k_splits: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused quantize + LoRA-down projection.

    Output format picks per backend: NVFP4 on CUDA (B200 tcgen05 has no
    INT4 scaled-MMA), INT4 on Ascend (its cube unit has no FP4). Caller
    passes `fp4` explicitly — we don't auto-detect from the tensor's
    device because triton-ascend may or may not show up as a distinct
    device type depending on the install.

    Returns (qout, oscales, lora_act):
      qout      : [M_pad, K/2]   uint8        — two NVFP4 nibbles per byte
      oscales   : [K/16,  M_pad] fp8_e4m3fn   — per-16-K-block scale (transposed)
      lora_act  : [M_pad, R]     fp32         — zero-padded beyond M

    `smooth` is optional; when None, the quantize branch sees raw x.

    We don't support nunchaku's `fuse_glu` path: that would require vLLM
    to hand us the pre-GLU [M, 2K] tensor, which is a pipeline intrusion
    we're avoiding.
    """
    assert input.dim() == 2 and lora_down.dim() == 2
    assert input.is_cuda and lora_down.is_cuda
    assert input.dtype == lora_down.dtype
    assert input.dtype in (torch.float16, torch.bfloat16)

    M, K = input.shape
    K2, R = lora_down.shape
    assert K == K2, f"K mismatch: input K={K}, lora_down K={K2}"
    assert K % 16 == 0, "NVFP4 group size is 16; K must be divisible by 16"
    assert block_n % 16 == 0, "BLOCK_N must be a multiple of 16 (NVFP4 group)"

    if smooth is not None:
        assert smooth.is_cuda and smooth.dtype == input.dtype
        assert smooth.shape == (K,)

    if not fp4:
        raise NotImplementedError("INT4 branch pending; for now only fp4=True.")

    M_pad = ((M + pad_size - 1) // pad_size) * pad_size

    qout = torch.empty((M_pad, K // 2), dtype=torch.uint8, device=input.device)
    oscales = torch.empty((K // 16, M_pad), dtype=torch.float8_e4m3fn, device=input.device)
    # lora_act is atomic-added into by K_SPLITS CTAs per M-tile; must start at zero.
    lora_act = torch.zeros((M_pad, R), dtype=torch.float32, device=input.device)

    block_r = max(16, triton.next_power_of_2(R))

    # K_SPLITS adapts to R. Two opposing costs:
    #   - Persistent fp32 `acc[BLOCK_M, BLOCK_R]` regs/thread = BLOCK_R/2 (at
    #     num_warps=4). Scales linearly with R; bigger R → bigger acc → fewer
    #     splits keeps reg pressure out of spill territory *and* shrinks the
    #     total atomic_add HBM footprint (K_SPLITS × M × R × 4).
    #   - More splits → more CTAs → better SM coverage. When R is small the
    #     persistent acc is tiny and the atomic HBM is tiny, so we want to
    #     push K_SPLITS up to the tile count — essentially the old fully-
    #     split-K launch.
    # Tiers pinned empirically on RTX-PRO-6000 Blackwell (SM_120, 188 SMs);
    # B200 has fewer SMs (~148) but the relative picture is the same.
    n_tiles_total = triton.cdiv(K, block_n)
    if k_splits is None:
        if R <= 32:
            # Match the original per-BLOCK_N launch: every CTA does 1 iter
            # with an inline atomic_add. Persistent-acc variants all measure
            # slower on R=32 — the register-file win doesn't materialise, and
            # the loop header / `cdiv` overhead shows up when CTAs are short.
            k_splits = n_tiles_total
            acc_at_end = False
        elif R <= 64:
            k_splits = min(n_tiles_total, 32)
            acc_at_end = True
        elif R <= 128:
            k_splits = min(n_tiles_total, 16)
            acc_at_end = True
        else:
            k_splits = min(n_tiles_total, 8)
            acc_at_end = True
    else:
        # Explicit override: persist acc unless the caller picked the maximum
        # split (i.e. one iter per CTA) where per-iter atomic is a freebie.
        acc_at_end = k_splits < n_tiles_total
    grid = (triton.cdiv(M, block_m), k_splits)
    _quantize_w4a4_act_fuse_lora_kernel[grid](
        input, lora_down, smooth if smooth is not None else input,  # dummy ptr when unused
        qout, oscales, lora_act,
        M, K, R,
        input.stride(0), input.stride(1),
        lora_down.stride(0), lora_down.stride(1),
        smooth.stride(0) if smooth is not None else 0,
        qout.stride(0), qout.stride(1),
        oscales.stride(0), oscales.stride(1),
        lora_act.stride(0), lora_act.stride(1),
        HAS_SMOOTH=smooth is not None,
        FP4=fp4,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        K_SPLITS=k_splits,
        ACC_AT_END=acc_at_end,
        num_warps=4,
        num_stages=2,
    )
    return qout, oscales, lora_act
