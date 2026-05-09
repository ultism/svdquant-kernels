"""INT4 (Ascend) row-wise quantization primitive.

Signed INT4 values (range `[-8, 7]`, 4-bit two's complement) with
per-K-block FP16 scales. Mirrors the layout of `_nvfp4` so the
`gemm_w4a4` reference can take packed inputs in either format.

Block size = 64 matches `KS=64` in
`csrc/kernels/gemm_w4a4/ascend/PLAN.md` — one `mad_s4` call per
K-block, scale applied in vec post-MMA. Symmetric RTN, no zero-point
(SVDQuant uses sym; deepcompressor RTN port).
"""
from __future__ import annotations

import torch

# Signed INT4 covers 16 levels in [-8, 7] (4-bit two's complement).
INT4_AMAX = 7.0
INT4_NEG_MIN = -8.0
# fp16 finite max — scales clamp here pre-cast.
FP16_MAX = 65504.0
# Per-K-block group size for the Ascend cube path. Matches AscendC
# `mad_s4` KS in PLAN.md so vec-side dequant aligns with one MMA call.
INT4_BLOCK_SIZE = 64


def _quantize_signed_int4(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 (post-scale) to signed INT4, return uint8 nibble.

    Encoding: bits 0-3 = signed value in two's complement.
    `0x0=0, 0x7=7, 0x8=-8, 0xF=-1`. The `& 0x0F` masks the sign-extended
    high bits int8 carries after `.to(int8)`.
    """
    q = x.round().clamp(INT4_NEG_MIN, INT4_AMAX).to(torch.int8)
    return (q & 0x0F).to(torch.uint8)


def _pack_nibbles(nibs: torch.Tensor) -> torch.Tensor:
    """Pack last-dim pairs of 4-bit nibbles into uint8. Low nibble = even k."""
    assert nibs.shape[-1] % 2 == 0
    lo = nibs[..., 0::2]
    hi = nibs[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_signed_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """`[*, K/2]` uint8 → `[*, K]` int8 with sign extension.

    Inverse of `_pack_nibbles` for signed INT4. Nibble `>=8` represents
    a negative value: bit 3 is the sign bit, so we subtract 16 to recover
    the int8 value (e.g. `0x8 → -8, 0xF → -1`).
    """
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    nibs = torch.stack([lo, hi], dim=-1).view(*packed.shape[:-1], packed.shape[-1] * 2)
    signed = nibs.to(torch.int8)
    return torch.where(signed >= 8, signed - 16, signed)


def quantize_int4_rows(
    x: torch.Tensor,
    *,
    block_size: int = INT4_BLOCK_SIZE,
    pad_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise signed-INT4 quantization with per-K-block FP16 scales.

    Args:
        x:          `[rows, K]` floating-point tensor.
        block_size: K-block grouping size. Default 64 matches AscendC.
        pad_size:   row-dim alignment.

    Returns:
        packed: `[rows_pad, K/2]`               uint8  — INT4 nibbles, low = even k.
        scales: `[K/block_size, rows_pad]`      fp16   — nunchaku-style transposed.

    Padding rows are zeros (packed) and `1e-12` (scales).
    """
    assert x.dim() == 2
    rows, K = x.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    rows_pad = ((rows + pad_size - 1) // pad_size) * pad_size

    x_pad = torch.zeros(rows_pad, K, dtype=torch.float32, device=x.device)
    x_pad[:rows] = x.to(torch.float32)

    blocks = x_pad.view(rows_pad, K // block_size, block_size)
    amax = blocks.abs().amax(dim=-1)                                 # [rows_pad, K/B]
    scale_f32 = (amax / INT4_AMAX).clamp(min=1e-12, max=FP16_MAX)
    scale_fp16 = scale_f32.to(torch.float16)
    scale_back = scale_fp16.to(torch.float32)

    x_scaled = blocks / scale_back.unsqueeze(-1)                     # [rows_pad, K/B, B]
    nibs = _quantize_signed_int4(x_scaled).view(rows_pad, K)
    packed = _pack_nibbles(nibs)                                     # [rows_pad, K/2]
    scales = scale_fp16.transpose(0, 1).contiguous()                 # [K/B, rows_pad]
    return packed, scales


def dequantize_int4_rows(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int = INT4_BLOCK_SIZE,
) -> torch.Tensor:
    """Inverse of `quantize_int4_rows`. Returns fp32 `[rows, K]`."""
    rows, K2 = packed.shape
    K = K2 * 2
    assert K % block_size == 0
    expected = (K // block_size, rows)
    assert tuple(scales.shape) == expected, (
        f"scales shape {tuple(scales.shape)} != expected {expected}"
    )
    nibs = _unpack_signed_nibbles(packed).to(torch.float32).view(rows, K)
    scale_per_block = scales.transpose(0, 1).to(torch.float32)       # [rows, K/B]
    return (nibs.view(rows, K // block_size, block_size) *
            scale_per_block.unsqueeze(-1)).view(rows, K)


# -------------------------------------------------------------------------
# nunchaku INT4 layout adapters
#
# nunchaku's CUDA INT4 path stores scales (and packed nibbles) in
# NVIDIA mma.sync `ldmatrix`-friendly permuted order so the GEMM kernel
# can issue mma.m16n8k64 without further reshuffling. Pure-PyTorch row-
# major layout is mathematically equivalent but ordered differently.
# These adapters convert nunchaku's permuted layout to/from row-major,
# so byte-/scale-level cross-validation against nunchaku is meaningful.
#
# The Ascend `mad_s4` cube fractal layout differs from BOTH nunchaku
# fragment and our row-major — hence the user's note "mmad ≠ mma". The
# row-major form is the canonical bridge: any backend's native layout
# can have its own pair of unpack/pack helpers that round-trip via row-
# major. See `csrc/kernels/gemm_w4a4/ascend/PLAN.md` for the Ascend pair.
# -------------------------------------------------------------------------

# Per nunchaku `gemm_base.cuh` (Config<INT4>):
#   WARP_M = 32, WARP_N = 128, WARP_SIZE = 32
#   ASCALES_PACK_SIZE = 2,  ASCALES_VALID_LANES = 16,  ASCALES_NUM_PACKS = 1
#   WSCALES_PACK_SIZE = 4,  WSCALES_VALID_LANES = 32,  WSCALES_NUM_PACKS = 1
_NUN_INT4_WARP_M = 32
_NUN_INT4_WARP_N = 128


def _ascales_perm_warp() -> list[int]:
    """Per-warp permutation: `out_idx → row_in_warp` for ascales (WARP_M=32).

    From `gemm_base.cuh::pack_ascales`:
        out_idx = laneId * 2 + pack_within_lane
        row     = (laneId // 8) * 16 + (laneId % 8) + pack_within_lane * 8

    Yields `[0, 8, 1, 9, ..., 7, 15, 16, 24, ..., 23, 31]`.
    """
    perm = []
    for out_idx in range(_NUN_INT4_WARP_M):
        lane_id = out_idx // 2
        pack = out_idx % 2
        row = (lane_id // 8) * 16 + (lane_id % 8) + pack * 8
        perm.append(row)
    return perm


def _wscales_perm_warp() -> list[int]:
    """Per-warp permutation: `out_idx → col_in_warp` for wscales (WARP_N=128).

    From `gemm_base.cuh::pack_wscales`:
        each lane writes 4 contiguous halves; lane laneId pulls from
        input[laneId/4 * 16 + laneId%4 * 2 + (within%2) + (within//2)*8]
        for within ∈ 0..3.
    """
    perm = []
    n_per_lane = 4  # WSCALES_PACK_SIZE
    for out_idx in range(_NUN_INT4_WARP_N):
        lane_id = out_idx // n_per_lane
        within = out_idx % n_per_lane
        col = (lane_id // 4) * 16 + (lane_id % 4) * 2 + (within % 2) + (within // 2) * 8
        perm.append(col)
    return perm


def _build_perm_tensor(
    perm_warp: list[int], full_extent: int, warp_extent: int, device: torch.device
) -> torch.Tensor:
    """Tile the per-warp permutation across the full M (or N) extent."""
    assert full_extent % warp_extent == 0, (full_extent, warp_extent)
    n_warps = full_extent // warp_extent
    out = []
    for w in range(n_warps):
        out.extend(w * warp_extent + i for i in perm_warp)
    return torch.tensor(out, dtype=torch.long, device=device)


def unpack_nunchaku_ascales(scales_nun: torch.Tensor) -> torch.Tensor:
    """Reorder nunchaku INT4 ascales `[K/G, M_pad]` to row-major along M.

    nunchaku's ascales' M axis is in mma.sync-friendly order
    (`ldmatrix`-fragment layout). After this call, `out[g, r]` holds the
    scale for input row `r` and K-block `g`, matching `quantize_int4_rows`.
    """
    G, M = scales_nun.shape
    perm = _build_perm_tensor(
        _ascales_perm_warp(), M, _NUN_INT4_WARP_M, scales_nun.device
    )
    # scales_nun[g, perm[r]] is the scale for logical row r → invert.
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(M, device=perm.device)
    return scales_nun.index_select(1, inv)


def unpack_nunchaku_wscales(scales_nun: torch.Tensor) -> torch.Tensor:
    """Reorder nunchaku INT4 wscales `[K/G, N]` to row-major along N."""
    G, N = scales_nun.shape
    perm = _build_perm_tensor(
        _wscales_perm_warp(), N, _NUN_INT4_WARP_N, scales_nun.device
    )
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(N, device=perm.device)
    return scales_nun.index_select(1, inv)


def pack_nunchaku_wscales(scales_logical: torch.Tensor) -> torch.Tensor:
    """Inverse of `unpack_nunchaku_wscales`: row-major `[..., N]` → fragment.

    Used to convert a row-major scale tensor (e.g. `smooth_factor` produced
    by the user / our pytorch ref) into the layout nunchaku's CUDA kernel
    expects when it loads via `load_wscale`. Required because nunchaku's
    public `smooth_factor` argument has docstring shape `(N,)` but the
    stored bytes are already in deepcompressor-calibrated fragment order.
    Passing a raw row-major tensor → kernel reads scrambled scales →
    wrong amax → wrong qout / oscales.
    """
    *lead, N = scales_logical.shape
    perm = _build_perm_tensor(
        _wscales_perm_warp(), N, _NUN_INT4_WARP_N, scales_logical.device
    )
    return scales_logical.index_select(-1, perm)


def pack_nunchaku_ascales(scales_logical: torch.Tensor) -> torch.Tensor:
    """Inverse of `unpack_nunchaku_ascales`: row-major `[K/G, M_pad]` → fragment.

    Symmetric with `pack_nunchaku_wscales`, but uses the M-axis (WARP_M=32)
    permutation. Useful when feeding ascales into `svdq_gemm_w4a4_cuda`
    after quantizing with our pytorch ref instead of the nunchaku quant op.
    """
    G, M = scales_logical.shape
    perm = _build_perm_tensor(
        _ascales_perm_warp(), M, _NUN_INT4_WARP_M, scales_logical.device
    )
    return scales_logical.index_select(1, perm)
