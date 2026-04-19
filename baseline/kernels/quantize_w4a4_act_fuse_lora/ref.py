"""Pure-PyTorch reference for `quantize_w4a4_act_fuse_lora` (NVFP4 path).

Produces the same three outputs as nunchaku's fused kernel:
    - qout        [M_pad, K/2]      uint8  (two NVFP4 E2M1 nibbles per byte)
    - oscales     [K/16,  M_pad]    fp8_e4m3fn  (per-16-K-block scale)
    - lora_act    [M_pad, R]        fp32

Readability over speed. No fused ops, no torch.compile.
"""
from __future__ import annotations

import torch

# NVFP4 E2M1 value set (positive magnitudes). Encoding = index in this list,
# sign bit in position 3.
_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
NVFP4_AMAX = 6.0
# FP8-E4M3 max finite value. Nunchaku clamps scales to this pre-cast
# (`gemm_w4a4.cuh:93`); we mirror for bit-for-bit agreement.
FP8_E4M3_MAX = 448.0


def _quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values in [-6, 6] to nearest NVFP4 level, return uint8 nibble.

    Not strictly RTNE — uses midpoint-then-floor, which matches hardware
    `cvt.rn` on the open intervals. Tie behaviour may differ on exactly
    the few rational midpoints (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0).
    """
    levels = _E2M1_LEVELS.to(x.device)
    abs_x = x.abs().clamp_max(NVFP4_AMAX)

    # idx = argmin |abs_x - level|, ties → smaller index. Computing via
    # thresholds so this stays vectorised and bit-reproducible.
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32, device=x.device,
    )
    idx = (abs_x.unsqueeze(-1) >= thresholds).sum(-1)  # int64 in [0, 7]
    sign_bit = (x < 0).to(torch.uint8) << 3
    nib = idx.to(torch.uint8) | sign_bit
    return nib


def _pack_nibbles(nibs: torch.Tensor) -> torch.Tensor:
    """Pack the last-dim pairs of 4-bit nibbles into uint8. Low nibble = even k."""
    assert nibs.shape[-1] % 2 == 0
    lo = nibs[..., 0::2]
    hi = nibs[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def quantize_w4a4_act_fuse_lora_ref(
    input: torch.Tensor,            # [M, K] fp16/bf16
    lora_down: torch.Tensor,        # [K, R] fp16/bf16
    smooth: torch.Tensor | None,    # [K]    fp16/bf16 or None
    *,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 reference. Matches nunchaku layout:
        qout    : [M_pad, K/2]   uint8
        oscales : [K/16,  M_pad] fp8_e4m3fn
        lora_act: [M_pad, R]     fp32
    """
    assert input.dtype in (torch.float16, torch.bfloat16)
    assert lora_down.dtype == input.dtype
    assert input.dim() == 2 and lora_down.dim() == 2
    M, K = input.shape
    K2, R = lora_down.shape
    assert K == K2
    assert K % 16 == 0, "NVFP4 group size is 16"

    M_pad = ((M + pad_size - 1) // pad_size) * pad_size

    # --- LoRA-down: unsmoothed X @ lora_down ---
    # Paper says X̂ @ L2^T; nunchaku absorbs diag(1/smooth) into stored
    # lora_down offline, so runtime uses raw X. The calibration pipeline
    # is the authority on what's in `lora_down` — we just take it as given.
    lora_act = input.to(torch.float32) @ lora_down.to(torch.float32)
    lora_act_pad = torch.zeros(M_pad, R, dtype=torch.float32, device=input.device)
    lora_act_pad[:M] = lora_act

    # --- Smooth divide + NVFP4 quantize ---
    x = input.to(torch.float32)
    if smooth is not None:
        x = x / smooth.to(torch.float32)

    x_pad = torch.zeros(M_pad, K, dtype=torch.float32, device=input.device)
    x_pad[:M] = x

    # Per-16-K-block amax → fp8_e4m3 scale.
    x_blocks = x_pad.view(M_pad, K // 16, 16)
    amax = x_blocks.abs().amax(dim=-1)                       # [M_pad, K/16]
    scale_f32 = (amax / NVFP4_AMAX).clamp(min=1e-12, max=FP8_E4M3_MAX)
    scale_fp8 = scale_f32.to(torch.float8_e4m3fn)            # [M_pad, K/16]
    scale_back = scale_fp8.to(torch.float32)                  # what the kernel actually uses

    # Quantize values with that scale.
    x_scaled = x_blocks / scale_back.unsqueeze(-1)            # [M_pad, K/16, 16]
    nibs = _quantize_e2m1(x_scaled)                           # uint8 in [0, 16)
    nibs = nibs.view(M_pad, K)
    qout = _pack_nibbles(nibs)                                # [M_pad, K/2]

    # nunchaku layout for oscales: transpose to [K/16, M_pad]
    oscales = scale_fp8.transpose(0, 1).contiguous()

    return qout, oscales, lora_act_pad
