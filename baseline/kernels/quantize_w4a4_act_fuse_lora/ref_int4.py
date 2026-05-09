"""Pure-PyTorch reference for `quantize_w4a4_act_fuse_lora` (INT4 / Ascend path).

Same op as `ref.py` (NVFP4) but the format primitive is signed INT4 +
per-64-K-block fp16 scales. The LoRA-down projection and the optional
smooth-divide are unchanged.

Outputs:
    qout      [M_pad, K/2]   uint8       two INT4 nibbles per byte
    oscales   [K/64,  M_pad] fp16        per-64-K-block scale
    lora_act  [M_pad, R]     fp32
"""
from __future__ import annotations

import torch

from .._int4 import INT4_BLOCK_SIZE, quantize_int4_rows


def quantize_w4a4_act_fuse_lora_ref_int4(
    input: torch.Tensor,            # [M, K] fp16/bf16
    lora_down: torch.Tensor,        # [K, R] fp16/bf16
    smooth: torch.Tensor | None,    # [K]    fp16/bf16 or None
    *,
    pad_size: int = 256,
    block_size: int = INT4_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """INT4 reference. Shapes in module docstring."""
    assert input.dtype in (torch.float16, torch.bfloat16)
    assert lora_down.dtype == input.dtype
    assert input.dim() == 2 and lora_down.dim() == 2
    M, K = input.shape
    K2, R = lora_down.shape
    assert K == K2

    M_pad = ((M + pad_size - 1) // pad_size) * pad_size

    # --- LoRA-down (unsmoothed; nunchaku absorbs diag(1/smooth) into stored
    #     lora_down offline). The calibration pipeline owns what's in
    #     `lora_down`; runtime takes it as-is.
    lora_act = input.to(torch.float32) @ lora_down.to(torch.float32)
    lora_act_pad = torch.zeros(M_pad, R, dtype=torch.float32, device=input.device)
    lora_act_pad[:M] = lora_act

    # --- Smooth divide + INT4 quantize ---
    x = input.to(torch.float32)
    if smooth is not None:
        x = x / smooth.to(torch.float32)
    qout, oscales = quantize_int4_rows(x, block_size=block_size, pad_size=pad_size)

    return qout, oscales, lora_act_pad
