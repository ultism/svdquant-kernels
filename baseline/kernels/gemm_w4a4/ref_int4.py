"""Pure-PyTorch reference for `gemm_w4a4` (INT4 / Ascend path).

Same op as `ref.py` (NVFP4 path) but the format primitive is signed
INT4 + per-64-K-block fp16 scales instead of NVFP4 + fp8 scales. The
math (scaled MMA + LoRA-up + bias + optional next-layer quant) is
identical — only `dequantize_*_rows` and the optional `quantize_*_rows`
on the epilogue swap.

Eats the canonical INT4 layout (block_size=64):

    act          [M_pad, K/2]           uint8     two INT4 nibbles per byte
    wgt          [N,     K/2]           uint8
    ascales      [K/64,  M_pad]         fp16      per-64-K-block act scale
    wscales      [K/64,  N]             fp16      per-64-K-block weight scale
    lora_act_in  [M_pad, R]             fp32      = previous-op output
    lora_up      [N,     R]             fp16/bf16
    bias         [N]                    fp16/bf16 (optional)
    wcscales     [N]                    fp16/bf16 (optional per-channel scale)
    smooth_next  [N]                    fp16/bf16 (optional; enables next-layer quant)

Output:
    out          [M_pad, N]             lora_up.dtype
    qout         [M_pad, N/2]           uint8     (None if smooth_next is None)
    oscales      [N/64,  M_pad]         fp16      (None if smooth_next is None)

All math in fp32; `out` is cast at the end. Readability over speed —
no fused ops, no torch.compile.
"""
from __future__ import annotations

import torch

from .._int4 import (
    INT4_BLOCK_SIZE,
    dequantize_int4_rows,
    quantize_int4_rows,
)


def make_int4_inputs(
    M: int,
    K: int,
    N: int,
    *,
    seed: int = 0xC0FFEE,
    amplitude: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic INT4-packed (act, wgt) + per-K-block fp16 scales.

    Returns `(act_packed, wgt_packed, ascales, wscales)` ready for the
    AscendC `gemm_w4a4` op:
        act_packed [M,         K/2]               uint8   nibbles
        wgt_packed [N,         K/2]               uint8
        ascales    [K/B, M]                       fp16
        wscales    [K/B, N]                       fp16

    Amplitude 0.5 matches `ref_mock.make_inputs`. After symmetric INT4
    quant the recovered fp32 values land in roughly the same dynamic
    range, so post-MMA magnitudes stay sane for fp16 epilogue casts.
    """
    g = torch.Generator().manual_seed(seed)
    act_fp32 = (torch.rand(M, K, generator=g) * 2 - 1) * amplitude
    wgt_fp32 = (torch.rand(N, K, generator=g) * 2 - 1) * amplitude
    act_packed, ascales = quantize_int4_rows(act_fp32, block_size=INT4_BLOCK_SIZE)
    wgt_packed, wscales = quantize_int4_rows(wgt_fp32, block_size=INT4_BLOCK_SIZE)
    return act_packed, wgt_packed, ascales, wscales


def gemm_w4a4_ref_int4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    wcscales: torch.Tensor | None = None,
    alpha: float = 1.0,
    smooth_next: torch.Tensor | None = None,
    block_size: int = INT4_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """INT4 reference. See module docstring for shapes."""
    assert act.dtype == torch.uint8 and wgt.dtype == torch.uint8
    assert ascales.dtype == torch.float16
    assert wscales.dtype == torch.float16
    M_pad, K2 = act.shape
    N, K2w = wgt.shape
    assert K2 == K2w, f"act/wgt K disagree: {K2} vs {K2w}"
    K = K2 * 2
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    R = lora_act_in.shape[1]
    assert lora_act_in.shape == (M_pad, R)
    assert lora_up.shape == (N, R)
    assert ascales.shape == (K // block_size, M_pad)
    assert wscales.shape == (K // block_size, N)

    # --- Main matmul in fp32 ---
    act_fp32 = dequantize_int4_rows(act, ascales, block_size=block_size)  # [M_pad, K]
    wgt_fp32 = dequantize_int4_rows(wgt, wscales, block_size=block_size)  # [N,     K]
    y = (act_fp32 @ wgt_fp32.T) * alpha                                    # [M_pad, N]

    # --- LoRA-up residual ---
    y = y + lora_act_in.to(torch.float32) @ lora_up.to(torch.float32).T

    # --- per-channel affine ---
    if wcscales is not None:
        assert wcscales.shape == (N,)
        y = y * wcscales.to(torch.float32)
    if bias is not None:
        assert bias.shape == (N,)
        y = y + bias.to(torch.float32)

    out_dtype = lora_up.dtype
    y_out = y.to(out_dtype)

    # --- optional next-layer INT4 quantize ---
    qout: torch.Tensor | None = None
    oscales: torch.Tensor | None = None
    if smooth_next is not None:
        assert smooth_next.shape == (N,)
        assert N % block_size == 0, f"N={N} not divisible by block_size={block_size}"
        y_for_quant = y / smooth_next.to(torch.float32)
        qout, oscales = quantize_int4_rows(y_for_quant, block_size=block_size, pad_size=1)

    return y_out, qout, oscales
