"""Phase 2 mock reference — used by the Ascend smoke harness only.

Phase 2 of the Ascend `gemm_w4a4` pod brings up cube + vec + FFTS
plumbing without yet wiring real INT4 dequant. The device-side math
collapses to:

    cube_tile = act_fp16 @ wgt_fp16.T          (fp16 inputs, fp32 accum)
    vec_out   = cube_tile * scale + cube_tile  (= (1+scale) * cube_tile)

with `scale` a constant baked into the kernel (Phase 2c/2d use 0.5,
giving the `1.5 * matmul` mock). Phase 3 swaps this for the real
INT4 dequant + ascales/wscales + lora_up + bias formula in `ref_int4.py`;
this file disappears at that point.

NumPy-only by design: the GitCode Space container has Gradio (and
therefore numpy) pre-installed; PyTorch is not guaranteed there.
Local development can substitute a torch-based variant without
changing the smoke harness contract — the harness only ever sees
the resulting fp16/fp32 byte buffers.

The Ascend kernel runs the same `act`/`wgt` through `kNumTiles` ring
iterations, so all `kNumTiles` vec_out segments must equal `vec_out`
element-wise — the smoke harness diffs every segment against this
single reference tensor.
"""
from __future__ import annotations

import numpy as np


def mock_gemm(
    act: np.ndarray,
    wgt: np.ndarray,
    scale: float = 0.5,
) -> np.ndarray:
    """Phase 2 mock reference.

    Args:
        act: [M, K] fp16 — activation tile. Same buffer the cube reads
            via TLOAD on every ring iteration.
        wgt: [N, K] fp16 — weight tile (storage already in the
            "K is the reduction axis" orientation, so the math is
            `act @ wgt.T`). Same convention as `ref.py` / `ref_int4.py`.
        scale: vec-pipe TROWEXPANDMUL scalar (kVecScale on device).
            Default 0.5 matches Phase 2c/2d.

    Returns:
        [M, N] fp32 — vec_out reference. fp32 because the device-side
        cube accumulator is fp32 and TSTOREs fp32 to the GM ring; vec
        consumes that fp32 tile.
    """
    assert act.dtype == np.float16, f"act must be fp16, got {act.dtype}"
    assert wgt.dtype == np.float16, f"wgt must be fp16, got {wgt.dtype}"
    assert act.ndim == 2 and wgt.ndim == 2
    M, K = act.shape
    N, Kw = wgt.shape
    assert K == Kw, f"act/wgt K disagree: {K} vs {Kw}"

    # Cast to fp32 before matmul so the reduction matches the device's
    # fp32 accumulator. NumPy's matmul on fp32 inputs uses fp32 BLAS,
    # which is the closest portable analogue to the cube's fp32 accum.
    cube = act.astype(np.float32) @ wgt.astype(np.float32).T  # [M, N]
    return cube * scale + cube


def make_inputs(
    M: int,
    K: int,
    N: int,
    *,
    seed: int = 0xC0FFEE,
    amplitude: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic act/wgt seed. Uniform [-amplitude, amplitude] fp16.

    Amplitude 0.5 keeps `act @ wgt.T` magnitude near 1.0 for K=128
    (per-row sum of 128 products of ~U[-0.5, 0.5] has std ~0.94),
    well clear of fp16 saturation and well above the fp16 noise floor.
    """
    rng = np.random.default_rng(seed)
    act = (rng.uniform(-amplitude, amplitude, size=(M, K))).astype(np.float16)
    wgt = (rng.uniform(-amplitude, amplitude, size=(N, K))).astype(np.float16)
    return act, wgt
