# gemm_w4a4 (CuTe DSL, CUDA)

Main SVDQuant W4A4 linear on Blackwell SM_100 / SM_103 — NVFP4 scaled
MMA (`tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X`) + LoRA
low-rank residual (β interleaved into the main K-loop) + optional
per-channel affine and next-layer NVFP4 quantize.

**Design**: `docs/kernels/gemm_w4a4.md` (ten sections covering β
justification, interleave pattern, tmem / smem budget, warp roles,
tiler choice, epilogue, skeleton source strategy, staged rollout).

**Contract**: `launch(...)` in `kernel.py`. Torch tensors at the host
boundary; NVFP4 is the uint8-packed layout produced by the preceding
`triton_kernels/quantize_w4a4_act_fuse_lora/` op.

**Reference math**: `baseline/kernels/gemm_w4a4/ref.py` (pure PyTorch
fp32 ground truth — what this kernel must match per `tmp/smoke_gemm.py`).

**Staging**:

| version | scope                                                        |
| ------- | ------------------------------------------------------------ |
| v0      | main NVFP4 only, no LoRA, no wcscales, no bias               |
| v1      | + LoRA β-interleaved per design §2                           |
| v2      | + per-col `* wcscales + bias` epilogue                       |
| v3      | + optional next-layer NVFP4 quantize                         |

Currently at v1 (task #34).

**Reference skeleton**: `tmp/cutlass/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py`
— stable-API scaled-MMA + persistent scheduler. The 1021-line
`experimental/blackwell/dense_block_scaled_gemm.py` is cleaner but
its `cute.experimental` API is hard-gated to CUDA 13.1+; we're on
CUDA 13.0 (both local and Modal's B200 image), so we base off the
stable persistent example.
