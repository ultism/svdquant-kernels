# CUDA backend

## Target SMs

Declared in `cmake/cuda_arch.cmake`. Scope is intentionally narrow:

| SM   | Arch      | Representative parts     |
|------|-----------|--------------------------|
| 100  | Blackwell | B100 / B200              |
| 103  | Blackwell | data-center Blackwell variants |

Everything else (Turing through Hopper, plus consumer Blackwell
SM_120a/121a) is covered by `nunchaku`; this repo exists to fill the
SM_100/SM_103 gap, not duplicate that work. See
`tmp/nunchaku/setup.py:41-64` for nunchaku's arch list.

Override with `-DSVDQUANT_CUDA_ARCHS="100;103"`. Each listed arch
also gets a `SVDQUANT_HAS_SM<N>=1` compile define, so files can opt
in per arch without the build system knowing about them.

## Per-pod layout

```
csrc/kernels/<op>/cuda/
    kernel.cu           # top-level launcher; dispatches by capability
    sm100.cu            # (added when real kernels land)
    sm103.cu
```

The scaffold only ships `kernel.cu` with a host-side stub; per-SM
variants land as real implementations arrive. Real kernels on this
path use **CuTe DSL** (CUTLASS 3.x) for `tcgen05.mma` scaled-MMA
variants — that's what B200's tensor cores speak.

## Build

```
CUDA=ON ASCEND=OFF ./scripts/build.sh
```

or directly:

```
cmake -S . -B build -G Ninja \
    -DSVDQUANT_ENABLE_CUDA=ON \
    -DSVDQUANT_ENABLE_ASCEND=OFF
cmake --build build
```

## Conventions

- Launch signatures take `void* stream` rather than `cudaStream_t` to
  keep the header free of CUDA includes — cast inside `kernel.cu`.
- `TensorRef::data` is a raw device pointer (`T*` cast from
  `cudaMalloc`/PyTorch storage).
- Kernels in `csrc/kernels/` should use CUTLASS 3.x / CuTe DSL
  primitives; bespoke hand-rolled CUDA is for shapes CuTe can't
  cover well.

## When to pick Triton instead

If an op is memory-bound on B200 (AI well below the ~281 FLOP/B FP16
tensor-core ridge) AND needs to also run on Ascend NPU, put it under
`triton_kernels/<op>/` instead — one `kernel.py` runs on both
backends (upstream Triton for CUDA, `triton-ascend` for NPU). See
`../triton_kernels/README.md` for the library-choice rule.

## Gotchas (CuTe DSL traps)

Silent-misbehavior traps on the SM_100 / SM_103 CuTe DSL path —
`const_expr` and `if`, divide-API nesting differences, 2-CTA
`cluster_layout_vmnk` axes, 2-CTA `TiledCopy.partition_D`
rest-mode trap, `num_acc_stage` vs `tile_n` interaction. See
[gotchas/cute_dsl.md](./gotchas/cute_dsl.md). Add new entries
there as you find them.

## Perf-comparison context

### nunchaku is hand-written PTX, not CuTe / CUDA C++

nunchaku's NVFP4 / INT4 scaled-MMA mainloop uses inline `asm
volatile` PTX (`tmp/nunchaku/src/kernels/zgemm/mma_earlycuda.cuh`),
not `cute::gemm` or CUTLASS templates. Register packing, scale
extraction, operand alignment are all manual. It's effectively a
*tuned-for-generation reference* (GB202 / sm_120a era).

When comparing against nunchaku numbers from our CuTe DSL kernels:

- Don't expect apples-to-apples efficiency. The compiler gap is
  real. The last 5-10 pp typically lives in register-allocation
  and instruction-scheduling decisions the PTX author makes
  explicitly but the DSL / MLIR lowering does generically.
- Single-digit pp behind = competitive. 15+ pp behind = something
  structural on our side, not just codegen.
- bf16 vs fp16 asymmetry is typically larger on hand-PTX kernels
  (different mma PTX ops, register banks, swizzle patterns) than
  on DSL output, which goes through the same MLIR path with dtype
  substitution.

Does not apply to the Triton pod
(`quantize_w4a4_act_fuse_lora`) — both sides go through Triton
MLIR, so codegen gap is narrower; wins / losses are about kernel
design, not PTX craft.

### Blackwell NVFP4 routes to `hmma` subpipe in ncu, not a `qmma` subpipe

NVFP4 scaled-MMA on gb100 / B200 (sm_100/103) is **UTCQMMA** at
the SASS level, but ncu's metric tree puts it on the **hmma**
subpipe. There is no standalone `qmma_*` counter — don't waste
time searching by that name.

Useful metrics (queried via `ncu --query-metrics --chips gb100`):

- Pipe util:
  `sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active`
  (covers HMMA + UTCHMMA + UTCQMMA + UTCOMMA all together).
- FLOPs, FP32 accumulator (TMEM):
  `sm__ops_path_tensor_op_utcqmma_src_fp4_fp6_fp8_dst_fp32`.
- FLOPs, FP16 accumulator:
  `sm__ops_path_tensor_op_utcqmma_src_fp4_fp6_fp8_dst_fp16`.
- Separate FP4-only path (UTCOMMA, different from QMMA):
  `sm__ops_path_tensor_op_utcomma_src_fp4_dst_fp32`.

`--section ComputeWorkloadAnalysis` auto-pulls the subpipe
breakdown — look for "Tensor" rows in SOL / CWA. UTCQMMA work
shows up under "HMMA Pipe" in the SOL "Compute (SM) Pipe
Utilization" panel.

## v2_fa4 baseline on B300 (Verda, 2026-05-12)

First real-hardware run of `kernel_v2_fa4` after the C2 patch
(`8f91240` — defer `pipeline_lora.consumer_wait` into the K-loop
inject site). Host: 2× B300 SXM6 AC, sm_103, ncu unrestricted.

Correctness: `tmp/smoke_gemm_v2_fa4.py` 48/48 pass across
{fp16, bf16} × {1-CTA, 2-CTA} × {wcscales on/off} × {bias on/off}
× R∈{32, 128}, fp16 rel ≤ 8e-4, bf16 rel ≤ 7e-3.

Production-shape TFLOPS (`tmp/bench_gemm_v2_fa4_c1.py`, fp16, 2-CTA):

| M    | K     | N     | R   | TFLOPS  | MFU / 13.5 PFLOPS |
|------|-------|-------|-----|---------|-------------------|
|  256 |  3840 |  3072 | 128 |    35   |    0.3%           |
| 4352 |  3840 |  3072 | 128 |   566   |    4.2%           |
| 4352 |  3840 | 15360 | 128 |  1881   |   13.9%           |
| 4352 | 15360 |  3840 | 128 |  1864   |   13.8%           |
| 4352 | 10240 |  3072 |  32 |  1530   |   11.3%           |

(MFU normalized to a B300 NVFP4 dense peak of 13.5 PFLOPS, ~1.35×
B200. The benched MFU printed by the script uses the B200 10 PFLOPS
constant and overstates by 1.35×.)

LoRA pipeline ladder (M=4352 K=3840 N=3072 R=128 fp16 2-CTA), via
`tmp/profile_gemm_v2_fa4.py --num-lora-stage 0|1|2` under ncu
SpeedOfLight:

| Stage          | Duration | SM%  | Mem% | DRAM% | L2%  |
|----------------|---------:|-----:|-----:|------:|-----:|
| 0 LoRA off     |  44.6 µs | 53.5 | 42.7 |  4.7  | 33.3 |
| 1 pre-C1       |  83.1 µs | 56.0 | 23.6 |  2.8  | 18.9 |
| 2 C1 (+C2 on)  |  70.2 µs | 46.6 | 28.2 |  3.4  | 21.5 |

C1 win (1-stage → 2-stage LoRA prolog): −12.9 µs / −15.6 %.
Reports kept at `log/verda_ncu_v2_C2_stage{0,1,2}_4352_3840_3072_R128.ncu-rep`.

C2 is enabled in all three stages above (it's a code change, not a
runtime knob), so its isolated win is not yet captured. To measure:
swap in pre-C2 `kernel_v2_fa4.py` (`git show 8f91240^:...`), rerun
stage=2 ncu, diff Duration.
