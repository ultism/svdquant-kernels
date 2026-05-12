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

### C2 standalone win (pre-C2 vs C2, both at stage=2)

Swapped `kernel_v2_fa4.py` between `8f91240^` and `8f91240` while
holding everything else constant, same shape and ncu flags:

| Metric           | pre-C2  | C2      | Δ            |
|------------------|--------:|--------:|-------------:|
| Duration         | 71.17 µs | 70.18 µs | -0.99 µs / -1.4 % |
| Compute (SM) %   | 45.00   | 46.55   | +1.55 pp     |
| L2 Cache %       | 20.50   | 21.48   | +0.98 pp     |
| Memory %         | 28.40   | 28.21   | ≈            |
| DRAM %           |  3.31   |  3.35   | ≈            |
| SM Active cycles | 63549   | 64068   | +0.8 %       |

Story is clean: deferring `pipeline_lora.consumer_wait` lets the MMA
warp start issuing main atom #0 ~1 µs before LA/LU TMA arrives. The
saved cycles surface as +1.55 pp SM throughput. Memory side is
unchanged — C2 is a scheduling change, not a bandwidth change.

Reports: `log/verda_ncu_v2_{preC2,C2}_stage2_4352_3840_3072_R128.ncu-rep`.

Reproduction script (uses an EXIT trap to guarantee the C2 file is
restored even on ncu failure): `tmp/verda_c2_ab.sh`.

## v2_fa4 SMEM budget at the production shape (B300, 2026-05-12)

Probed on Verda via a `print` injected into `_compute_stages`. All
numbers below are per-CTA, occupancy=1, tile=(256, 128, 64), R=128,
fp16 ab/c, fp4 mma a/b, fp8 sf.

| Component                            | Bytes  | KB  |
|--------------------------------------|-------:|----:|
| SMEM capacity (sm_100 == sm_103)     | 232448 | 227 |
| `ab_bytes` per stage (A+B+SFA+SFB)   |  28672 |  28 |
| `c_bytes_per` per epi stage          |   8192 |   8 |
| `mbar_helpers`                       |   1024 |   1 |
| `LA` per CTA (`tile_m*R/cta_group`)  |  32768 |  32 |
| `LU` per CTA (`tile_n*R`)            |  32768 |  32 |
| per-stage LoRA = LA+LU               |  65536 |  64 |

Stage-by-stage feasibility on this shape:

| num_lora_stage | LoRA  | c(2)  | ab budget | ab_stages | fit?  |
|---------------:|------:|------:|----------:|----------:|:------|
|             2  | 128 K | 16 K  |   82 K    |     2     | yes   |
|             3  | 192 K | 16 K  |   18 K    |     0     | **assert** |
|             3  | 192 K | 8 K (c=1) | 26 K  |    0     | still no |

The headroom for a 3rd LoRA stage is one full LoRA stage short:
each costs 64 KB but only ~26 KB of slack exists after c_stage=1.
Naive stage=3 doubles LoRA SMEM (128 KB → 192 KB), which violates
the "without doubling" constraint of task #58 anyway.

> **2026-05-13 follow-up: the LU row above is wrong by 2×.** The
> handwritten `lu_bytes` formula treated LU as full N=128 per CTA,
> but the 2-CTA dense MMA atom **halves LU via N-split** inside
> `partition_shape_B` (same mechanism that halves main B). Real LU
> per CTA = 16 KB / stage, not 32 KB. See the next section for the
> probe, the fix, and the much larger win it unlocked. The
> "paths to stage=3" list above is preserved for context, but is
> now mooted — stage=3 became feasible with no code redesign, and
> the bench in the next section shows it is also no longer the
> right knob to tune.

The probe artifact lives at `tmp/probe_smem_budget.py` and the
inline `_compute_stages` print used to capture the numbers above
was reverted in this commit.

## v2_fa4 LU SMEM accounting fix (B200, 2026-05-13)

The handwritten `lora_smem_bytes` in `_setup_attributes` over-counted
LU by 2× — `_compute_stages` therefore reserved double the LoRA SMEM
it needed, and `num_ab_stage` was clamped to 2 instead of 4 at the
R=128 production shape. This was a single-line bug that *silently
hid the real perf headroom* behind a misleading SMEM-budget message.

### Probe (task #96)

Injected `cute.cosize(slice_(lu_smem_layout_staged, ...))` into
`_setup_attributes` so the actual per-stage byte count surfaces at
trace time:

```
[PROBE96] num_lora_stage=2 cta_group_size=2
[PROBE96] la_one cosize=16384 -> 32768 B (handwritten 32768 B, factor 1.000)
[PROBE96] lu_one cosize=8192  -> 16384 B (handwritten 32768 B, factor 0.500)
```

LA matches (M-split was already correct in the handwritten formula).
LU is half — confirms the Modular blog claim (Part 3, "2xSM MMA: Shared
Memory Optimization") that the 2xSM atom halves the B tile via
`partition_shape_B`. The fix is one extra `// self.cta_group_size`
on the `lu_bytes` line; comment in
`cute_kernels/gemm_w4a4/kernel_v2_fa4.py::_setup_attributes` cites
this section.

### Re-solved budget (R=128, fp16, 2-CTA, tile=(256, 128))

| Component                            | Bytes  | KB  |
|--------------------------------------|-------:|----:|
| SMEM capacity (sm_100 == sm_103)     | 232448 | 227 |
| `LA` per CTA (M-split)               |  32768 |  32 |
| `LU` per CTA (**N-split, was 32**)   |  16384 |  16 |
| per-stage LoRA = LA+LU               |  49152 |  48 |
| `ab_bytes` per stage                 |  28672 |  28 |
| `c_bytes_per` per epi stage          |   8192 |   8 |

Feasibility per `num_lora_stage`:

| stage | LoRA  | c stages chosen | ab stages chosen | fit? |
|------:|------:|----------------:|-----------------:|:-----|
|     2 |  96 K |               2 |            **4** | yes  |
|     3 | 144 K |               3 |                2 | yes  |
|     4 | 192 K |               1 |                1 | assert |

The pre-fix code thought stage=2 had only 2 ab_stages of headroom and
stage=3 didn't fit at all. Post-fix, stage=2 lands at ab=4 and stage=3
becomes solvable too.

### Wall-clock impact at stage=2 (the actual main path)

Comparing the same `tmp/bench_gemm_v2_fa4_c1.py` shapes pre-fix
(B300, doc'd) vs post-fix (B200, fresh run, fp16, 2-CTA):

| M    | K     | N     | R   | pre-fix TF (B300) | post-fix TF (B200) | Δ          |
|------|-------|-------|-----|------------------:|-------------------:|-----------:|
|  256 |  3840 |  3072 | 128 |              35   |              **100** | +186%    |
| 4352 |  3840 |  3072 | 128 |             566   |             **1532** | +171%    |
| 4352 |  3840 | 15360 | 128 |            1881   |             **2720** | +45%     |
| 4352 | 15360 |  3840 | 128 |            1864   |             **2733** | +47%     |
| 4352 | 10240 |  3072 |  32 |            1530   |             **2623** | +71%     |

(Numbers are absolute TF and so cross-card comparable; B300 has 1.35×
more peak NVFP4 than B200, so a "same TF" reading would still mean we
got faster against a weaker card. Logs:
`log/verda_bench_lufix.log` (post-fix bench),
`log/verda_smoke_lufix.log` (48/48 smoke pass at the new ab=4).)

### Stage sweep — `num_lora_stage` is no longer the bottleneck

Post-fix wall-clock sweep at M=4352 K=3840 N=3072 R=128 fp16 2-CTA
(`tmp/bench_gemm_lora_stage_sweep.py`, 200 iter, CUDA-event timing):

| stage | µs/launch | TFLOPS | (num_ab, num_lora, num_c) | vs stage=2 |
|------:|----------:|-------:|--------------------------:|-----------:|
|     0 |     51.82 |   1981 |                   (7, 0, 3) | −10.76 µs / −17.2 % |
|     1 |     86.36 |   1189 |                   (5, 1, 4) | +23.78 µs / +38.0 % |
| **2** | **62.58** | **1641** |               **(4, 2, 2)** | (baseline) |
|     3 |     73.10 |   1405 |                   (2, 3, 3) | +10.52 µs / +16.8 % |

Stage=3 is *feasible* but **slower**: the solver buys the extra LoRA
prolog by giving up two main `num_ab` stages, and the main K-loop
loses more than the LoRA prolog gains. This kills tasks #58 (deepen
prolog) and #59 (multicast LoRA TMA) as wins — both were proposed
under the false-assumption regime; the real ceiling now sits in main
K-loop / TMEM occupancy, not LoRA-side latency hiding.

LoRA overhead at the new baseline: 62.58 − 51.82 = 10.76 µs / +20.8 %
on top of the LoRA-off path. That delta is what tasks #60 (overlap
LoRA MMA with main K-loop epilogue tail) and future work would
target, not LoRA prolog depth.

Log: `log/verda_lora_stage_sweep.log`.

### ncu A/B at the production shape (same B200, same launch config)

Reports captured 2026-05-13 on the same Verda B200 instance: HEAD^
(pre-LU-fix, `num_ab=2`) vs HEAD (`7296e90`, post-LU-fix, `num_ab=4`).
Same shape, same launch flags, same `num_lora_stage=2`. The kernel was
swapped on-disk between runs (the script ships with an EXIT trap to
guarantee restore on failure — `tmp/verda_lufix_ncu_ab.sh`).

| Metric                  | pre-LU-fix | post-LU-fix | Δ                  |
|-------------------------|-----------:|------------:|-------------------:|
| Duration                |  46.69 µs  |   32.13 µs  | **−14.56 µs / −31.2 %** |
| Compute (SM) %          |  41.63     |   53.62     | **+11.99 pp**      |
| Memory %                |  25.58     |   38.91     | +13.33 pp          |
| L1/TEX Cache %          |  28.50     |   44.75     | +16.25 pp          |
| L2 Cache %              |  24.57     |   36.18     | +11.61 pp          |
| DRAM %                  |   5.04     |    7.31     | +2.27 pp           |
| SM Active Cycles        |  72 433    |   46 126    | **−36.3 %**        |
| Memory Throughput       |   386 GB/s |    561 GB/s | +45 %              |
| Achieved Occupancy      |    8.55 %  |     8.66 %  | ≈                  |
| Grid Size / Block Size  | 148 / 192  |  148 / 192  | identical          |

Reads consistent with the budget story: same launch shape (148 ×
192-thread blocks, ~8.6 % occupancy), 2× more `num_ab` stages keep the
SM-side pipeline fed → SM% jumps +12 pp and SM Active Cycles drop 36 %.
L1/TEX and L2 throughput both rise proportionally because the TMA
producers now have more in-flight in-flight buffers to fill (it's not a
"bandwidth saving" — it's the bandwidth being more *evenly used* across
the kernel's wall-time). DRAM stays low (compute-bound regime
preserved).

The ncu single-launch Duration (32.13 µs) is lower than the bench-side
CUDA-event average (62.58 µs / iter): the bench averages over a tight
200-iter Python loop with `cute_dsl` launch overhead included; the ncu
report measures just the device-side kernel. Both directions agree;
treat the bench number as "kernel + launch tax" and the ncu number as
"kernel only."

Reports kept at
`log/ncu_v2_{preLUfix,postLUfix}_4352_3840_3072_R128.ncu-rep` and the
text excerpt at `log/verda_ncu_lufix_ab.log`.
