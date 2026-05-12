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
| ~~v3~~  | ~~+ optional next-layer NVFP4 quantize~~ — **dropped**       |

v3 dropped 2026-04-24. nunchaku only exercises `smooth_factor`+`qout`
via `fused_gelu_mlp` (Flux v2 MLP, fc1=GELU), which requires the frame
to pass fc2's `smooth_factor` into fc1's gemm call — same kind of
pipeline-intrusive fusion that CLAUDE.md excludes (cf. `fuse_glu`).
Not reachable under our drop-in-at-linear-boundary API.

Currently at v2 (task #35, terminal). `kernel_v0_fa4.py` is frozen as
the v0/v1 reference (flag-gated on `enable_lora`); `kernel_v2_fa4.py`
is the post-processing fork (v1 + per-col `* wcscales + bias` in the
epilogue) and the shipping surface.

**Reference skeleton**: Flash-Attention 4 (FA4) Blackwell forward in
`tmp/flash-attention/flash_attn/cute/`:

- `flash_fwd_sm100.py` — warp-specialized mainloop (separate `load`,
  `mma`, epilogue methods). The mapping: our `mma` inherits the
  two-MMA-in-one-warp pattern from FA4's `tiled_mma_qk + tiled_mma_pv`
  signature, but points both MMA ops at the **same** tmem acc region
  (β accumulation) instead of FA4's chained S→P→O dataflow.
- `pipeline.py` — `PipelineStateSimple` (single `_phase_index`
  counter, `% stages` / `// stages` properties); `_w_index_phase`
  mixin so each warp drives its own state; `PipelineTmaUmma`
  override adds `extra_tx_count` (multiple TMAs share one barrier)
  and `is_leader_cta` gate (2CTA-aware).
- `tile_scheduler.py::StaticPersistentTileScheduler` — grid clamped
  to `sm_count`, `tile_idx += grid_dim()` advance. Dead simple.
- `blackwell_helpers.py::gemm` / `gemm_ptx_partial` — `zero_init`
  controls first-iter ACCUMULATE predicate; `gemm_ptx_partial`
  takes raw `acc_tmem_addr: Int32` so two MMAs can target the
  same tmem region (enables the β interleave without aliasing
  through a `cute.Tensor`).
- `AI/DEBUG_2CTA.md` — debugging guide that directly lists the
  2CTA-specific footguns (tx_count ×`cta_group_size`, phase parity,
  `producer_tail` deadlock, `tcgen05.commit` empty groups).

Why swap from CUTLASS's `dense_blockscaled_gemm_persistent.py`: that
example's implicit `PipelineState` is single-dimension and assumes
1-tile-per-CTA. Our state space is 5-dimensional (pipeline stages ×
2CTA pair barriers × persistent tile loop × LoRA β second MMA ×
epilogue correction chain). Implicit state handles dimension 1. A
prior persistent port passed correctness at 1-tile-per-CTA but hung
500× when each CTA processed ~20 tiles (see commit `61905df`) —
classic signature of phase/state drifting across tile boundaries.
FA4's explicit per-warp `PipelineStateSimple` driven via
`_w_index_phase` decouples pipeline state from kernel boundaries,
so persistent iteration composes cleanly.

## Architecture (FA4-derived 3-pipeline / 3-warp)

### Warp roles

| warp       | role                                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------------------- |
| `load`     | TMAs for `act + ascales + wgt + wscales`; v1+ also `lora_act_in + lora_up`.                           |
| `mma`      | main NVFP4 scaled MMA; v1+ also LoRA β FP16 MMA into the **same** tmem acc (no chained dependency).   |
| `epilogue` | v0/v1 stub: tmem → gmem copy. v2+: `* wcscales + bias`. v3+: requantize to NVFP4 for next layer.      |

### Pipelines

| pipeline        | class              | stages | producer → consumer           | notes                                                                |
| --------------- | ------------------ | ------ | ----------------------------- | -------------------------------------------------------------------- |
| `pipeline_aw`   | `PipelineTmaUmma`  | 3–4    | `load` → `mma`, per-K-block   | `act + ascales + wgt + wscales` share one barrier via `extra_tx_count`. |
| `pipeline_lora` | `PipelineTmaUmma`  | 1      | `load` → `mma`, per-tile      | v1+; `lora_act_in + lora_up` together (R ≤ 128, fits in one stage).  |
| `pipeline_acc`  | `PipelineUmmaAsync`| 1      | `mma` → `epilogue`, per-tile  | single-stage: bare `producer_acquire_w_index_phase` replaces tail.   |

### Pipeline state convention

Each warp holds its own state, advances explicitly (FA4 `mma()` line
1614-1618 pattern):

```python
# load warp
aw_producer_state  = make_pipeline_state(Producer, k_stage)
lora_producer_phase = Int32(1)         # single-stage producer starts at 1

# mma warp
aw_consumer_state  = make_pipeline_state(Consumer, k_stage)
lora_consumer_phase = Int32(0)         # single-stage consumer starts at 0
acc_producer_phase  = Int32(0)

# epilogue warp
acc_consumer_phase  = Int32(0)
```

State advances never reset at tile boundaries — the persistent `while
work_tile.is_valid_tile:` loop just keeps incrementing. This is the
whole point of the FA4 explicit-state pattern.

### 2CTA conventions

- `tx_count` in `PipelineTmaUmma.create(...)` **must** be computed
  with `cta_group_size` multiplier (both CTAs' TMAs sign the same
  cluster barrier). Baked in at pipeline creation time, not runtime.
- `is_leader_cta` gates `arrive_and_expect_tx` — only leader CTA in
  the cluster calls it; the barrier sees both CTAs' TMA contributions
  against a single tx_count threshold.
- `producer_tail` stays as-is for multi-stage `pipeline_aw`. For
  single-stage `pipeline_lora` / `pipeline_acc`, use bare
  `producer_acquire_w_index_phase(0, phase)` at kernel end (FA4
  `load()` line 1505-1506 pattern) — default `producer_tail` tries
  to acquire an already-drained slot and deadlocks under 2CTA.

### What carries over from current `kernel.py`

- NVFP4 SF atom repack (`repack_sf_to_cutlass_atom`).
- `make_blockscaled_trivial_tiled_mma(...)` atom selection.
- tmem / smem layout helpers, per-tile allocators.
- Host-side `launch(...)` signature and torch-tensor boundary.

### What gets rewritten

- Device body split into `load()`, `mma()`, `epilogue()` warp-specialized
  methods (replacing the monolithic `@cute.kernel`).
- Pipeline creation (3 explicit `PipelineStateSimple`-driven pipelines).
- Persistent tile iteration via `StaticPersistentTileScheduler.{get_current_work,
  advance_to_next_work}`.

## Baseline — CUTLASS NVFP4 on same B200 / same shapes

This is the honest ceiling. CUTLASS's own `dense_blockscaled_gemm_
persistent.py` (main NVFP4 MMA, no LoRA / no epilogue scale / no
next-quant — strictly the same op our v0 does) vs our kernel on
`GEMM_SHAPES` in fp16 out. Run with `modal run
scripts/modal_app.py::cutlass_nvfp4_bench`. MFU vs 10 PFLOPS B200
dense NVFP4 peak.

| shape (M, K, N)       | CUTLASS 1-CTA 128×256 | CUTLASS 2-CTA 256×128 | CUTLASS 2-CTA 256×256 | ours 1-CTA        | ours 2-CTA Phase 1 |
| --------------------- | --------------------- | --------------------- | --------------------- | ----------------- | ------------------ |
|  256 × 3840 × 3072    |   564 TF  5.6%        |   734 TF  7.3%        |   588 TF  5.9%        |    98 TF  1.0%    |   100 TF  1.0%     |
| 4352 × 3840 × 3072    |  3847 TF 38.5%        |  4202 TF 42.0%        |  4545 TF 45.4%        |  1309 TF 13.1%    |  1185 TF 11.8%     |
| 4352 × 3840 × 15360   |  4167 TF 41.7%        |  5181 TF 51.8%        |  5836 TF 58.4%        |  2735 TF 27.4%    |  2599 TF 26.0%     |
| 4352 × 15360 × 3840   |  4096 TF 41.0%        |  5903 TF 59.0%        |  6339 TF 63.4%        |  2646 TF 26.5%    |  2964 TF 29.6%     |
| 4352 × 10240 × 3072   |  4174 TF 41.7%        |  5375 TF 53.8%        |  6074 TF 60.7%        |  2299 TF 23.0%    |  2350 TF 23.5%     |

Takeaways:

- **Real NVFP4 ceiling on this HW ≈ 60% MFU** (CUTLASS 2-CTA
  256×256). Not the 30-40% that I had been quoting from memory.
- **1-CTA gap**: CUTLASS ≈ 41% MFU, ours ≈ 27%. At the same tile
  (128×256). Missing pieces are on our side — persistent scheduler,
  stage count, epilogue / MMA overlap. This is task #41.
- **2-CTA Phase 1 gap**: CUTLASS 2-CTA 256×128 hits ≈ 53-59% even
  though FLOPs/atom equals 1-CTA 128×256. Ours gets essentially
  zero 2-CTA benefit (≈ 28% vs 27%). So Phase 1's ~0 speedup is not
  inherent — it's our implementation. Phase 2 (256×256 +
  overlapping_accum, task #39) should target ~60% to match CUTLASS.
- Small-M (M=256) is grid-limited for both — 12 tiles vs 148 SMs,
  tensor cores idle most of the kernel. Don't over-read that row.

### FA4 rewrite lineage — v0_fa4 (no LoRA) → v2_fa4 (+ LoRA + C1)

On the same `cutlass_nvfp4_bench` / `gemm_v2_fa4_c1_bench` run, with
v0_fa4 (persistent FA4 skeleton) and v2_fa4 (+ β-interleaved LoRA on
shared-tmem acc + C1 two-stage LoRA prolog):

| shape (M, K, N, R)         | v0_fa4 1-CTA | v0_fa4 2-CTA | v2_fa4+C1 2-CTA | pre-C1 v1_fa4 2-CTA |
| -------------------------- | ------------ | ------------ | --------------- | ------------------- |
| 4352, 3840, 3072, R=128    |    7.7%      |    7.6%      |    14.2%        |    6.0%             |
| 4352, 3840, 15360, R=128   |   23.6%      |   24.2%      |    18.6%        |   15.2%             |
| 4352, 15360, 3840, R=128   |   23.6%      |   26.4%      |    18.1%        |   17.0%             |
| 4352, 10240, 3072, R=32    |   16.6%      |   17.1%      |    26.1%        |   11.6%             |
| 4352, 10240, 3072, R=256   |   16.9%      |   16.9%      |    SKIP†        |   SKIP*             |

\* 2-CTA + R=256 overflows LA/LU smem at single-stage.
† C1 2-stage doubles LA/LU smem → R=256 no longer fits at any tile
(production R ≤ 128; stress shape is out of scope).

- **C1 (`num_lora_stage=2`)** lifts 2-CTA LoRA across every shape:
  +8.2pp on the smallest (K=3840 N=3072 R=128: 6.0% → 14.2%, ~2.4×),
  +14.5pp on R=32 (11.6% → 26.1%, ~2.2×). The pre-C1 v1_fa4 "2-CTA
  LoRA costs more than 1-CTA" anomaly is eliminated — 2-CTA is now
  ≥ 1-CTA on every LoRA shape (1-CTA column dropped as out of scope
  per `CTA1 去掉`).
- **Mechanism (validated, ncu on Verda B200, task #48)**: ran 6 single-
  kernel ncu captures on B200 (`--set detailed`, hmma subpipe metric is
  the NVFP4 tensor-pipe counter on Blackwell — UTCQMMA routes there).
  Three v2_fa4 configurations on shape A (M=4352 K=3840 N=3072 R=128
  fp16) — `num_lora_stage=0/1/2` via the `_patch_lora_stage` harness
  override (`tmp/profile_gemm_v2_fa4.py`) — plus CUTLASS NVFP4 persistent
  GEMM at the same shape and at K-heavy shape B (M=4352 K=15360 N=3840):

  | metric                        | v2 stage0 (LoRA off) | v2 stage1 (pre-C1) | v2 stage2 (C1) | CUTLASS A |
  | ----------------------------- | --------------- | --------------- | --------------- | ---------- |
  | duration (µs)                 |     42.0        |     77.1        |     69.6        |     40.9   |
  | SM throughput %               |     52.3        |     54.6        |     41.2        |     57.9   |
  | Memory throughput %           |     42.0        |     27.5        |     27.6        |     54.4   |
  | DRAM throughput %             |      5.7        |      3.5        |      3.9        |      5.9   |
  | hmma subpipe % (NVFP4 tcore)  |   **60.5**      |     31.8        |   **34.9**      |   **60.3** |
  | warp cycles / issued inst     |     15.0        |     18.6        |     25.9        |     13.9   |
  | long_scoreboard cyc (L1TEX)   |     10.6        |     13.8        |     21.8        |      9.5   |

  Three findings:

  1. **v2_fa4 main MMA matches CUTLASS in isolation.** stage0 vs
     CUTLASS A: 42.0 vs 40.9 µs (Δ 2.7%), hmma 60.5 vs 60.3%. The FA4
     skeleton's main K-loop is competitive with the hand-tuned CUTLASS
     persistent kernel — the gap is not in the main MMA.
  2. **LoRA prolog halves NVFP4 tensor-pipe utilization.** stage0 → stage2
     drops hmma from 60.5% to 34.9% (−25.6pp). DRAM throughput is low
     (≤6%) across all configs, so it is **not** DRAM bandwidth — the
     dominant stall is `long_scoreboard` on L1TEX (warp waiting for an
     SMEM load). LA/LU loads are serialized against the main K-loop's
     A/B SMEM consumption inside the same SM.
  3. **C1 (`num_lora_stage=2`) is a partial fix.** stage1 → stage2 cuts
     duration 9.7% (77.1 → 69.6 µs) and lifts hmma +3.1pp. The 2-stage
     prolog amortizes prolog cost across more main MMA iterations, but
     L1TEX wait per warp-cycle actually rises (13.8 → 21.8 cyc) because
     the kernel issues fewer instructions overall — net win on duration,
     no improvement on the latency root cause. The rest of the
     pre-C1 v1_fa4 → v2_fa4+C1 ~2.4× speedup (table above) came from
     the v1→v2 `lora_smem_bytes` accounting fix (was cluster-level,
     freed one pipeline_aw stage), not from C1's stage count.
- **v2_fa4+C1 2-CTA still 35-50pp below CUTLASS 2-CTA 256×128 ceiling**
  (~53-59% MFU). On K-heavy shape B (M=4352 K=15360 N=3840) the gap is
  **1.98×** (v2_fa4 287 µs / hmma 35.6% vs CUTLASS 145 µs / hmma 70.3%).
  Remaining gap is outside the C1 patch's scope — directions worth
  exploring next:

  - Increase LoRA prolog stage count beyond 2 (stage=3?) to deepen
    the latency-hiding window without doubling smem again — task #58
    (blocked on #57 C2 ncu validation; same pipeline-stage axis).
  - Move LoRA TMA loads to the `multicast` cluster path so LA/LU
    bytes are shared across both CTAs (currently only A/B are
    multicast in 2-CTA mode) — task #59.
  - Overlap LoRA MMA with the main K-loop's epilogue tail rather
    than running it strictly before the main MMA exit — task #60
    (most invasive; needs `num_acc_stage=2`, attempt last).

### Cross-arch reference — nunchaku `gemm_w4a4` on RTX PRO 6000 Blackwell

Nunchaku NVFP4 is gated on `__CUDA_ARCH__ >= 1200` (sm_120a/121a);
the wheel doesn't ship an SM_100 build, so this path can't run on
B200. We run it on RTX PRO 6000 Blackwell Server Edition (SM_120a)
as an implementation-quality reference — not a ceiling, and **not
apples-to-apples with our v2_fa4+C1 numbers** (different chip,
different peak, different tensor-core ISA). GEMM_SHAPES, LoRA on,
bias+wcscales on, `fp4=True`. MFU vs 4000 TFLOPS dense FP4 per
RTX PRO 6000 Blackwell datasheet. Run with
`modal run scripts/modal_app.py::nunchaku_gemm_bench`.

| shape (M, K, N, R)         | nunchaku fp16 | nunchaku bf16 |
| -------------------------- | ------------- | ------------- |
|  256, 3840,  3072, R=128   |     2.4%      |     2.6%      |
| 4352, 3840,  3072, R=128   |    16.2%      |    17.7%      |
| 4352, 3840, 15360, R=128   |    19.5%      |    24.7%      |
| 4352, 15360, 3840, R=128   |    25.0%      |    30.5%      |
| 4352, 10240, 3072, R=32    |    21.4%      |    25.2%      |
| 4352, 10240, 3072, R=256   |    19.8%      |    23.2%      |

- **bf16 consistently +3-5pp over fp16** on nunchaku. Our v2_fa4+C1
  does **not** show this lift — ran the same shapes on B200 and got
  fp16 ≈ bf16 to within ±0.1pp on 3/4 shapes:

  | shape (M, K, N, R)         | ours fp16 | ours bf16 | ours Δ | nunchaku Δ |
  | -------------------------- | --------- | --------- | ------ | ---------- |
  | 4352, 3840,  3072, R=128   |   11.9%   |   13.9%   |  +2.0  |  +1.5      |
  | 4352, 3840, 15360, R=128   |   17.9%   |   18.0%   |  +0.1  |  +5.2      |
  | 4352, 15360, 3840, R=128   |   18.2%   |   18.2%   |  +0.0  |  +5.5      |
  | 4352, 10240, 3072, R=32    |   26.0%   |   26.0%   |  +0.0  |  +3.8      |

  (fp16 numbers drifted vs the earlier table by ~2pp — Modal cross-run
  variance.) The asymmetry makes sense: nunchaku's MMA is inline PTX
  (`asm volatile mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
  vs `.f16.f16.f16.f16` in `mma_earlycuda.cuh`), two separate hand-
  tuned PTX paths with different register packing / acc-precision
  choices. Our CuTe DSL goes through one tcgen05 atom with ab_dtype
  substitution — same MLIR lowering for both, no per-dtype tuning.
  Head-to-head nunchaku fp16 ≈ us fp16 within single-digit pp;
  nunchaku bf16 pulls away to 6-12pp ahead.
- Peak MFU 30.5% at (M=4352 K=15360 N=3840, R=128, bf16). Production
  Flux/ZImage shapes generally land in the 17-25% range — this is
  the "what does a mature, hand-PTX-tuned W4A4 NVFP4 kernel achieve
  on consumer-grade Blackwell" reference. We are pre-ncu and pre-
  bf16-specific optimization; single-digit-pp on fp16 is already a
  good checkpoint for a CuTe DSL first-pass. Getting within bf16
  hand-PTX territory likely requires dropping to inline PTX (out of
  scope — defeats the CuTe DSL path choice).
- Small-M (M=256) grid-limited (tiles / SMs < 1), 2-3% — same
  grid-starvation pattern as CUTLASS on our B200 table.
