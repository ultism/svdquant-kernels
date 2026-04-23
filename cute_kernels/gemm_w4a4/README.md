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

### FA4 rewrite lineage — v0_fa4 (no LoRA) and v1_fa4 (+ LoRA)

On the same `cutlass_nvfp4_bench` run, with v0_fa4 (persistent FA4
skeleton) and v1_fa4 (+ β-interleaved LoRA on shared-tmem acc):

| shape (M, K, N, R)         | v0_fa4 1-CTA | v0_fa4 2-CTA | v1_fa4 1-CTA | v1_fa4 2-CTA |
| -------------------------- | ------------ | ------------ | ------------ | ------------ |
| 4352, 3840, 3072, R=128    |    7.7%      |    7.6%      |    6.6%      |    6.0%      |
| 4352, 3840, 15360, R=128   |   23.6%      |   24.2%      |   18.0%      |   15.2%      |
| 4352, 15360, 3840, R=128   |   23.6%      |   26.4%      |   18.5%      |   17.0%      |
| 4352, 10240, 3072, R=32    |   16.6%      |   17.1%      |   15.3%      |   11.6%      |
| 4352, 10240, 3072, R=256   |   16.9%      |   16.9%      |   11.1%      |   SKIP*      |

\* 2-CTA + R=256 overflows LA/LU smem (no small 2-CTA tile).

- **v0_fa4 beats old v0 by +1–3 pp** (persistent scheduler win).
- **v1_fa4 LoRA cost is larger on 2-CTA than on 1-CTA**, which is
  anomalous — LoRA FLOPs and atom count don't depend on cluster
  size. K-heavy / N-heavy shapes regress worst (e.g., K=3840 N=15360
  R=128: 1-CTA 18.0% vs 2-CTA 15.2%, so 2-CTA loses 2.8pp to 1-CTA
  once LoRA is on). v0_fa4 alone gains +0.6 pp going 1→2-CTA on the
  same shape; adding LoRA flips the sign.
- Hypothesis (not yet measured): LoRA atom under CtaGroup.TWO incurs
  extra `tcgen05.commit` cluster-barrier overhead per atom, or the
  single-stage `pipeline_lora` mbar handshake is serializing across
  the pair. Root-causing needs ncu instruction-timeline counters —
  Modal blocks those. Rolls into the Verda profiling track (task #41
  equivalent for 2-CTA LoRA).
