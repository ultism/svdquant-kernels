# Plan: write the SVDQuant W4A4 Blackwell blog post as a teaching post

## Context

Create a teaching post for community Blackwell-kernel developers about
this repo's `gemm_w4a4` kernel — main NVFP4 scaled-MMA + β-interleaved
LoRA + per-column `× wcscales + bias`, shipped at
`cute_kernels/gemm_w4a4/kernel_v2_fa4.py`. The earlier draft (kept outside
this repo) read as a running account; the goal here is to rewrite around
a single editorial spine and ship it into the repo.

**`docs/blog/` does not yet exist** — this is a creation task. Output:
two new files, EN and ZH mirrors, written natively in each language.

**Editorial spine**: `v1 (kernel.py) → v2_fa4 (kernel_v2_fa4.py)` is a
*re-architecture* forced by the synchronization-state-space outgrowing
stock `cutlass.pipeline.PipelineState`, not an additive change. The two
"why" anchors the post must drive home:

- **Why we moved off v1**: 1-CTA + stock `PipelineState` + monolithic
  kernel caps at ~27 % MFU; the 2-CTA Phase-1 attempt (`cta_group=TWO`)
  got essentially **zero benefit** (28 % vs 27 %) on production shape.
  The state space we need (pipeline stages × 2-CTA pair barriers ×
  persistent tile loop × LoRA β second-MMA × epilogue correction chain)
  is 5-dimensional and outgrows the single-dimensional implicit
  `PipelineState`. Empirical evidence: a prior persistent port passed
  correctness at 1-tile-per-CTA but **hung 500× when each CTA
  processed ~20 tiles** (commit `61905df`).
- **Why we reference FA4**: FA4 had already solved the warp-spec +
  persistent-tile + per-warp-state pattern for `tcgen05.mma` + TMEM +
  2-CTA. We adopt its *scaffolding* (warp specialization,
  `PipelineStateSimple`, `_w_index_phase`, `StaticPersistentTileScheduler`,
  `gemm_ptx_partial(acc_tmem_addr: Int32)`, the `AI/DEBUG_2CTA.md`
  pitfall list) — **not** its online softmax / S→P→O dataflow / Q-K-V
  partitioning.

**Editorial claim** (made once, in §2, no defense): this op is a better
teaching vehicle than FA4 for Blackwell primitives — the math fits on
one screen (matmul + low-rank residual + affine), but the implementation
stretches every SM_100 primitive worth knowing (`tcgen05` scaled-MMA,
2-CTA dense MMA via `cta_group=TWO`, TMEM accumulator sharing, TMA
bundles via `extra_tx_count`, warp specialization, persistent tile
scheduler) without the online-softmax cognitive tax.

Bumps go **inline** within each design step, not in a separate
"gotchas" section. Pedagogy assumes CUTLASS 2.x + CUDA baseline;
everything SM_100-new gets explained.

**Target length**: ~40 KB each (EN + ZH). Strict structural mirrors; ZH
written natively, not machine-translated.

**The hero finding** is the LU SMEM ÷ cta_group_size fix (§7,
`+198 % TF / +12.7 pp MFU`, single-line patch). Secondary findings: the
v1 → v2_fa4 re-architecture spine, and C1 2-stage LoRA prolog. The
nunchaku-fp16 register-spill observation is **demoted** to a brief
mention inside §9 (per user input 2026-05-13): it explains an asymmetry
in the reference, it is not a hero moment of our work.

## The change at a glance

```
        ┌────────────────────────────────────────────────────┐
        │ NEW: docs/blog/2026-05-13-svdquant-w4a4-blackwell  │
        │      .md  + .zh.md  (~40 KB each)                  │
        └────────────────────────────────────────────────────┘

                              source material (read-only):

  ┌──────────────────────────────────────────────────────────┐
  │ cute_kernels/gemm_w4a4/                                  │
  │   kernel.py            (v1 baseline; 1-CTA, stock        │
  │                         PipelineState; docstring 1-36)   │
  │   kernel_v0_fa4.py     (FA4 scaffolding, no LoRA)        │
  │   kernel_v2_fa4.py     (production; 1-54 why-FA4;        │
  │                         429-444 LU SMEM fix; 1248-1253   │
  │                         producer phase=1 init; 1261-1270 │
  │                         shared-tmem; 1309-1376 β-iter)   │
  │   README.md            (RICHEST single source; staging   │
  │                         table 19-37; why-FA4 63-73;      │
  │                         CUTLASS baseline 145-175;        │
  │                         FA4 lineage 177-254; nunchaku    │
  │                         cross-arch 256-306)              │
  └──────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────┐
  │ docs/gpu.md             (CANONICAL numbers: 129-355;     │
  │                          ncu A/B 383-422; stage sweep    │
  │                          357-381; hmma routing 105-127)  │
  │ docs/gotchas/cute_dsl.md  (LU÷cta_group_size at 289-347; │
  │                          partition_D 153-229; cluster_   │
  │                          layout_vmnk 90-151; const_expr) │
  │ docs/architecture.md § "Scope decisions" (v3 out-of-scope) │
  │ docs/kernels/gemm_w4a4.md  (β interleave math, why-β 26-52)│
  │ docs/kernels/gemm_w4a4_fa4_v0_bringup.md  (Bug 1 — 9-min  │
  │                          hang, producer-phase=1; Bug 2)  │
  └──────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────────────┐
        │  Editorial spine                                   │
        │                                                    │
        │  §1 hook → §2 op + claim → §3 Blackwell primitives │
        │              (teaching) → §4 v1 baseline → §5 why  │
        │              FA4 → §6 v2_fa4 rewrite (inline       │
        │              bumps) → §7 LU SMEM hero →            │
        │              §8 ncu methodology → §9 calibration   │
        │              → §10 levers → §11 thanks             │
        └────────────────────────────────────────────────────┘
```

## Outline (final)

1. **tl;dr / hook** (~1 KB) — one paragraph: "we re-architected from
   1-CTA stock CUTLASS port (`kernel.py`, 27 % MFU ceiling) to FA4-derived
   2-CTA persistent (`kernel_v2_fa4.py`, 16.9–27.3 % MFU on production
   shapes — fp16 4/4 ahead of nunchaku, bf16 3/4 ahead, all on the
   apples-to-apples shapes). Single most valuable single-line fix:
   halve LU SMEM per CTA under 2-CTA, +198 % TF on the production
   shape." Numbers from `docs/gpu.md:286-326`.

2. **Why this op (and why this post exists)** (~2.5 KB) —
   - Math: `y = scaled_mma(act₄, wgt₄) · wcscale + bias + lora_act_in @ lora_up`
     (`docs/kernels/gemm_w4a4.md:5-8`).
   - vLLM drop-in constraint, no `fuse_glu` (`CLAUDE.md` + `docs/architecture.md:76-100`).
   - SM_100/103 specifically — nunchaku's coverage gap (`tmp/nunchaku/setup.py:41-64`).
   - CuTe DSL Python over CUDA C++ headers (~10× less template
     boilerplate, same `cutlass-dsl` package).
   - **Editorial claim** (two sentences, no defense): better teaching
     vehicle than FA4 — one screenful of math, but exercises every
     Blackwell primitive worth knowing, no online-softmax tax.

3. **The Blackwell primitives this kernel uses** (~7–9 KB; risk
   section, see below) — teaching section, assumes CUTLASS 2.x + CUDA
   baseline. Each subsection: what's new on SM_100, what API CuTe DSL
   exposes, what trap to know.
   - `tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X` — NVFP4
     layout (E2M1 values + FP8-E4M3 per-16-K block scales);
     `make_blockscaled_trivial_tiled_mma` exposes MXF4/NVFP4/MXF8 only
     (no INT4) — explains the format split (`CLAUDE.md`).
   - 2-CTA dense MMA via `cta_group=TWO` — `cluster_layout_vmnk` axes
     (V is at index [0] for cluster_m, not [2] — anchor to gotcha at
     `docs/gotchas/cute_dsl.md:90-151`). How `partition_shape_{A,B}`
     halves per-CTA SMEM (A along M, B along N). External anchor:
     Modular `matmul-on-blackwell-part-3` § "Shared Memory Optimization".
   - TMEM as a programmable accumulator space — preview that two atoms
     can target the same `Int32 acc_tmem_addr`, full β-interleave
     payoff in §6.
   - TMA bundles with `extra_tx_count` — one barrier for
     `act+ascales+wgt+wscales` (`kernel_v2_fa4.py:906-914`); the
     `is_leader_cta` gate under 2-CTA (`PipelineTmaUmma` override,
     anchored to FA4 `pipeline.py`).
   - `StaticPersistentTileScheduler` — grid clamp to `sm_count`,
     `tile_idx += grid_dim()` (`kernel_v2_fa4.py:885-892`).
   - Warp specialization — preview only (load/mma/epilogue split),
     full picture lands in §6.
   - **Diagram**: 2-CTA cluster sketch (ASCII; 10 lines) showing
     `partition_shape_A` (M-split) vs `partition_shape_B` (N-split)
     across the V partners.

4. **v1 (`kernel.py`) — the pre-FA4 baseline** (~4–5 KB) —
   - Lineage: ported from `tmp/cutlass/.../dense_blockscaled_gemm_persistent.py`
     (`kernel.py:23-27`). Stripped: persistent TileScheduler, clusters > 1,
     TMA multicast, overlapping_accum, tile_n ∈ {64, 192} SFB-shift
     hacks. Uses 1-CTA MMA on shape-adaptive tilers (128×128 small-M,
     128×256 otherwise).
   - Mechanics: monolithic `@cute.kernel`, stock
     `cutlass.pipeline.PipelineState`, β-interleaved LoRA via aliased
     `cute.Tensor` for the shared TMEM acc, TV-layout match referenced
     by `kernel.py:30-33`.
   - Honest numbers vs CUTLASS NVFP4 on same B200 + same shapes — kept
     terse, source table at `cute_kernels/gemm_w4a4/README.md:154-160`:
     - 1-CTA 128×256: ours 27.4 % vs CUTLASS 41.7 % at production-shape
       row.
     - 2-CTA Phase 1 (256×128, cluster=(2,1)): ours 26 % vs CUTLASS
       51.8 % — essentially **no 2-CTA benefit** despite 2× FLOPs/atom.
   - Diagnosis: 2-CTA needs cluster-aware barriers, `cta_group=TWO`,
     persistent tile-loop with cross-tile state survival, and a sync
     machine to survive all three. Implicit `PipelineState` can't. →
     hand-off to §5.
   - Frame v1 as "the right first attempt that taught us what the
     actual constraints are" — not defective, just outscaled.

5. **Why FA4 — the crisis that forced re-architecture** (~4–5 KB) —
   - **The 5-dimensional state space**: pipeline stages × 2-CTA pair
     barriers × persistent tile loop × LoRA β second-MMA × epilogue
     correction chain. Implicit `PipelineState`'s branching `advance()`
     is single-dimensional and assumes 1-tile-per-CTA (source: README
     63-73; kernel_v2_fa4.py:1-54).
   - **The empirical hit**: prior persistent port hung 500× when each
     CTA processed ~20 tiles (commit `61905df`); classic phase/state
     drift across tile boundaries. Anchored to docstring at
     `kernel_v2_fa4.py:275-279`.
   - **What we adopted from FA4** (`tmp/flash-attention/flash_attn/cute/`):
     - `flash_fwd_sm100.py`: warp-specialized mainloop (separate
       `load()` / `mma()` / `epilogue()`). Our mapping: `mma` inherits
       FA4's two-MMA-in-one-warp pattern, but both MMAs point at the
       **same** TMEM acc region for β-accumulation (not FA4's chained
       S→P→O dataflow).
     - `pipeline.py`: `PipelineStateSimple` (single `_phase_index`
       counter, `% stages` / `// stages` properties);
       `_w_index_phase` mixin so each warp drives its own state;
       `PipelineTmaUmma.create` adds `extra_tx_count` + leader-CTA gate.
     - `tile_scheduler.py::StaticPersistentTileScheduler`.
     - `blackwell_helpers.py::gemm_ptx_partial` — takes raw
       `acc_tmem_addr: Int32` so two MMAs can target the same TMEM
       region without `cute.Tensor` aliasing.
     - `AI/DEBUG_2CTA.md` pitfall list.
   - **What we did NOT take**: online softmax, S→P→O chain, Q/K/V
     partitioning. Used as scaffolding only.

6. **v2_fa4 — the rewrite** (~10–12 KB) — longest section; bumps
   threaded inline. Subsection structure:
   - **v0_fa4 as scaffolding** — FA4 skeleton without LoRA, used to
     validate the FA4 mechanics in isolation. Numbers: 7.7 % / 7.6 %
     MFU at 1-CTA/2-CTA on the production shape — *lower* than v1's
     27 %, but it's a partial-feature scaffold, not the destination
     (`README.md:183-189`). Frozen as the v0/v1 reference, flag-gated
     on `enable_lora`.
   - **Bump (inline): 9-min hang on first smoke** — producer phase=0
     init blocks forever after `pipeline_init_arrive` pre-arms empty
     barrier to parity=1. Fix: `acc_producer_phase = Int32(1)`. Anchor:
     `kernel_v2_fa4.py:1247-1253` + bring-up doc Bug 1 at
     `docs/kernels/gemm_w4a4_fa4_v0_bringup.md:27-44`.
   - **Re-adding LoRA — the β-interleave on shared TMEM**:
     - The math: both atoms write the same `acc_tmem_addr`; `zero_init`
       only on first main K-block, `accumulate` everywhere else
       (`kernel_v2_fa4.py:1307`, `1347`). One TMEM region, one final
       epilogue, no GMEM round-trip.
     - The Blackwell mechanism: `tcgen05` issue queue is per-CTA
       in-order, so the LoRA atom sees the preceding main atom's TMEM
       write. β-interleave sprinkles LoRA atoms with
       `stride = K_atoms // R_atoms`
       (`kernel_v2_fa4.py:1309-1376`).
     - Why FA4's `gemm_ptx_partial` matters: raw `Int32 acc_tmem_addr`,
       no `cute.Tensor` alias needed (v1 went the alias route — works
       but messier).
     - Why β beats serial α: measured in `docs/kernels/gemm_w4a4.md:26-52`;
       LoRA atom count is too small to saturate `tcgen05`'s async-issue
       depth, α inflates `t_lora` by 2-20×.
     - **Diagram**: β-interleave timing as ASCII (or short mermaid) —
       main atoms in K-loop, LoRA atom injected every `stride`,
       per-CTA in-order tcgen05 queue.
   - **Bump (inline): 2-CTA LoRA regression** — v1_fa4 (= v0_fa4 +
     single-stage LoRA prolog) at 2-CTA gives **6.0 % MFU**, *worse*
     than v0_fa4's 7.6 % (`README.md:183-189`). LoRA SMEM ate the
     budget; `num_ab_stage` got squeezed. → motivates C1.
   - **C1 fix: 2-stage LoRA prolog** — `num_lora_stage = 2`. Lifts
     2-CTA LoRA from 6.0 % → 14.2 % on production shape; +14.5 pp at
     R=32 (11.6 % → 26.1 %). ncu mechanism (Verda B200): hmma stage1
     → stage2 +3.1 pp, duration −9.7 %, but per-warp `long_scoreboard`
     wait actually rises (13.8 → 21.8 cyc) — the win is amortization
     of prolog cost across more main MMA iters, not latency reduction
     (`README.md:201-239`).
   - **Adding the fused `× wcscale + bias` epilogue (v2 step)** —
     epilogue warp's job grows, `pipeline_acc` consumer side changes.
     Fold here to avoid an extra epilogue pass; column-wise `wcscale`
     aligns with TMEM output layout. Quick anchor; this is not a
     surprising step.
   - **Diagram**: 3-warp / 3-pipeline mermaid showing load + mma +
     epilogue warps with `pipeline_aw`, `pipeline_lora`, `pipeline_acc`
     (sources: README 75-115).

7. **The silent SMEM-budget bug — LU ÷ cta_group_size** (~5–6 KB) —
   the hero finding; own section.
   - The handwritten SMEM-budget formula
     (`kernel_v2_fa4.py:429-444`): `lora_smem_bytes = (la + lu) *
     num_lora_stage`. LA correctly halved (M-split,
     `cluster_shape_mn[0] == cta_group_size`). LU **not** halved — bug.
   - Why it's a bug: `make_smem_layout_b(tiled_mma_2cta, …)` halves LU
     via N-split inside `partition_shape_B`, identical mechanism to
     main B (per Modular `matmul-on-blackwell-part-3`, "Shared Memory
     Optimization"). Handwritten formula double-counts.
   - **Symptom: nothing**. Kernel compiles, runs, numerically correct.
     `num_ab_stage` silently clamps 4 → 2; main K-loop pipeline depth
     halves; wall-clock "fine."
   - The probe: `cute.cosize(slice_(lu_smem_layout_staged, …))` at
     trace time — runs in Int32, returns the actual SMEM cosize. Two
     minutes to write (`docs/gpu.md:243-261`). Factor = 0.500 → found
     the bug.
   - Fix: one extra `// self.cta_group_size` on the `lu_bytes` line
     (commit `7296e90`).
   - The payoff at M=4352, K=3840, N=3072, R=128, fp16, 2-CTA:
     **566 TF → 1685 TF (+198 %)**, **4.2 % → 16.9 % MFU**
     (`docs/gpu.md:286-296`). ncu A/B at the same launch config
     (`docs/gpu.md:383-422`): Duration −31.2 %, SM % +11.99 pp, SM
     Active Cycles −36.3 %, Memory Throughput +45 %, L1/TEX +16.25 pp,
     L2 +11.61 pp. Commit `4a2d068`.
   - **Why this generalizes**: any handwritten SMEM-budget arithmetic
     feeding the stage solver, for an operand whose SMEM came from
     `make_smem_layout_{a,b}(tiled_mma_2cta, …)`, must divide by
     `cta_group_size` along the partitioned axis. Otherwise the solver
     silently under-counts headroom; kernel runs wrong-but-fast-enough
     to look right. Documented in `docs/gotchas/cute_dsl.md:289-347`.
   - **Diagram**: a small SMEM-budget table comparing handwritten
     estimate vs `cute.cosize` actual (mirrors `docs/gpu.md:262-283`
     before/after).

8. **Reading ncu like a Blackwell kernel author** (~4–5 KB) —
   methodology, post-implementation. Folds the Modal-vs-Verda tool
   story into this section (drop separate §11 from earlier draft —
   redundant). Items:
   - **Routing quirk** (the most copy-pasted item): NVFP4 tensor-pipe
     is `hmma` subpipe in ncu's metric tree, not `qmma`. Use
     `sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active`.
     `sm__ops_path_tensor_op_utcqmma_src_fp4_fp6_fp8_dst_fp{16,32}`
     for FLOPs. UTCOMMA is a separate FP4-only path
     (`docs/gpu.md:105-127`).
   - **SOL breakdown** for a 2-CTA UMMA kernel — table from the C1 ncu
     run (`README.md:209-217`), reading guide: hmma % is "how busy the
     tensor pipe is", warp cycles / issued inst is "the kernel's
     average IPC penalty", `long_scoreboard` cyc is "L1TEX wait", etc.
   - **Trace-time `cute.cosize` probe pattern** — minimal code snippet
     from `docs/gotchas/cute_dsl.md:319-324`. Two-minute write-up,
     unblocked the LU finding.
   - **Counter-access constraints**: Modal blocks ncu
     (`NVreg_RestrictProfilingToAdminUsers=1`); Verda unblocks it
     (`CLAUDE.md` execution-environment section). The LU fix would
     have read as wall-clock noise without ncu.

9. **Calibration: vs CUTLASS NVFP4 and vs nunchaku** (~4–5 KB) — two
   reference points, two distinct claims. The nunchaku-fp16
   register-spill observation is demoted here from a hero moment to a
   one-paragraph mention (per user input 2026-05-13).
   - **vs CUTLASS NVFP4 on same B200 (the honest ceiling)** —
     `README.md:154-175` table. CUTLASS 2-CTA 256×256 reaches ~60 %
     MFU on production shape; our v2_fa4+C1+LU-fix at 16.9 %.
     Frame: "the kernel is not yet a hand-tuned NVFP4 GEMM; on this op
     LoRA + wcscale + bias adds real cost, CUTLASS's reference does
     none of that. 17 % MFU is a reasonable B200 first pass on the
     full SVDQuant op. The remaining 40+ pp is the work in §10."
   - **vs nunchaku on RTX PRO 6000 (SM_120a) — implementation-quality
     reference, not a ceiling**: hardware peak and tensor-core ISA
     differ. Cross-shape MFU table from `docs/gpu.md:314-319`. The
     real story (the apples-to-apples one): fp16 4/4 ahead, bf16 3/4
     ahead; the one bf16 shape where nunchaku still leads (M=4352
     K=15360 N=3840, −3.2 pp) is the "bf16 hand-PTX vs DSL MLIR
     lowering" asymmetry called out in `docs/gpu.md:79-103`.
   - **Brief mention** (no diagram, no own subsection): "nunchaku's
     hand-PTX has a known fp16 register-spill cliff (255 regs +
     ~2.28 M LMEM spills at fp16 vs 248 regs + zero at bf16, hence the
     ~5 pp fp16→bf16 jump inside their column). We don't reproduce it
     — single tcgen05 atom with ab_dtype substitution, same MLIR
     lowering for both. It explains the shape of their column, not
     the location of ours." Two-three sentences, source-anchored, move on.
   - Absolute throughput at 2.6–3.4× (B200 vs PRO 6000, peak ratio
     ~2.5×) — `docs/gpu.md:330-335`.

10. **Remaining levers** (~1–2 KB) — terse, not a TODO list.
    - bf16 register tuning (the ~2 pp DSL-vs-PTX gap when nobody spills).
    - Wave quantization (Waves Per SM at production shape).
    - LoRA prolog stage=3 — **measured slower** post-LU-fix; the
      solver buys the extra LoRA stage by giving up `num_ab` stages,
      and main K-loop loses more than LoRA prolog gains
      (`docs/gpu.md:357-381`). Task #58 is dead in its current form.
      Real bottleneck moved to main K-loop / TMEM occupancy.
    - Closing the gap to CUTLASS 2-CTA 256×256 (~60 % MFU
      ceiling): `overlapping_accum` at tile_n=128, tile 256×256 (mutually
      exclusive on TMEM budget per
      `docs/gotchas/cute_dsl.md:231-287`), tile-promotion tradeoff.
    - Out of scope: next-layer NVFP4 quant
      (`docs/architecture.md:76-100`); kernel-side `fuse_glu`.

11. **Where the code lives + thanks** (~1 KB) — repo + the three
    kernel files; key commits (`7296e90`, `4a2d068`, `c0d8e9e`,
    `61905df`, `8f91240`); Ascend-side cross-link
    (`csrc/kernels/gemm_w4a4/`); Verda thanks for unrestricted ncu.

## Diagrams

Three diagrams; the rest stays prose.

- **§3**: 2-CTA cluster sketch — ASCII, ~10 lines. Shows `partition_
  shape_A` (M-split) and `partition_shape_B` (N-split) across V partners.
- **§6**: 3-warp / 3-pipeline diagram (mermaid). Adapted from
  `cute_kernels/gemm_w4a4/README.md:75-115`. Boxes for load / mma /
  epilogue warps; arrows for `pipeline_aw`, `pipeline_lora`,
  `pipeline_acc`.
- **§6**: β-interleave timing — short ASCII showing the main K-loop
  with LoRA atoms injected every `stride` positions, and the per-CTA
  in-order tcgen05 queue depth.

Skip the "v1 vs v2_fa4 architecture side-by-side" diagram from the
earlier draft — the section header progression §4 → §5 → §6 carries
that contrast already, and a mermaid diagram of "monolithic vs
warp-spec" would be busier than illuminating.

Skip the LU-SMEM-budget visual — the before/after table in §7
(2 rows × 3 cols) carries the same information more compactly than a
diagram would.

## Critical files to read while writing

| section | source |
| ------- | ------ |
| §1, §6, §7, §8 | `docs/gpu.md:129-422` (canonical numbers, ncu A/B, stage sweep, hmma routing) |
| §3 primitives | `docs/gotchas/cute_dsl.md:90-151` (cluster_layout_vmnk) + `153-229` (partition_D) + `289-347` (LU÷cta_group_size) |
| §4 v1 baseline | `cute_kernels/gemm_w4a4/kernel.py:1-70` (docstring) + `cute_kernels/gemm_w4a4/README.md:145-175` (CUTLASS table) |
| §5 why-FA4 | `cute_kernels/gemm_w4a4/README.md:39-73` + `kernel_v2_fa4.py:1-54, 275-280` |
| §6 v2_fa4 | `kernel_v2_fa4.py:429-444, 885-961, 1247-1253, 1261-1270, 1307-1376` + `cute_kernels/gemm_w4a4/README.md:75-254` |
| §6 bring-up bumps | `docs/kernels/gemm_w4a4_fa4_v0_bringup.md:27-60` |
| §6 β math | `docs/kernels/gemm_w4a4.md:26-115` |
| §7 hero finding | `docs/gpu.md:234-422` + `docs/gotchas/cute_dsl.md:289-347` |
| §8 ncu methodology | `docs/gpu.md:105-127` + the C1 ncu table at `README.md:209-217` |
| §9 calibration | `cute_kernels/gemm_w4a4/README.md:154-306` + `docs/gpu.md:79-103, 305-326` |
| §10 levers | `docs/gpu.md:357-381` (stage=3 dead) + `docs/gotchas/cute_dsl.md:231-287` (tile_n vs num_acc_stage) + `docs/architecture.md:76-100` |

## Files to create / modify

| path | action | size |
| ---- | ------ | ---- |
| `docs/blog/` | mkdir (does not yet exist) | — |
| `docs/blog/2026-05-13-svdquant-w4a4-blackwell.md` | create | ~40 KB |
| `docs/blog/2026-05-13-svdquant-w4a4-blackwell.zh.md` | create | ~40 KB |

No source files are touched.

## EN/ZH parity convention

Strict structural mirrors. Same section count, same headers, same
table layouts, same code snippets, same anchors, same diagrams. ZH
written natively (not machine-translated) to keep prose tight. Each
section written in EN first then mirrored to ZH in the same pass
(not all-EN-then-all-ZH) so they don't drift.

## Verification

Before declaring done:

- Both files parse as CommonMark + GFM tables + mermaid fenced blocks
  (`mdformat --check` or open in VS Code preview).
- All internal repo paths resolve in this checkout: `cute_kernels/
  gemm_w4a4/{kernel.py,kernel_v0_fa4.py,kernel_v2_fa4.py,README.md}`,
  `docs/{gpu.md,architecture.md,gotchas/cute_dsl.md,kernels/gemm_w4a4.md,
  kernels/gemm_w4a4_fa4_v0_bringup.md}`.
- All cited commits resolve in this repo (`git rev-parse 7296e90 4a2d068
  c0d8e9e 61905df 8f91240` exits 0 — already verified during planning).
- All cited line ranges in the kernel files actually contain the cited
  code (`grep -nE "(acc_producer_phase = Int32\(1\)|lu_bytes|β-interleave)"
  cute_kernels/gemm_w4a4/kernel_v2_fa4.py` returns the expected hits at
  the line numbers cited above).
- All numeric claims trace back to a source file in this checkout:
  - 566 → 1685 TF and 4.2 % → 16.9 % MFU → `docs/gpu.md:286-296`
  - SM % +11.99 pp, Duration −31.2 % → `docs/gpu.md:393-403`
  - v2_fa4+C1 14.2 % at production shape, 6.0 % pre-C1 →
    `cute_kernels/gemm_w4a4/README.md:185-189`
  - v0_fa4 7.7 %/7.6 % at production shape → same table
  - CUTLASS 60 % at 2-CTA 256×256 → `cute_kernels/gemm_w4a4/README.md:158-160`
  - hmma stage0/1/2 ncu table → `cute_kernels/gemm_w4a4/README.md:209-217`
  - fp16 4/4 ahead, bf16 3/4 ahead → `docs/gpu.md:314-326`
- EN ↔ ZH structural parity: same section count, same table count, same
  diagram count, same anchor-link count. Eyeball both rendered.
- Read aloud the EN tl;dr + §4 (v1 baseline) + §5 (why-FA4) + §7
  (LU SMEM) to check the v1 → v2_fa4 re-architecture spine reads
  without the rest.

## Risks worth flagging during write

- **§3 (Blackwell primitives teaching) is the riskiest section** —
  easy to over- or under-shoot the assumed reader. Plan: write a first
  pass at "knows CUTLASS 2.x, hasn't touched SM_100" and adjust if it
  reads thin/thick. If §3 exceeds 10 KB, split: first half = NVFP4 +
  `cta_group=TWO` + `partition_shape_{A,B}`; second half = TMEM + TMA
  bundles + scheduler + warp-spec preview.
- **§4 → §5 → §6 transition is the load-bearing arc.** If §4 reads
  as critical of `kernel.py` (which works correctly, it just doesn't
  scale to 2-CTA), the rest of the post lands as snide. Frame v1 as
  the right first attempt that revealed the actual constraints.
- **§7 hero finding has to do its work without making the rest of
  the post feel deflated** (the headline number is *much* bigger than
  any other single delta). The way to land it: §7 is the highest
  single-line ROI, but it's the LU SMEM finding *as a generalizable
  CuTe DSL pattern* (any handwritten budget feeding the stage solver
  under 2-CTA) — that's what makes it teaching content, not just
  bragging.
- **§9 calibration is depressing if read as "we're at 17 % MFU and
  CUTLASS hits 60 %".** The honest read is: this op is harder than
  CUTLASS's reference (LoRA + wcscale + bias add real per-tile cost),
  17 % on the full op is a reasonable CuTe DSL first pass, the gap
  has explicit fixable items in §10. Write it that way.
- **The nunchaku-fp16 register-spill paragraph in §9** is the single
  point where the draft over-rotated. Two-three sentences, no
  diagram, no own subsection.
- **The β-interleave timing diagram** in §6 is new content; if it
  doesn't fit cleanly in ASCII/mermaid, fall back to prose + the
  `kernel_v2_fa4.py:1309-1376` inline quote.
