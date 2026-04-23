# gemm_w4a4 v0 FA4 — bring-up notes

Running log of what broke, what fixed it, what's still wrong. Scope
is the FA4-pattern rewrite at `cute_kernels/gemm_w4a4/kernel_v0_fa4.py`
(see README.md → "Architecture (FA4-derived 3-pipeline / 3-warp)" for
the target shape).

Status: **not passing**. Kernel runs end-to-end on B200 but every
shape fails correctness. Two fix candidates below; next action listed
at the end.

## Trace-check (dev box, sm_100a via `CUTE_DSL_ARCH`)

All three configurations lower to MLIR clean on the local SM_120 box
with the arch gate override:

| tiler       | cluster | dtype |
| ----------- | ------- | ----- |
| (128, 128)  | (1, 1)  | fp16  |
| (128, 256)  | (1, 1)  | fp16  |
| (256, 128)  | (2, 1)  | fp16  |
| (128, 256)  | (1, 1)  | bf16  |

Good enough to prove the layout math composes; correctness only
validates on B200.

## Bug 1 — `acc_producer_phase` initial value (fixed)

**Symptom**: 9-minute hang on Modal with no stdout past `nvidia-smi`.
Kernel didn't abort, didn't print, just stalled.

**Cause**: MMA warp's single-stage `pipeline_acc` producer phase
initialized to `Int32(0)`. After `pipeline_init_arrive`, the empty
mbarrier is pre-arrived to parity 1, so a `producer_acquire` with
phase 0 blocks forever — consumer never flips `full`, epilogue never
drains, deadlock.

**Fix**: initialize to `Int32(1)`. Mirrors what stock
`cutlass.pipeline.make_pipeline_state(Producer, …)` does (returns
`phase=1`). Also matches the FA4 comment in `load()`: "single-stage
producer starts at 1".

**Commit**: patched in-line in `kernel_v0_fa4.py`, same branch as the
scaffold (pre-push).

## Bug 2 — `tiled_mma.set(ACCUMULATE, …)` inside a runtime `while` (open)

**Symptom** (after Bug 1 fixed): 0/24 shapes pass. Split by K:

| K      | CTAs/tile | tiles/CTA | symptom       |
| ------ | --------- | --------- | ------------- |
|  3840  | 48 / 148  | 1 / ~5    | `rel = nan`   |
| 10240  | 148       | ~5        | `rel ≈ 1.0`   |
| 15360  | 148       | ~7–28     | `rel ≈ 1.0`   |

Same split on both 1-CTA and 2-CTA paths. `rel ≈ 1.0` means the output
is essentially zero (`|y_ref − 0| / |y_ref| ≈ 1`).

**Suspect**: `tiled_mma.set(tcgen05.Field.ACCUMULATE, False/True)` is
a host-side (trace-time) mutation of the Python `tiled_mma` object. A
runtime `while tile_idx < total_tiles_cluster:` traces the body once;
setter calls inside the body don't re-execute per-iteration. So what
ends up in the MLIR:

- The outer while-body is captured with ACCUMULATE=False at the first
  `cute.gemm` site (kblock 0 of first trace-pass) and True for all
  others — frozen for the whole runtime loop.
- Starting the *second* tile on the same CTA, the captured
  `gemm(ACCUMULATE=False)` at kblock 0 re-executes → **overwrites**
  the prior tile's acc, which would produce zero in the *first* tile
  and garbage in subsequent ones. Matches the `rel ≈ 1.0` pattern for
  multi-tile-per-CTA shapes.
- The `rel = nan` pattern for K=3840 (where shape M=256 has 1
  tile/CTA and shape M=4352 has ~5) is less cleanly explained — it
  may be a separate bug (SF tmem pointer or pipeline_aw ordering);
  resolving Bug 2 is prerequisite either way.

**Why the current `kernel.py` (non-persistent v0) works**: same
pattern, but with exactly one tile per CTA, the captured
`gemm(ACCUMULATE=False)` fires once per CTA lifetime — which is
correct for single-tile kernels. The moment you layer a persistent
loop on top, it breaks.

**Planned fix** (not yet tried): **two pre-built `mma_atom`s** —
`mma_atom_init` with ACCUMULATE=False and `mma_atom_accum` with
ACCUMULATE=True, both baked at trace time. In the K-loop, pick at
runtime based on `(k_tile, kblock_idx) == (0, 0)`. Mirrors FA4's
`blackwell_helpers.gemm(…, zero_init=bool)` which compiles two paths
via `const_expr`. For our case the `zero_init` branch is runtime
(first iteration of a persistent tile), so we need the two-atom
pattern, not FA4's `const_expr`.

Alternative: explicitly zero the acc tmem region before the K-loop
starts (via `tcgen05.tmem_store` of zeros or equivalent). More
instructions, same correctness.

## Note — `cutlass.Int32.__index__` + Python `range`

Unrelated to the bug but worth recording. `cutlass.Int32.__index__`
returns the stored Python int when `.value` is `int/float/bool`, and
raises `DSLRuntimeError` with a "use `range_dynamic`" hint when
`.value` is an MLIR symbolic value:

```
def __index__(self):
    if isinstance(self.value, (int, float, bool)):
        return self.value
    else:
        raise DSLRuntimeError(
            f"'{type(self.value)}' object cannot be interpreted as an integer",
            suggestion="Mark the loop as dynamic with `dynamic_expr` or ..."
        )
```

Implication: `for k_tile in range(k_tile_cnt):` in `kernel.py` only
traces because `cute.size(...)` on a partially-compile-time layout
returns an Int32 whose `.value` is a Python int at trace. If the
shape involved any symbolic dim, Python `range` would reject it — use
`cutlass.range(k_tile_cnt)` (runtime loop, body traced once) instead.

Relevant for Bug 2: when I switch to `cutlass.range` for the outer
K-loop, the body traces once and ACCUMULATE state freezing becomes
visible. Current `kernel.py` likely gets away with `range(...)` only
because the JIT input stubs have concrete zero values and the
`unroll=1` hint degrades it to a compile-time micro-unroll — worth
verifying via MLIR dump before relying on the same pattern in v0 FA4.

## Attempted fix for Bug 2 — Python `range()` unroll (insufficient)

Turned out the two-atom helper from the earlier plan was overkill —
the proper FA4-alike knob for this case is just **full trace-time
unroll of the K-tile loop**. `kernel.py` already does exactly this
(`for k_tile in range(k_tile_cnt):`) and it works: one `set(ACC, False)`
before the loop + `set(ACC, True)` after the first unrolled `cute.gemm`
→ MLIR holds exactly one `gemm(ACC=False)` at position 0 and
`gemm(ACC=True)` everywhere else. The outer persistent `while tile_idx`
re-runs the whole unrolled block once per tile, giving exactly the
per-tile "zero then accumulate" pattern we want.

Applied the one-line change in `kernel_v0_fa4.py`:

```python
-                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
+                for k_tile in range(k_tile_cnt):
```

**Result**: `0/24` still fails. The rel pattern only nudged on 2-CTA:

| K      | mode  | before  | after   |
| ------ | ----- | ------- | ------- |
|  3840  | 1-CTA | nan     | nan     |
|  3840  | 2-CTA | nan     | nan     |
| 10240  | 1-CTA | ≈ 1.0   | 1.00    |
| 10240  | 2-CTA | ≈ 1.0   | 0.96    |
| 15360  | 1-CTA | ≈ 1.0   | 1.00    |
| 15360  | 2-CTA | ≈ 1.0   | 0.98    |

2-CTA moved a hair, 1-CTA did not — which kills the "ACCUMULATE alone
is the bug" theory.

The smoking gun: `M=256 K=3840 N=3072 1-CTA` also fails (nan). That
shape has **exactly one tile per CTA** — the persistent `while` iterates
once, there is no cross-tile state reuse. So the bug fires within a
single pass through the mainloop body, not across persistent iterations.
ACCUMULATE trace-freeze only matters across iterations, so it can't
be the root cause for single-tile failures.

Kept the `range()` change since it's structurally correct (mirrors
kernel.py) and does help 2-CTA marginally — but it's not Bug 2 in
the sense of being load-bearing.

## Bug 3 — single-pass correctness failure (open)

Every shape fails, including `M=256` (1 tile/CTA). So every of the
body-level paths is broken independently of the persistent scheduler.
Two rel patterns survive:

- `rel ≈ 1.0` on 1-CTA multi-tile ↔ `y_kern ≡ 0`. Epilogue runs but
  stores zero, or tmem acc is zero at read time.
- `rel = nan` on K=3840 shapes. `y_kern` contains Inf/NaN or `|y_ref|
  = 0`. SF tmem pointer or scales misread → dequant blows up.

Likely suspects (ordered by plausibility):

1. **SFA/SFB s2t → mma sync missing**. `_mainloop_s2t_copy` lowers to
   `tcgen05.cp` which is async. MMA reads SFA/SFB from tmem; without
   a commit/wait between the s2t copy and the first `cute.gemm`, mma
   reads stale / uninitialized tmem. `kernel.py` has the same
   structural pattern without explicit sync — investigate whether
   the sync is implicit via the consumer_wait→s2t→gemm chain at PTX
   level, or whether `kernel.py` is relying on timing luck and
   `v0_fa4`'s different warp split breaks it.
2. **`pipeline_acc` phase/parity**. Fixed Bug 1 with `producer_phase=1`
   init (unhangs), but the commit/consumer_wait/release/flip sequence
   for single-stage may still be off by a half-cycle somewhere.
   Epilogue `acc_consumer_phase = Int32(0)` — consumer starts at 0,
   reads first commit (parity flips 0→1), waits for !=0 → passes.
   But `consumer_release` arrives on empty, flipping empty 1→0 → next
   producer_acquire waits for !=producer_phase. Follow the parity chain
   through a full tile.
3. **tmem layout sharing between acc and SFA/SFB**. Acc is at
   `acc_tmem_ptr..+num_accumulator_tmem_cols` (= cta_tile_shape_mnk[1]
   × num_acc_stage). SFA at `+num_accumulator_tmem_cols`. If
   `num_accumulator_tmem_cols` under-counts (e.g., 2-CTA needs double,
   or acc dtype width was miscomputed), SFA overlays part of acc, and
   s2t SFA overwrites accumulator mid-K → garbage + possible NaN on
   small K.

## Bug 3 root cause — scheduler tile-coord decode (fixed)

None of the three suspects were load-bearing. The real bug was in the
tile-index → `(m_tile, n_tile, l_tile)` decode inside the kernel body:

```python
# WRONG — `local_tile`'s .shape is flat, not nested.
gc_shape = gC_mnl.shape
num_m_cta = gc_shape[1]
num_n = gc_shape[2]
num_l = gc_shape[3]
```

The code assumed `gC_mnl.shape` was nested `((tile_m, tile_n),
num_m_cta, num_n, num_l)` and indexed `[1], [2], [3]` to pick off
`(num_m, num_n, num_l)`. Actual shape is **fully flat**:
`(tile_m, tile_n, num_m_cta, num_n, num_l) = (128, 128, 2, 24, 1)`.
So the code got `(num_m, num_n, num_l) = (128, 2, 24)` — off by one
on every index. The decoder then saw `num_n=2` and silently
misrouted every `tile_idx >= num_m_cta`.

This has nothing to do with `c_layout`'s `order=(1, 0, 2)`:
`local_tile` does **not** reorder coord modes based on the input
layout's stride order. Verified empirically by
`tmp/trace_layout_order.py` — N-major and M-major inputs both yield
`local_tile.shape = (128, 128, 2, 24, 1)`.

The right API to pick coord modes positionally is `zipped_divide`,
whose shape is **nested** `((tile_m, tile_n), (num_m, num_n, num_l))`.
Indexing with `[(0, (None, None, None))]` extracts the coord
3-tuple directly, regardless of tiler rank or input stride order.

**Fix**: mirror the host's `_compute_grid_persistent`:

```python
_c_tile = cute.slice_(self.cta_tile_shape_mnk, (None, None, 0))
_gc_zipped = cute.zipped_divide(mC_mnl, tiler=_c_tile)
num_m_cta, num_n, num_l = _gc_zipped[(0, (None, None, None))].shape
```

Cheat sheet for CuTe DSL divide variants:

| API             | `.shape`                       | structure   |
| --------------- | ------------------------------ | ----------- |
| `zipped_divide` | `((tile), (num_m, num_n, ...))`| nested      |
| `tiled_divide`  | `((tile), num_m, num_n, ...)`  | tile nested, rest flat |
| `flat_divide`   | `(tile, num_m, num_n, ...)`    | fully flat  |
| `local_tile`    | `(tile, num_m, num_n, ...)`    | fully flat (same as `flat_divide`) |

**Why `kernel.py` escaped it**: the non-persistent path uses grid
dims directly (`mma_tile_coord_mnl = (bidx // atom, bidy, bidz)`),
never decoding from a linear tile index — so the `num_n` miscount
never fires.

## Bisect — how we got here

Strategy: add `cute.printf` probe right after `tmem_load` in the
epilogue (`tile_idx==0 ∧ subtile_idx==0 ∧ warp_idx==epi[0] ∧
tidx==0`) to peek at fp32 acc values, and compare against
`y_ref[0, 0..3]` from the smoke. Expected thread-0-owned positions:
output cols `[n_tile*128 .. n_tile*128+3]` of row 0.

Data collected for `M=256 K=3840 N=3072 fp16 1-CTA`:

| tile | probe acc[0..3] | expected `y_ref` | match |
| ---- | ----------------------------- | ----------------------------- | ----- |
| 0 (n=0) | 1.226 -1.968  4.082 -7.554 |  1.227 -1.968  4.082 -7.555 | ✓ |
| 1 (n=1) | 12.557 4.744 -4.062 -10.077 | 12.555 4.746 -4.062 -10.078 | ✓ |
| 2 (n=2) | -1.346 -4.109 6.987 -1.232  | -6.684 -2.867 -3.049 -0.673 | ✗ |

MMA output was correct for tiles 0/1, wrong for tile 2. Confirmed bug
is **before MMA** (load-warp TMA source addr or SFA/SFB indexing).
Host-side diagnostics ruled out per-element corruption in epi: bad
values cluster per-N-tile, with N-tile 0 and 1 perfectly clean and
every N-tile ≥ 2 uniformly garbage (~500 bad cells per 128×128 tile).

Final probe: print `(block_idx, tile_idx, m_tile, n_tile, l_tile)`
from the load warp (lane 0, CTAs 0/1/2):

```
[sched] bid=0 tile_idx=0 m=0 n=0 l=0   ✓
[sched] bid=1 tile_idx=1 m=0 n=1 l=0   ✓
[sched] bid=2 tile_idx=2 m=1 n=0 l=0   ✗  (should be m=0 n=2)
```

→ num_n=2 inside the kernel. Root cause located.

## Bug 4 — 2-CTA m_tile unit mismatch (fixed)

After Bug 3: 1-CTA was green, but 2-CTA M=4352 shapes kept failing
(nan on K=3840, rel≈1 on K=10240/15360). M=256 2-CTA passed only
because it has `num_m_cluster=1` and never exercised the
multi-cluster decoder path.

**Root cause**: inside the kernel body, `gC_mnl` is tiled by
`mma_tiler` (256×128 under 2-CTA), so `gC_mnl.num_m = M / 256 = 17`
for M=4352. `thr_mma.partition_C(gC_mnl)` further slices the V axis
so each CTA in a 2-CTA pair owns rows 0–127 / 128–255 of each mma
tile automatically — both CTAs in a cluster must therefore use the
**same** `m_tile` value (= m_cluster, in mma-tile units).

The persistent decoder was instead computing:

```python
m_tile = m_cluster * cluster_m + block_in_cluster_coord_vmnk[2]
```

Two independent mistakes:

1. `cluster_m` was applied at the wrong level. `m_cluster` is
   already in mma-tile units, not cta-tile units; multiplying by
   `cluster_m` turns a valid `[0, 17)` index into `{0, 2, …, 32}`
   — 9 of the 17 mma tiles never covered, 8 out-of-range writes.
2. `block_in_cluster_coord_vmnk[2]` is the **N axis** of
   `cluster_layout_vmnk` (shape `((2,), 1, 1, 1)` for cluster
   (2,1) + atom `CtaGroup.TWO`), which is always 0 under (2,1).
   The per-CTA M offset lives on index `[0]` (V), not `[2]`.

Even if the intent had been "cta-tile-unit m + V offset", the
indexing target (`gC_mnl`) is in mma-tile units, so that scheme
is also wrong.

**Reference**: `kernel.py` (non-persistent 2-CTA, working) uses
`mma_tile_coord_mnl[0] = bidx // atom_thr_size` — both CTAs in the
pair share the same `[0]` and rely on `partition_C` for V.

**Fix**: drop the V offset and the `cluster_m` multiplier. Also
compute `num_m_cluster` by dividing the C tensor by `mma_tiler`
directly (not `cta_tile_shape_mnk` followed by `// cluster_m`) so
the host grid count and device `gC_mnl.num_m` use the same
tiler.

```python
_c_tile = cute.slice_(self.mma_tiler, (None, None, 0))
_gc_zipped = cute.zipped_divide(mC_mnl, tiler=_c_tile)
num_m_cluster, num_n, num_l = _gc_zipped[(0, (None, None, None))].shape
...
m_tile = mn_rem // num_n    # mma-tile units, shared by both CTAs
n_tile = mn_rem % num_n
```

For 1-CTA, `mma_tiler == cta_tile_shape_mnk` so `num_m_cluster ==
num_m_cta`; the old decoder's `m_cluster*1 + 0` and the new
`m_cluster` are numerically identical — 1-CTA stayed bit-exact
under the rewrite.

**Trace-level confirmation** (runs on any box with cutlass-dsl,
no B200 needed):

- `tmp/trace_cluster_layout.py` — shows `cluster_layout_vmnk =
  ((2,), 1, 1, 1)`, so `[2]` = 0 always under cluster (2,1).
- `tmp/trace_gc_mnl_2cta.py` — shows `gC_mnl.shape =
  (256, 128, 17, 24, 1)` for M=4352; `num_m_cta` from cta-tile
  divide = 34 = 2× the mma count.

## Result

1-CTA path: **12/12 shapes pass** (was 0/12). rel = 0.00e+00 on all
shapes — bit-exact vs fp32 dequant ref.

2-CTA path: **12/12 shapes pass** (was 2/12 after Bug 3 — only the
M=256 single-cluster shapes). rel = 0.00e+00 on all shapes.

## Perf baseline vs CUTLASS (task #39 investigation)

First pass at task #39 ("tile 256×256 + overlapping_accum"). Measured
v0_fa4 2-CTA at tile_n ∈ {128, 256} in the same Modal session (same
B200 instance, shared clock state) so tile effect isn't confounded
with run-to-run Modal variance.

```
shape                   tile (256,128)   tile (256,256)   Δ
M=256  K=3840  N=3072     1.1% MFU         1.0% MFU        −11%
M=4352 K=3840  N=3072    13.8%            13.3%            −3.5%
M=4352 K=3840  N=15360   34.0%            35.5%            +4.2%
M=4352 K=15360 N=3840    32.2%            32.1%            −0.4%
M=4352 K=10240 N=3072    25.8%            26.6%            +3.1%
```

Takeaways:

- Tile-only gain is ≤ +4% on K-heavy / N-heavy shapes and **negative**
  on small M (M=256) or small K·N. Not worth forcing (256,256) as
  the 2-CTA default — keep (256,128), expose (256,256) opt-in via
  `launch_v0(tiler_mn=(256, 256))`.
- **Overlapping_accum ≠ tile_n=256.** CUTLASS's own `_compute_stages`
  sets `num_acc_stage = 1 if tile_n == 256 else 2` — identical to
  ours. CUTLASS hits 63% MFU on tile_n=256 with `num_acc_stage=1`
  too; the 20pp gap to our 32-34% at matched tile is **not** from
  an acc ping-pong we're missing.
- True overlapping_accum (`num_acc_stage=2`) lives only at
  tile_n=128 (tmem budget: 128×2 acc = 256 cols fits; 256×2 = 512
  doesn't). Would need kernel-body surgery (dynamic stage index in
  MMA + epilogue; phase flip every 2 advances) — filed back under
  task #41 for now since tile_n=128 is where it applies.
- Real MFU gap work (task #41): pipeline depth / TMA multicast
  discipline / s2t issue rate / warp-spec thread counts — not
  the tile lever.

Code state: assertion `mma_tiler_mn[1] == 128` under 2-CTA is
lifted so (256,256) compiles, but `_pick_tiler_v0` still returns
(256,128) by default. Smoke is 24/24 bit-exact with default
tile; one-off `tiler_mn=(256, 256)` runs were also 24/24 in an
earlier iteration. Opt-in kept live for future tuning.

## Lessons

- `cute.local_tile(X, tiler, (None,...))` returns a **flat** shape,
  same as `flat_divide`. It does **not** reorder based on the input
  layout's stride order (my first guess was wrong — verified in
  `tmp/trace_layout_order.py`). Use `zipped_divide` for positional
  coord-mode indexing.
- Pattern applies to any persistent scheduler where tile-coord decode
  runs on the device from a flat tile index. Host-side grid computation
  (which used `zipped_divide`) was correct; the device-side decode
  was off by one on every index and got silently wrong values.
- Under 2-CTA `CtaGroup.TWO`, the **per-CTA M-within-cluster
  position** lives on `cluster_layout_vmnk`'s `V` axis (index 0),
  not `M` / `N` (indices 1/2). V = atom_thr_shape for tcgen05 2-CTA;
  the residual M axis collapses to 1 whenever `cluster_m ==
  atom_thr_size`. Always cross-check by tracing
  `cluster_layout_vmnk.shape` before reading it positionally — same
  class of positional-indexing trap as Bug 3.
- When tiling C / AB tensors, pick the tiler that matches the
  consumer's expected coord granularity. Under 2-CTA that is
  `mma_tiler` (cluster-wide M), **not** `cta_tile_shape_mnk`;
  `partition_C` / `partition_A` / `partition_B` handle the V split
  for you.
- `cute.printf` bisect — three rounds, each narrowing by one order of
  magnitude — took the bug from "epi is broken" (wrong) to "MMA on
  tile 2 is broken" to "scheduler misdecodes tile 2". Worth keeping
  the probe harness (`tmp/probe_gemm_v0_fa4.py` +
  `modal_app.py::gemm_v0_fa4_probe`) around.
