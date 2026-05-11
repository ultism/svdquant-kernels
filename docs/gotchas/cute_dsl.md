# CuTe DSL (cutlass-dsl Python) gotchas

Traps that cost real debugging time on the SM_100 / SM_103 CuTe DSL
path. Most are silent: code traces, lowers, runs, and produces wrong
numbers — not exceptions. Add new entries here when you find one,
before the context falls out.

---

## Python `if` inside `@cute.jit` traces BOTH branches — wrap with `const_expr`

Inside `@cute.jit` and `@cute.kernel` bodies, a plain Python
`if self.some_python_bool:` does **not** resolve at trace time.
The DSL's AST preprocessor turns it into `scf.if` and traces both
branches. If the "false" branch references attributes that only
exist when the condition is true (e.g. `self.lora_mma_tiler`,
built only when `self.enable_lora`), the trace raises at the
inactive branch.

Fix: wrap the condition with `cutlass.const_expr(...)`:

```python
if cutlass.const_expr(self.enable_lora):
    ...
else:
    ...
```

`const_expr` accepts a Python value and returns it unchanged; the
AST preprocessor recognizes it as a compile-time marker and emits
no `scf.if` — only the truthy branch is traced.

Apply:

- Any in-`@cute.jit` / in-`@cute.kernel` branch gated by a Python
  bool on `self` or any other compile-time-known value must use
  `const_expr`.
- Ternaries inside `@cute.jit` need it too — easiest to rewrite as
  statement-form `if cutlass.const_expr(cond): x = a; else: x = b`.
- No need to wrap in plain Python helpers (`_setup_attributes`,
  `_compile`, etc.) — those aren't traced, and `const_expr` is a
  no-op on a Python bool anyway.

Reference pattern: `dense_blockscaled_gemm_persistent.py`
`make_and_swap_ab` uses `if cutlass.const_expr(...)` for the
dtype swap path.

First hit: `gemm_w4a4` v1 development — v0 compile path started
failing after v1 LoRA setup was added under a bare
`if self.enable_lora:`. The preprocessor entered `then_block_1`
for v0 (`enable_lora=False`) and tried to build
`cute.make_ordered_layout` with `self.R == 0` → positive-shape
assertion fired.

## `local_tile` / `flat_divide` / `tiled_divide` / `zipped_divide` return different nesting

For tensor `(M, N):(1, M)` (or any stride pattern — these APIs do
NOT reorder modes based on layout order) split by tiler `(4, 8)`:

| API             | `.shape`                   | structure                  |
|-----------------|----------------------------|----------------------------|
| `zipped_divide` | `((4, 8), (num_m, num_n))` | nested                     |
| `tiled_divide`  | `((4, 8), num_m, num_n)`   | tile nested, rest flat     |
| `flat_divide`   | `(4, 8, num_m, num_n)`     | fully flat                 |
| `local_tile`    | `(4, 8, num_m, num_n)`     | fully flat (= flat_divide) |

Apply: to index coord modes positionally (e.g. decoding a flat
`tile_idx → (m_tile, n_tile)` in a persistent scheduler), use
`zipped_divide` and read `zd[(0, (None, None, ...))].shape` —
that gives the coord tuple directly, regardless of tiler rank or
input stride order.

**Do NOT** read `local_tile(X, (128, 128), ...).shape[1]`
expecting it to be `num_m_cta` — it's `tile_n`.

Real trip hazard: scheduler decode in
`cute_kernels/gemm_w4a4/kernel_v0_fa4.py` used
`gC_mnl = local_tile(mC_mnl, (128, 128), (None, None, None))`
then read `gc_shape[1], [2], [3]` expecting
`(num_m, num_n, num_l)`. Actual shape was
`(128, 128, 2, 24, 1)`, so it got `(128, 2, 24)` — misrouted
every `tile_idx >= num_m_cta`. Fix: use `zipped_divide`, then
unpack the coord 3-tuple.

Verification trace: `tmp/trace_layout_order.py` (trace-level, runs
on any box with cutlass-dsl). Same flat shape comes out for
N-major and M-major inputs — the "layout-order-reorders-modes"
hypothesis was wrong; it was just indexing the wrong flat slot.

## 2-CTA `cluster_layout_vmnk[0]` is the per-CTA M position (V), not `[2]`

Under `cluster_shape=(2, 1)` + `CtaGroup.TWO`, the cluster layout
factors into `(V, M, N, K)`:

- `V` = atom_thr_shape size (2 under `CtaGroup.TWO`, 1 under `ONE`)
- `M` = cluster_shape_m / atom_thr_size (residual M after V tiling)
- `N` = cluster_shape_n
- `K` = 1

For the common 2-CTA M-major pair `cluster_shape_mn = (2, 1)`:

```
cluster_layout_vmnk.shape = ((2,), 1, 1, 1)
rank=0 → (0, 0, 0, 0)    ← leader CTA
rank=1 → (1, 0, 0, 0)    ← follower CTA
```

The per-CTA M-within-cluster position lives at **index [0]** (V).
Indices `[1]/[2]/[3]` are all degenerate (size 1). Reading `[2]`
as "M-within-cluster" — an easy mistake because it's the *N* axis
of the *problem* — gives 0 for both CTAs and breaks persistent
tile decode under multi-cluster.

The tiler side:

- `gC_mnl` is tiled by `mma_tiler` (256 rows, spans both CTAs), so
  `gC_mnl.num_m = M / mma_tiler_m` (= M/256 under 2-CTA).
- `thr_mma.partition_C(gC_mnl)` slices V and hands each CTA in the
  pair its own 128-row half **at the same num_m index**.

So both CTAs in a 2-CTA cluster use the **same** `m_tile` index —
the V split is handled by `partition_C`, not by offsetting
`m_tile`. A persistent decoder must emit `m_tile` in *mma-tile
units* (= `m_cluster`), not cta-tile units.

Anti-pattern (broke v0_fa4 Bug 4):

```python
m_tile = m_cluster * cluster_m + block_in_cluster_coord_vmnk[2]
                                                            # ^^^ N axis, wrong
```

Right:

```python
m_tile = m_cluster   # mma-tile units, shared by both CTAs
```

And compute `num_m_cluster` by dividing `C` by `mma_tiler`
directly, not `cta_tile_shape_mnk` followed by `// cluster_m` —
same number when M is a clean multiple, but the former matches
device-side `gC_mnl.num_m` one-to-one.

Reference anchors:

- `tmp/trace_cluster_layout.py` — prints
  `cluster_layout_vmnk.shape` and `get_flat_coord` for ranks 0/1.
- `tmp/trace_gc_mnl_2cta.py` — confirms `gC_mnl.num_m = M/256`
  under 2-CTA tiler for M=4352 (= 17, not 34).
- Non-persistent counterpart: `kernel.py:956-960` uses
  `bidx // atom_thr_size` for the shared mma M coord.

## 2-CTA `TiledCopy.partition_D` is adaptive — feeding it cluster-size aux tensors silently misaligns

Under `CtaGroup.TWO`, `tiled_mma.thr_id.shape` has a size-2 V dim.
`tiled_mma.get_slice(v_coord)` + `partition_{A,B,C}` automatically
V-splits any cluster-size input into per-CTA halves.

`TiledCopy.thr_id` (e.g. built from
`sm100_utils.get_tmem_load_op(cta_tile_shape_mnk, ...)`) has **no
V dim** — it's a thread layout over the atom's tile.

`TiledCopy::partition_D(dtensor)` calls `tidfrg_D(layout)`, which
factors the **input layout** into `(tid, frg, rest...)`:

- `tid` = TiledCopy atom's thread dim
- `frg` = per-thread fragment
- `rest` = excess outer dims

So `partition_D` **is adaptive** — it doesn't misalign silently in
the small-input case; it surfaces excess capacity as **outer
"rest" modes**. That's the subtle trap. When you feed a
cluster-size `(256, 128)` tensor to a TiledCopy built for
`cta_tile_shape_mnk = (128, 128)`:

- `tidfrg_D` factors M = `(128_tid, 2_rest)`.
- Result has an **extra outer M mode of 2** the atom didn't
  anticipate. No crash — but it misaligns with the downstream
  tAcc partition.

Failure shape (v2_fa4 Bug, commit b251149):

1. `tTR_rAcc` (from per-CTA 128×128 `tAcc`) has `N` elts/thread.
2. `tTR_cC` (from cluster `(256, 128)`) has `2N` elts/thread — an
   extra `m_outer = 2` ring.
3. `group_modes(tTR_cC, 3, rank)` folds `m_outer` into the
   subtile dim → subtile count appears 2× actual.
4. The subtile loop is bounded by `tTR_tAcc`'s (correct) subtile
   count, so it reads only the first half of `tTR_cC`.
5. The first-half subtile's "i-th element for this thread" sits at
   a different `(m, n)` than `tTR_rAcc`'s i-th element.
6. N coord extracted from `tTR_cC_sub[i][1]` doesn't match the
   real N position of `tTR_rAcc[i]` → wrong `sWC[n]` multiplied
   → ~0.7 rel err. Trace compiles, runs.

`partition_D` vs `retile_D`:

- `partition_D(full_tensor)` — **adaptive**: reads input layout,
  factors `(tid, frg, rest)`. Surfaces excess as rest modes.
  Requires matching input shape to avoid rest-mode surprise.
- `retile_D(already_partitioned_tensor)` — **restrictive**: takes
  a per-thread partitioned tensor (typically from another
  TiledCopy), re-views it into this TiledCopy's dst form. No
  layout adaptation. This is what
  `tiled_copy_r2s.retile(tTR_rAcc).load()` does in v0/v1/v2 epi.

Apply:

- Aux tensors headed for `thr_copy.partition_D` (identity tensors,
  coord tensors, per-row/col scale arrays): build at
  **`cta_tile_shape_mnk`**, not `mma_tiler`. No rest-mode
  surprise; shape matches tAcc downstream.
- Equivalent alternative: build at `mma_tiler` then pre-partition
  through `thr_mma.partition_C` first — uses the V-aware route to
  do the 2-CTA split, then the V-less copy route. Extra
  indirection; the direct form is cleaner.
- When in doubt: print rank/shape before and after `partition_D`.
  If the rank differs from what tAcc gives, you have a rest-mode
  mismatch.

Reference anchors:

- `cute_kernels/gemm_w4a4/kernel_v2_fa4.py:1400-1426` — fixed v2
  epilogue uses `cta_tile_shape_mnk[0:2]`.
- Commit `b251149` — first Modal run 58/88 (30× 2-CTA wc BAD at
  rel ~0.7); fix flipped to `cta_tile_shape_mnk` → 88/88.
- `kernel_v2_fa4.py:996-1006` — `tCgC =
  thr_mma.partition_C(gC_mnl)` is the V-aware pattern that
  accepts cluster shape.

## `num_acc_stage` is tied to `tile_n`, not "tile size" — 256×256 disables overlapping_accum

SM_100 TMEM = 512 cols max. NVFP4 block-scaled MMA uses:

- Accumulator: `cta_tile_n * num_acc_stage` cols
- SFA: `(cta_tile_m / 32) * 4` cols (small, single-stage)
- SFB: `(cta_tile_sfb_n / 32) * 4` cols (small, single-stage)

At 2-CTA `tile_m = 256`, `cta_tile_m = 128`, so SFA ≈ 16 cols, SFB
≤ 32 cols. Acc budget ≈ 464 cols.

- `tile_n=128, num_acc_stage=2` → 128·2 = 256 cols. Fits.
- `tile_n=256, num_acc_stage=2` → 256·2 = 512 cols. **Busts**
  (after counting SF).
- `tile_n=256, num_acc_stage=1` → 256 cols. Fits.

So overlapping_accum (ping-pong between two acc TMEM buffers,
hiding epilogue latency under the next tile's MMA) is **available
only at tile_n=128**. Picking `tile_n=256` forces single-stage acc.

CUTLASS does the same — `tmp/cutlass_bs_gemm_persistent.py:1662`:

```python
num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2
```

CUTLASS's best 2-CTA 256×256 config runs **without**
overlapping_accum. Its bigger-tile win is from compute density
(bigger MMA per tile, fewer tile-boundary stalls, less
epilogue-launch overhead per FLOP), not from acc ping-pong.

Apply:

- Any "enable overlapping_accum at 256×256" framing is hardware-
  impossible on the current TMEM budget.
- Tile promotion (256×128 → 256×256) and overlapping_accum
  (num_acc_stage 1→2 at tile_n=128) are **separate optimization
  tracks**:
  - Tile promotion: modest ≤4% MFU gain on big K·N; loses
    3-11% on small M / small K·N. Opt-in, not default.
  - Overlapping_accum: needs kernel-body surgery (dynamic TMEM
    stage index on MMA + epilogue warps; phase flip every 2
    advances, not every 1). Useful but doesn't compose with
    `tile_n=256`.
- When measuring default tilers, run both back-to-back in the
  same Modal session — cross-run variance on Modal B200 is
  15-25%, easily masks real ±4% tile effects. See
  `tmp/bench_gemm_v0_fa4_tile_ab.py` for the A/B harness.

Reference anchors:

- `docs/kernels/gemm_w4a4_fa4_v0_bringup.md` § "Perf baseline vs
  CUTLASS" — numbers from the A/B run.
- `cute_kernels/gemm_w4a4/kernel_v0_fa4.py:_pick_tiler_v0` —
  default kept at `(256, 128)`;
  `launch_v0(tiler_mn=(256, 256))` opt-in.
