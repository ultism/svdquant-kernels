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

## Next action

1. Add a `cute_kernels/gemm_w4a4/_accum_atoms.py` helper exposing
   `make_init_and_accum_atoms(tiled_mma)` returning the pre-baked pair.
2. Replace the in-loop `tiled_mma.set(ACCUMULATE, …)` calls with
   runtime selection of `mma_atom_init` (first kblock of first K-tile
   of each persistent tile) vs `mma_atom_accum` (everywhere else).
3. Re-run `gemm_v0_fa4_smoke` on Modal; expect `rel ≈ 1.0` cases to
   flip to `OK` first. If `rel = nan` cases persist, pursue as Bug 3.
