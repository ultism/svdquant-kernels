# Ascend / AscendC / PTO ISA gotchas

Traps that cost real debugging time on the Ascend 910B (a2a3) cube +
vec mix-mode path. Each entry is a *silent* misbehavior — the kernel
compiles, runs, and produces deterministic-looking numbers that are
wrong. Add new entries here when you find one, before paging the
context out.

Organized roughly by hardware layer (top) → PTO ISA layer (bottom).

---

## Hardware: cube↔vec handoff is L2-resident, not HBM round-trip

A2/A3 cube and vec have physically isolated on-chip storage (L0C vs
UB) — there is no GPU-SMEM-style addressable shared scratch. So
cross-core data goes through GM **addresses**. But L2 caches those
addresses for both cube's FIX-pipe TSTORE and vec's MTE2 TLOAD; cube
just wrote, vec immediately reads, the line is hot in L2 and never
hits HBM.

Implication for design:

- Size cube↔vec FIFO buffers by **L2 working set**, not GM offset
  distance. Comparing slot bytes against HBM bandwidth is the wrong
  axis.
- The slot-count upper bound is L2 capacity minus what act/wgt tiles
  need re-resident; not GM capacity.
- PTO FA reference: `pto-isa/tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp`
  uses `qkGlobalTensorNBuffers = 1 + qkPreloadNum = 6` exactly to
  keep the cube→vec ring inside L2.
- svdquant `gemm_w4a4` per-K-block int32 acc handoff at BM=128,
  BN=256 → 128 KB/slot. 4–8 slots fits L2 comfortably; the
  optimization axis is keeping act+wgt+ring total ≤ L2, not
  reducing slot count.

If you can't model L2 hit-rate by inspection, run cce profiler /
hccl-perf for the L2 hit ratio before tuning the ring layout.

## Hardware: cube minimum addressable = 1 byte (int8); INT4 is mad-internal

PTO `TLoad` / `TExtract` whitelists accept `int8_t / half / bf16 /
float`. They do **not** accept `pto::int4b_t`, `int4b_x2_t`, or any
4-bit-typed tile. This is **not** a PTO oversight — Ascend cube
L1/L0 data movement minimum addressable unit is **1 byte**. INT4
values have no offset / pointer representation; nibble decoding
happens only inside `mad_s4`, whose ABI takes `__ca__/__cb__ void*`
and unpacks 2 signed INT4s per byte at issue time.

Implication:

- The W4A4 cube path uses `Tile<Mat, int8_t, M, K_packed>` +
  `TileLeft/Right<int8_t, ...>` + `TileAcc<int32_t, M, N>` + raw
  `mad_s4(c, void*, void*, m, k_logical, n, ...)`. `K_packed` is
  bytes (= `K_logical / 2`). This is canonical, not a workaround.
- Do not use `uint8_t` for L1/L0 tiles. TLoad accepts it but
  TExtract's whitelist omits it. Bit-pattern is identical to
  `int8_t` — pick `int8_t` to pass both gates.
- Do not propose an `int4b_x2_t` twin type / TMATMUL_S4 wrapper to
  PTO upstream. Even if accepted, it would be cosmetic typing over
  int8 storage — runtime behavior unchanged.
- ascale / wscale INT4 dequant is per-K-block (default 64 nibbles)
  on the vec side. K_packed=32 → one mad_s4 issue ↔ one K-block.

## Hardware: `mad` is a coarse macro MAC, not a warp-level fragment

Ascend cube `mad` (and `mad_s4`) is a **macro MAC**: one issue
takes m/k/n up to `MMAD_MAX_SUPPORT_LENGTH = 4095`
(`pto-isa/include/pto/npu/a2a3/TMatmul.hpp:17`); the cube array
finishes the entire tile in hardware. NVIDIA `mma.sync.aligned.*`
is a warp-level fragment instruction (typical m16n8k64 for INT4);
a full GEMM tile needs tens-to-thousands of mma issues.

Concrete differences:

- Work per issue: one `mad` ≈ hundreds-to-thousands of `mma`s.
- K direction: NVIDIA expands K *outside* the instruction
  (scheduler streams multiple `mma`s); Ascend expands K *inside*
  (`mad`'s k parameter directly hits thousands).
- Pipeline granularity: NVIDIA worries about warp-scheduler ILP;
  Ascend worries about 4-stage TMATMUL/TLOAD/TEXTRACT/TSTORE
  overlap (see `pto-isa/kernels/manual/a2a3/gemm_performance/README_zh.md`).
- Packing / reorder cost amortizes over the entire macro issue,
  not per fragment — can be cheaper than NVIDIA fragment-level
  amortization.

Implications for design:

- A `TMatmulS4`-style wrapper should NOT expose "warp-fragment
  shape" parameters. The right signature mirrors PTO's `TMatmul`:
  `(cTile, aTile, bTile)` + a thin `mad_s4(...)` adapter. Fractal
  layout is hardware-determined inside the cube array; the wrapper
  only handles ABI + static checks.
- Tile selection anchor: PTO GEMM example uses
  `[baseM, baseN, baseK] = [128, 256, 64]` for fp16 (saturates L0B
  32 KiB). INT4 halves element bytes, so `[128, 256, 128]` is
  theoretically headroom — confirm against CCE doc INT4 fractal
  constraints before picking.
- MFU comparison against nunchaku: single-point single-point is
  meaningless. Ascend side wants pipeline-stage occupancy
  (TMATMUL/TLOAD/TEXTRACT/TSTORE %) and msprof, a different
  axis from NVIDIA `ncu sm__pipe_tensor_subpipe_*`.

## Hardware: don't reference nunchaku for the Ascend cube ABI

nunchaku's INT4 GEMM is hand-written PTX (see `gpu.md` § perf
context). It targets NVIDIA `mma.sync.aligned.*.s4.s4.s32`,
fragment layout per NVIDIA tcgen / ldmatrix conventions. That is a
completely different hardware ABI from the Ascend cube unit's L0A
/ L0B fractal layout (described by `pto-isa CheckMadValid`:
left-RowMajor/ColMajor SFractal, right-ColMajor/RowMajor SFractal,
Acc-RowMajor SFractal, K aligned to cube preferred granularity).

Rule:

- Ascend INT4 packing / fractal questions: CCE intrinsic docs +
  PTO INT8 reference path + AscendC docs. Skip `tmp/nunchaku/`.
- The reverse also holds: NVIDIA NVFP4 / SM_100 tcgen path
  questions: don't reach into PTO / AscendC.
- Boundary: math (GEMM + LoRA epilogue + quant math) is
  cross-backend referencable — that's why CLAUDE.md says "keep
  math and tensor shapes in sync". Hardware ABI (instructions,
  fragment layout, packing order, scale arrangement) must follow
  the target hardware's own docs.

---

## Decision: W4A4 cube path uses raw `mad_s4` inside svdquant, not a PTO wrapper

Settled 2026-05-04. The earlier "add a TMATMUL_S4 wrapper to PTO"
plan (RFC #332) is closed. Path inside svdquant's Ascend pod:

- A/B matrix tiles: `Tile<Mat, int8_t, M, K_packed>`,
  `TileLeft/Right<int8_t, ...>` — PTO byte-typed path, native.
- Cube issue: inline `mad_s4(c, (__ca__ void*)a.data(),
  (__cb__ void*)b.data(), m, k_logical, n, unitFlag,
  false /*kDirAlign*/, src, init)`, bypassing the PTO type wrapper.
- Everything else (activation TLoad, scale broadcast, bias, LoRA
  epilogue, TStore) continues to go through PTO abstractions.

Why the wrapper path was abandoned:

- A2/A3's `TLoad` / `TMov` / `TExtract` leaf CCE intrinsics are
  byte-primitive-typed (`signed char *`, `__bf16 *`, …). PTO's
  existing `int4b_t` is a vec-only struct (only works via
  `TCvt`'s `is_same_v<DType, int4b_t>` specialization on
  conversions). Cube path has no equivalent specialization;
  struct types are rejected at the intrinsic boundary.
- Making `TMATMUL_S4` real requires a dtype-aware
  TLoad/TMov/TExtract pass on a2a3 (mirroring how a5 added FP4) —
  30+ stride patches plus byte-primitive intrinsic wiring. Not a
  single-commit change.
- PTO SIG response timeline (#115 silent 3 months; #332
  self-closed in < 4 days) made waiting infeasible.
- svdquant is an operator, not a public bottom-layer library —
  input validation lands in PyTorch / vLLM, not inside the kernel.
  PTO's `CheckStaticMadS4` static guard is not a value-add here.
- The internal-raw-cce pattern matches AscendC's own `MmadCal` s4
  branch in `dav_c220/kernel_operator_mm_impl.h` — standard
  practice, not a hack.

Apply:

- When writing mmad calls in the Ascend pod, inline `mad_s4(...)`
  directly. Do not search for / create a PTO wrapper.
- Do not patch PTO's a2a3 to add dtype-aware INT4 plumbing. That
  is SIG's surface and is outside the svdquant scope.
- If SIG follow-up asks: backup branch `feat-mad-s4` on
  `qq_42927189/pto-isa` (gitcode) and `ultranationalism/pto-isa`
  (github), commit `724b973a`, is a single-commit unblock
  (`int4b_x2_t` = `uint8_t` alias + `mad_s4` `kDirectionAlign`
  parameter fix + ST testcase). Clean PR ready but not pushed by
  default.
- ABI sanity check anchor: if a raw `mad_s4` call gets argument
  count wrong, the CCE ABI is 10 parameters. Cross-reference
  AscendC `dav_c220/kernel_operator_mm_impl.h` `MmadCal` s4 branch
  for the exact parameter order.

---

## PTO: `TLoad` of `ColMajor [N, 1]` reduce tile from GM only loads 1 element

Discovered Phase 3a 2026-05-11 cycle 15.

**`TLoad` of `Tile<Vec, T, N, 1, BLayout::ColMajor, N, 1>` from any
`GlobalTensor` (both ND-layout AND DN-layout) only writes element
`[0, 0]` of the UB tile. Elements `[1..N-1, 0]` retain whatever was
in UB before the load (zero if just zeroed, otherwise stale).**

The kernel compiles cleanly, raises no `TASSIGN` / `TLOAD` /
`PTO_ASSERT`, and produces a deterministic-looking number at
element 0. The bug is discoverable only by dumping the post-`TLoad`
/ post-`TCvt` tile to GM and comparing against expected per-row
values.

Why: PTO's `ColMajor [N, 1]` tile is intended as the **computed
result** of a row-wise reduction (`TRowMax`, `TRowSum`, `TSub`
across rows) or as a **constant-fill destination** of `TExpandS`.
It is not a supported `TLoad` destination for variable per-row
scalars from GM. FlashAttention's
`ReduceTileF_T = Tile<Vec, float, Vec_S0, 1, ColMajor, Vec_S0, 1>`
(`pto-isa/tests/npu/a2a3/src/st/testcase/tfa/`) is the canonical
user — and it **never** `TLoad`s from GM; `exp_max` is computed in
UB via `TReshape + TSub + TExp` from a separate RowMajor input.

Downstream symptom in Phase 3a: `TRowExpandMul` with this "loaded"
ColMajor `[32, 1]` src1 silently used garbage scalars for rows
1..31, producing per-row scale multipliers like 5.92e-06 instead
of 0.07135.

Fix template for per-row variable scales loaded from GM:

1. Load as a RowMajor flat row:
   `Tile<Vec, T, 1, N, RowMajor, 1, N>` with
   `GlobalTensor<T, Shape<1,1,1,1,N>, Stride<1,1,1,N,1>>`. Same
   pattern as wscale loads, well-tested.
2. After `TCvt` to fp32, broadcast manually to `[N, 8]` RowMajor:
   ```cpp
   vbrcb(broadcast_ub_ptr, ascale_f32_ptr,
         /*dstBlockStride=*/1, /*dstRepeatStride=*/8,
         /*repeats=*/CeilDivision(N, 8));
   pipe_barrier(PIPE_V);
   ```
   Each row `r` ends up as `[s_r] × 8` (one 32-byte block).
3. Feed the RowMajor `[N, 8]` to `TRowExpandMul` as src1. PTO
   takes the RowMajor src1 path (assertion `src1ValidCol ==
   32/sizeof(T) = 8`), skips its internal vbrcb scratch dance,
   goes directly to vmul.

Alternative if `TRowExpandMul` still misbehaves on the RowMajor
path: broadcast to full `RowMajor [N, kBN]` and use plain `TMul`
elementwise. Costs `N*kBN*4` UB bytes but removes PTO's expand
machinery from the equation.

Diagnostic: when per-row scaled output looks "scaled by something
random", dump the reduce tile post-`TCvt` / pre-`TRowExpandMul`
to a side GM region. If only row 0 matches expected → it's a
ColMajor `[N, 1]` `TLoad` bug, not a `TRowExpandMul` or `vbrcb`
bug.

Generalization: don't `TLoad` into `ColMajor` reduce tiles. Load
flat as `RowMajor [1, N]`; broadcast / reshape in UB to whatever
shape downstream ops need.

## PTO: `TRowExpand` leaves the vector mask register contaminated

Discovered Phase 3a 2026-05-11 cycle 17.

**After `pto::TRowExpand` (or any PTO op whose internal `vbrcb`
sets a count-mode mask without restoring norm mode), the vector
mask register is left in a state that causes downstream vector ops
to silently process only a fraction of their declared `repeats`
argument.**

Mechanism in
`pto-isa/include/pto/npu/a2a3/TRowExpandBinOp.hpp`'s
`TRowExpandBinaryNormModeTail`:

```cpp
if (DstRowStride < elementsPerRepeat || ...) {
    SetContMaskByDType<T>(validCol);     // explicit mask setup
    Op::RowExpandBinInstr(...);
    SetFullVecMaskByDType<T>();
} else {
    for (i = 0; i < numLoop; i++) {
        Op::RowExpandBinInstr(...);      // NO mask setup
        ...
    }
}
```

When `DstRowStride ≥ elementsPerRepeat`, the else-branch executes
without `SetContMask`. It inherits whatever mask the caller left
in place. If the caller just ran `TRowExpand`, the internal
`vbrcb` left a count-mode mask of `ceil(target_size / 8)` — so
`vmul` silently processes only the first 4 repeats (for
target_size=32), leaving rows 0..3 of each AIV's row band
untouched.

Fix template:

```cpp
pto::TRowExpand(bcast_tile, flat_tile);
pipe_barrier(PIPE_V);            // PIPE_V serializes anyway,
                                  //   but documents intent
set_mask_norm();                  // restore norm mode
set_vector_mask(-1, -1);          // full vec mask for dtype
pto::TRowExpandMul(dst, src0, bcast_tile);
```

Apply:

1. Wrap any `pto::TRowExpand`, `pto::TBrcb`, or other
   broadcast-style PTO op in this pattern before chaining into
   another vector op (`vmul`, `TRowExpandMul` on RowMajor path,
   anything that enters `NormModeTail`'s else-branch).
2. `TColExpandMul` is suspected to have the same risk after a
   contaminated mask; apply the same reset.
3. Note: this is independent of the mask-reset documented for
   AIV mix-mode kernel entry (PTO issue #218). The kernel-entry
   reset handles the *initial* mask state; this rule handles
   *during-execution* contamination.

Diagnostic: when a chained PTO vector op covers most rows/cols of
its tile correctly but a fixed prefix is wrong/skipped/zero,
suspect mask contamination. Skipped count typically =
`ceil(target_size / 8)` of the broadcast op that left mask in
count mode.

## PTO: AIV K-loop reusing a partial UB region needs V→MTE2 cross-iter sync

Discovered Phase 3a 2026-05-11.

On AscendC mix-mode AIV, when a vec K-loop reuses the same UB
region across iterations for a partial (`TLoad partial_int32` →
`TCvt i32→f32` → `TRowExpandMul / TColExpandMul` → `TMov / TAdd`
→ reuse on next iter), you **must** add an explicit
PIPE_V → PIPE_MTE2 cross-iter flag.

The race: `PIPE_MTE2` (TLoad) and `PIPE_V` (TRowExpandMul etc.)
are independent pipes. Without explicit V→MTE2 sync, iter N+1's
TLoad may fire before iter N's PIPE_V writes drain. The PIPE_V
writes then land **after** the new TLoad, overwriting the
freshly-loaded int32 bytes with stale fp32 dequant bit patterns.
Iter N+1's `TCvt i32→f32` reinterprets the fp32 bit patterns as
int32 — e.g. `0.1 = 0x3DCCCCCD = 1036831949` reads as int32
1.036e9, then casts back to fp32 1.036e9. Output magnitude near
INT32_MAX is the dead giveaway.

The MTE2→V flag at the *top* of each iter only enforces "V waits
for MTE2's TLoad". The reverse direction ("next iter's MTE2 waits
for prior iter's V drain") needs its own flag. Other UB regions
(running accumulator, scales) don't trigger this because they're
only written by V and never re-loaded by MTE2 — the partial
region is the unique cross-pipe-write spot.

Fix template:

```cpp
// Seed before the K-loop so iter 0's wait is satisfied trivially
set_flag(PIPE_V, PIPE_MTE2, EVENT_ID_X);

for (kb = 0; kb < kNumKBlocks; ++kb) {
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID_X);  // gate next TLoad
                                                //   on prior V drain
    TLoad(partI32, ...);
    // ... TCvt, TRowExpandMul, TColExpandMul, TMov/TAdd on PIPE_V ...
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID_X);   // signal V drained
}
wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID_X);      // drain seed on exit
```

Pick an `EVENT_ID` not already used by other flags on the AIV
section (Phase 3a used `EVENT_ID2`; `0/1` were taken).

Diagnostic: `TStore` the post-`TCvt` fp32 tile to a side GM,
compare against `partial_int32.float()` from the caller. Iter 0
looks correct; iter 1+ is off by ~INT32_MAX when this race fires.

Generalization: any UB region that is BOTH a `PIPE_MTE2` TLoad
target AND a `PIPE_V` output target across loop iterations needs
V→MTE2 sync. UB regions written by V only (running accumulator,
scratch) don't.
