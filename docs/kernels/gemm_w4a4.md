# gemm_w4a4 — CuTe DSL design

CUDA path of the compute-bound half of SVDQuant:

```
y = (scaled_mma(act_nvfp4, wgt_nvfp4) + lora_act_in @ lora_up) * wcscales + bias
[optional]  qout, oscales = nvfp4_quant(y / smooth_next)
```

Target: Blackwell SM_100 / SM_103 (`tcgen05` scaled-MMA, TMEM, 2-CTA).
Reference host launcher: `tmp/nunchaku/src/kernels/zgemm/gemm_w4a4.cu:34-105`.
Ref math (ground truth for tests): `baseline/kernels/gemm_w4a4/ref.py`.

Two distinct MMA atoms run against the **same** fp32 accumulator:

| path | atom                        | A dtype                | B dtype                | scale                | reduction |
| ---- | --------------------------- | ---------------------- | ---------------------- | -------------------- | --------- |
| main | `mxf4nvf4.block_scale`      | NVFP4 (E2M1)           | NVFP4 (E2M1)           | FP8 E4M3, vec 16     | K         |
| lora | `mma.sync` (dense)          | fp16 or bf16           | fp16 or bf16           | —                    | R         |

The decision to interleave (β) — rather than run the two serially (α) —
comes from measurement, not intuition. See "Why β" below.

---

## 1. Why β (interleaved), not α (serial)

Measured on B200 via `torch.profiler` CUPTI Activity (kernel wall time,
launch overhead stripped):

- REF-sat baseline 4096³ cuBLAS fp16/bf16: **63–66 %** of 2250 TFLOPS
  dense peak. That's the achievable tensor-pipe saturation ceiling on
  this part, not 100 %.
- LoRA sub-problems (all six ZImage shapes, both dtypes): `sat / REF-sat`
  ranges **0.05 to 0.54**. Every shape sits at or below the "β
  required" border (≤ 0.5).

Why so low: the LoRA MMA has too few atoms to keep `tcgen05`'s
in-flight pipeline full. For R=32, `R_atoms = R / LORA_K = 2`. The
tensor pipe's async-issue depth is 4–8; two atoms can never saturate
it. Even R=256 → 16 atoms barely covers the depth and nothing more.

α (serial) cost  = `t_main_alone + t_lora_alone`
β (interleaved) cost ≈ `t_main_alone + fair_share(lora_atoms)`

`t_lora_alone` is inflated by `1 / sat_ref ≈ 2 – 20×` over its fair-share
cost, because running LoRA alone wastes the pipe. β closes that gap by
never letting LoRA run alone.

Worst-case shape (M=4352, N=3840, R=128): α wall-time ~50 % higher
than β. Not a margin worth giving up.

---

## 2. The β interleave pattern

One MMA warp owns the issue stream. Main K-loop runs as normal; LoRA
atoms are **sprinkled uniformly** into the same stream:

```python
stride = K_atoms // max(R_atoms, 1)
r = 0
for k_atom in range(K_atoms):
    main_mma.set(ACCUMULATE, k_atom != 0)       # first main atom initializes acc
    main_mma_issue(acc_tmem, A_stage, B_stage, Asc_stage, Bsc_stage)

    if r < R_atoms and k_atom % stride == 0:
        lora_mma.set(ACCUMULATE, True)           # LoRA always accumulates
        lora_mma_issue(acc_tmem, LA_smem, LU_smem, r)
        r += 1
```

Per-shape `(K_atoms, R_atoms, stride)` with `K_BLK = 64`, `LORA_K = 16`:

| M     | K     | N     | R   | K_atoms | R_atoms | stride |
| ----- | ----- | ----- | --- | ------- | ------- | ------ |
| 256   | 3840  | 3072  | 128 | 60      | 8       | 7      |
| 4352  | 3840  | 3072  | 128 | 60      | 8       | 7      |
| 4352  | 3840  | 15360 | 128 | 60      | 8       | 7      |
| 4352  | 15360 | 3840  | 128 | 240     | 8       | 30     |
| 4352  | 10240 | 3072  | 32  | 160     | 2       | 80     |
| 4352  | 10240 | 3072  | 256 | 160     | 16      | 10     |

**Why sprinkle, not head or tail:**
- *Head*: pipe sees only LoRA atoms at start → queue depth can't fill
  (sat < 1); main loop then runs alone fine. Net: the LoRA region still
  under-utilizes the pipe.
- *Tail*: symmetric, same loss.
- *Sprinkle*: the issue queue holds a mix of main + LoRA atoms
  throughout. In-flight depth never drops below 4 for any contiguous
  window. LoRA atoms don't extend wall time, they replace would-be
  main issue slots, and the main K-loop has enough atoms left to
  finish behind them.

**Two invariants** the implementation must preserve:

1. **Shared-tmem TV-layout match.** Main NVFP4 acc fragment layout
   == LoRA fp16/bf16 acc fragment layout over the chosen
   `mma_tiler_mn`. Verified at trace time by
   `cute_kernels/gemm_w4a4/verify_tmem_layout.py`
   — both `1SM 128x256` and `2SM 256x256` must pass. If a future
   config breaks the match, this whole design collapses to separate
   tmem regions + epilogue sum (extra tmem traffic, uglier code).

2. **LoRA tail clears before main tail.**
   `stride * R_atoms < K_atoms` must hold for every config. Table above
   confirms (`stride * R_atoms` = 56, 56, 56, 240, 160, 160 vs
   K_atoms = 60, 60, 60, 240, 160, 160). Equality is fine — LoRA's
   last issue can coincide with main's last issue; the combined
   `umma.arrive` barrier fences both. Strict `<` is ideal; strict `>`
   would require an extra drain loop and is banned as a config.

---

## 3. Tmem layout (single shared accumulator)

One fp32 region of shape `(BLOCK_M, BLOCK_N)` per CTA, tiled by the
chosen `mma_tiler_mn`. Both MMAs write in, both set ACCUMULATE after
the first main atom.

```
tmem:
  acc_fp32[BLOCK_M, BLOCK_N]
```

No separate LoRA acc. No epilogue reduction between the two paths.
The "shared-tmem" label is literal: same bytes, same TV-layout.

---

## 4. Smem layout

**Main (multi-stage pipeline, stages = 4 by default)**

```
per stage:
  A_tile[BLOCK_M, K_BLK / 2]   uint8   (NVFP4 packed)
  B_tile[BLOCK_N, K_BLK / 2]   uint8   (NVFP4 packed)
  Asc_tile[K_BLK / 16, BLOCK_M]  fp8_e4m3
  Bsc_tile[K_BLK / 16, BLOCK_N]  fp8_e4m3
```

Size (`BLOCK_M = BLOCK_N = 256`, `K_BLK = 64`, 4 stages):
- A/B: `256 × 32` B each → 8 KB per tile × 2 × 4 stages = 64 KB
- Asc/Bsc: `4 × 256` B each → 1 KB per tile × 2 × 4 stages = 8 KB
- **main total: ~72 KB**

**LoRA (single-buffer prolog — loaded once per CTA)**

```
LA_tile[BLOCK_M, R]   operand_dtype    (cast from fp32 at load time)
LU_tile[BLOCK_N, R]   operand_dtype
```

Size (`BLOCK_M = BLOCK_N = 256`, `R = 128`, fp16):
- LA: `256 × 128 × 2` = 64 KB
- LU: `256 × 128 × 2` = 64 KB
- **LoRA total: 128 KB** (R=128 worst case; R=32 → 32 KB; R=256 → 256 KB)

Blackwell SMEM per SM: 228 KB. 2-CTA mode splits across two SMs so
each CTA sees its half — tight for `R = 256`, still fits. If it ever
blows up, fall back to LoRA double-stage TMA inside the K-loop.

**Open decision — LA dtype on the wire.** `lora_act_in` is emitted
fp32 by the preceding `quantize_w4a4_act_fuse_lora` op (per
`baseline/.../quantize_w4a4_act_fuse_lora/ref.py`, for precision).
LoRA MMA wants fp16 or bf16. Two options:
(a) load fp32 via TMA, cast in registers before smem write (spends
    BW on the fp32 load, halves smem usage).
(b) cast to operand dtype on the previous op's store side (changes
    the inter-op contract, halves wire BW).

Default: **(a)**. Pre-op emit stays at full fp32 precision; cast cost
is a single `cvt.rn.f16.f32` per element, inconsequential next to the
MMA work. Revisit if profiling shows the load warp choked.

---

## 5. Warp roles

fmha-style warp specialization, minus softmax and correction:

| group            | count   | responsibility                                                          |
| ---------------- | ------- | ----------------------------------------------------------------------- |
| **load**         | 1 warp  | TMA prolog for LA/LU; main K-loop TMA for A/B/Asc/Bsc, stage rotation  |
| **mma**          | 1 warp  | issues main atoms + sprinkled LoRA atoms per §2; sets ACCUMULATE flags |
| **epilogue**     | 4 warps | tmem load → × wcscales → + bias → cast → optional NVFP4 quant → store  |

Total 6 warps = 192 threads per CTA. 2-CTA mode: 384 threads, two SMs
sharing one main MMA via `CtaGroup.TWO`.

**Barriers (pipeline_stages = 4)**:
- `main_load → main_mma`: 4-stage mbarrier (TMA arrive → MMA issue)
- `lora_load → lora_mma`: 1-stage mbarrier (prolog TMA arrive → first LoRA issue)
- `mma → epilogue`: `umma.arrive` after the last issue in §2's stream;
  epilogue waits on that single fence.

No softmax, no correction, no mS scratchpad. The load warp sequence
is linear; the mma warp sequence is the loop in §2 verbatim.

---

## 6. Tile shape

```
BLOCK_M × BLOCK_N = 256 × 256    2SM (CtaGroup.TWO)
BLOCK_M × BLOCK_N = 128 × 256    1SM (CtaGroup.ONE)   # small-M fallback
K_BLK              = 64
```

Start with 2SM 256×256 for all six shapes. Small-M (M=256) leaves at
most one 2SM tile active, which won't fill the 148 SMs. Measure first
— if M=256 stalls below target, fall back to 1SM 128×256 for
`M < 4 * BLOCK_M_2sm` (heuristic; tune empirically).

---

## 7. Epilogue (v2)

```
y_fp32 = tmem_load(acc_tmem)                     # [BLOCK_M, BLOCK_N]
y_fp32 = y_fp32 * wcscales_tile[BLOCK_N]         # per-col broadcast
y_fp32 = y_fp32 + bias_tile[BLOCK_N]             # per-col broadcast
y_out  = cast<out_dtype>(y_fp32)
tma_store(y_out)
```

No alpha — the scaled-MMA already absorbed the per-16-K block scales
through NVFP4 `make_blockscaled_trivial_tiled_mma`'s SF path.

---

## 8. Optional next-layer quant (v3) — **dropped 2026-04-24**

Originally scoped to mirror nunchaku's `EpilogueQuantize`:

```
y_scaled = y_fp32 / smooth_next_tile[BLOCK_N]    # after wcscales+bias
qout, oscales = nvfp4_quant_rows(y_scaled)
tma_store(qout); tma_store(oscales)
```

**Why dropped**: nunchaku exercises this path only via `fused_gelu_mlp`
(`nunchaku/ops/fused.py:15`) — fc1's `gemm_w4a4` is fed `fc2.smooth_factor`
(and `fc2.proj_down` for LoRA-down fold-in) so fc1's epilogue produces
fc2's quantized input directly. Two adjacent linears must be aware of
each other's parameters. That crosses the linear-op boundary and needs
vLLM-side pipeline cooperation — the same reason CLAUDE.md excludes
`fuse_glu`. Unreachable under our drop-in API. Re-open only if the
vLLM integration strategy for consecutive w4a4 linears changes.

---

## 9. Skeleton source strategy

Neither CUTLASS example is a drop-in fork:

- `blackwell/dense_blockscaled_gemm_persistent.py` — single MMA,
  correct scaled-MMA atom + SF TMA, wrong pipeline topology for us.
- `blackwell/fmha.py` — dual back-to-back MMA, persistent CTA,
  warp-specialized load/MMA/epilogue — but dragging softmax +
  correction warps that we don't need, and no scaled-MMA path.

Plan: **hand-write the skeleton**, cherry-picking:

| from                                     | take                                                   |
| ---------------------------------------- | ------------------------------------------------------ |
| `dense_blockscaled_gemm_persistent.py`   | `make_blockscaled_trivial_tiled_mma` wiring; SF TMA path; NVFP4 A/B + FP8 scale smem layout |
| `fmha.py`                                | persistent CTA scheduler; warp-specialization split; multi-stage TMA → mma barrier idiom |
| new                                      | the §2 interleave loop; LoRA prolog TMA; single shared-tmem acc; the §7 epilogue         |

No copy-paste of a full example — the structural differences are too
large, and dead scaffolding is worse than clean rewriting.

---

## 10. Staged implementation

Matches task #33 – #35 (task #36 / v3 dropped — see §8):

| version | scope                                                                                                |
| ------- | ---------------------------------------------------------------------------------------------------- |
| v0      | main NVFP4 only, no LoRA, no wcscales, no bias. `y = scaled_mma(act, wgt)` + direct TMA store.       |
| v1      | + LoRA β-interleaved per §2. Shared tmem acc. epilogue still bare cast.                              |
| v2      | + per-col `* wcscales + bias` epilogue. **Terminal version.**                                        |
| ~~v3~~  | ~~+ optional next-layer NVFP4 quantize (§8).~~ **Dropped — requires vLLM pipeline fusion.**          |

Each version is wired through `tmp/smoke_gemm.py` (add the kernel path
to the three-way diff) and `tmp/bench_gemm.py` (adds a kernel column
to the fp16-torch baseline comparison) before the next one starts.
