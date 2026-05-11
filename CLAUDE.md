# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Execution environment

This machine is primarily an **edit / trace-check host**. The full
CUDA 13 toolkit (`/usr/local/cuda-13.0`, `nvcc` 13.0.88) and CANN 8.5
toolkit (`ccec`, aarch64 `hcc` cross-toolchain) are installed;
`cmake` + `./scripts/build.sh` covers the AscendC cross-compile path.
The canonical Python venv has `cutlass-dsl` installed for CuTe DSL
trace-level checks. There is also a **local consumer Blackwell GPU**
(RTX 5060 Ti, SM_120) — enough to JIT-run Triton kernels for
correctness, **not** enough to execute CuTe DSL NVFP4 kernels (SM_100
/ SM_103 only). There is no local Ascend NPU.

Remote execution lives on two serverless channels:

- **CUDA / Blackwell B200 → Modal (iterate) → Verda (deep trace).**
  No local compile step for CUDA; Python sources (`cute_kernels/`,
  `triton_kernels/`, `baseline/`, `tmp/`) are mounted into the Modal
  image and JIT-compiled on-device. Entry point:
  `scripts/modal_app.py` via `modal run`.

  **Split of responsibility between the two CUDA hosts:**

  - **Modal** is the fast-iteration loop — correctness smokes, wall
    time, cheap benches. `torch.profiler(activities=[CUDA])` works
    (CUPTI Activity buffer — per-kernel device wall time with launch
    overhead stripped), and `nsys profile --trace=cuda,nvtx` gives the
    kernel timeline. **Counter-level profiling does not work** on
    Modal: host has `NVreg_RestrictProfilingToAdminUsers=1`, so `ncu`,
    `nsys --gpu-metrics-device`, and any perf-counter query fail with
    `LibraryNotLoaded`. No container env / capability override
    unblocks this.
  - **Verda** is the deep-trace host — advertises native `ncu` support
    (perf counters unrestricted on their B200 image). Use when wall
    time + activity trace aren't enough to explain a regression:
    tensor-pipe busy %, SM active %, memory-workload analysis,
    roofline, SpeedOfLight breakdown. Only go there after Modal has
    taken the kernel as far as it can.

  Canonical workflow: write → smoke + perf-iterate on Modal → if a
  shape or version stalls below expected speedup and the activity
  timeline doesn't explain why, pull that one kernel onto Verda for
  roofline + ncu metrics → fix → land back on Modal.
- **Ascend NPU / Atlas A3 → OpenI (启智社区).** Cross-build aarch64
  artifacts locally, tar them, `scripts/ship.sh` uploads to the OpenI
  dataset; the serverless pod auto-extracts at boot.

### Local Python environment

**`/root/vllm-omni/.venv` is the canonical Python env for this repo.**
Python 3.12, torch 2.11 + cu130 (matches the host CUDA 13 toolkit),
triton 3.6. Use it for all Triton correctness runs, `baseline/`
reference code, and anything else under `tmp/` that needs to
import torch or triton:

```
/root/vllm-omni/.venv/bin/python tmp/smoke_lora_down.py
```

Do not use `/root/miniconda3` (base ships torch-cpu) or
`/root/miniconda3/envs/vllm` (torch is cu128, triton 3.5 — lags cu130
and will eventually drift). The repo itself has no Python packaging
yet; nothing is installed *into* the venv for this project — the venv
is just a known-good toolset we borrow.

### Consequences for how you work here

- **CUDA kernels are Python CuTe DSL, JITed at first call.** No local
  nvcc step. For trace-level sanity on a wrong-arch dev box (NVFP4 /
  tcgen05 are rejected by the SM_120 probe), set
  `CUTE_DSL_ARCH=sm_100a` in the environment; the MLIR still lowers.
  Real execution only makes sense on B200.
- **Local cmake build now only covers AscendC pods** (plus tests/bench
  when enabled). On a CUDA-only box it produces an effectively empty
  `build/`. That's fine — the CUDA side ships as Python source.
- **Triton and CuTe DSL pods can be trace-/import-checked locally.**
  The SM_120 card is fine as a Triton correctness target; for CuTe DSL
  perf tuning has to happen on B200.
- `nvcc` is still on PATH for the Ascend cross-compile story and any
  future C++ CUDA pod. `scripts/build.sh` scrubs the environment with
  `env -i` before configuring CMake to keep conda's shadowing `nvcc`
  12.8 out of the way.
- **Scratch artifacts go in `tmp/`** (gitignored): trace dumps pulled
  back from remote, generated sources, one-off scripts, staging
  tarballs. Don't pollute the repo root.
- **Command / bench / smoke output goes in `log/`** (gitignored):
  `modal run ... 2>&1 | tee log/<task>.log`, `nvidia-smi` dumps, nsys
  reports, anything textual you want to eyeball. Keep `tmp/` for
  *source-like* scratch (scripts, traces) and `log/` for *output*.
  Cross-reference by filename (e.g., `tmp/bench_gemm_v0.py` →
  `log/bench_gemm_v0.log`).

## What this repo is (and isn't)

A **kernel development workbench** for SVDQuant (W4A4 linear with low-rank
correction), **targeting vLLM's diffusion path**. Cross-architecture:
NVIDIA GPUs (CUDA, SM_100/SM_103 data-center Blackwell only — nunchaku
covers everything else, see `tmp/nunchaku/setup.py:41-64`) and Huawei
Ascend NPUs (AscendC) from one source tree.

Because vLLM is the consumer, kernels must be **drop-in at the linear
boundary**: `input → svdquant_op → output`. Fusions that would require
modifying vLLM's own pipeline (e.g., absorbing a preceding SwiGLU / GeGLU
gate into a quantize op, as nunchaku does with `fuse_glu`) are
**explicitly out of scope** — the perf win isn't worth the upstream
patch burden and the maintenance coupling. Keep the API narrow.

**4-bit format splits by backend**:
- **CUDA (SM_100/103)**: NVFP4 only (E2M1 values + 16-element FP8-E4M3
  block scales). Blackwell tcgen05 dropped INT4 scaled-MMA; CuTe DSL's
  `make_blockscaled_trivial_tiled_mma` only exposes MXF4/NVFP4/MXF8.
- **Ascend NPU**: INT4 (signed INT4 values + per-64-K-block FP16
  scales). Ascend's cube unit has INT4 MMA but no FP4. Math mirrors
  nunchaku's pre-Blackwell INT4 path.

So the compute-bound pods (`cute_kernels/gemm_w4a4/kernel.py` vs
`csrc/kernels/gemm_w4a4/ascend/*.cpp`) are naturally format-specialized —
each backend uses what its tensor unit actually supports, and the
languages differ by necessity. Triton pods (one source, two backends)
branch on `fp4: bool` at the Python host layer, which becomes a
`tl.constexpr` inside the kernel.

Kernels come in three flavors, one per compiler path:

- **CuTe DSL (Python)** — `cute_kernels/<op>/kernel.py`. `@cute.jit`
  kernel + torch-tensor host wrapper, JIT-lowered through MLIR → PTX
  by `cutlass-dsl` at first call. Used for compute-bound CUDA-only ops
  that need `tcgen05` scaled-MMA / TMEM / 2-CTA. The Python CuTe DSL
  is NVIDIA's authoring path on Blackwell — chosen over the CUDA C++
  CuTe headers because the DSL cuts ~10× the template boilerplate and
  ships directly through the same `cutlass-dsl` package used by upstream.
- **AscendC (C++)** — `csrc/kernels/<op>/ascend/`. Host launcher +
  `__aicore__` device code built by `ccec`. Used for compute-bound
  Ascend-only ops that need the cube unit.
- **Triton** — `triton_kernels/<op>/kernel.py`. One `@triton.jit`
  kernel JIT-compiled by upstream Triton on CUDA and by `triton-ascend`
  on NPU. Used for memory-bound ops that ship on both backends without
  writing the op twice.

Decision rule: memory-bound + cross-backend → Triton. Compute-bound
CUDA (needs tcgen05) → CuTe DSL. Compute-bound Ascend (needs the cube
unit) → AscendC. No shared header between CuTe DSL and AscendC because
the languages differ.

It is explicitly **not** a shipping library. Consequences that affect how
you work here:

- **No runtime dispatcher.** Callers pick the backend directly — for
  CUDA that's the Python `cute_kernels.<op>.launch(...)` (or
  `triton_kernels.<op>.launch(...)`); for Ascend that's
  `svdquant::ascend::<op>(...)` in C++. Do not add a router. If you
  think one is needed, you're in the wrong repo — that belongs to
  whoever integrates these kernels into a framework.
- **Torch only in Python host wrappers.** `baseline/`, `cute_kernels/`,
  and `triton_kernels/` take torch tensors at their host boundary.
  Do not pull `torch` into `csrc/`; AscendC-side Python bindings come
  later, after kernels stabilize.
- **C++ pods are OBJECT libs, not a single `.so`.** Linking is an
  integration concern. Python pods (CuTe DSL, Triton) are plain
  modules under their respective trees.

Current state: the Triton pod `quantize_w4a4_act_fuse_lora` is real
and passes smoke. The CuTe DSL `gemm_w4a4` is in progress — the
shared-tmem feasibility check lives at `tmp/verify_tmem_layout.py`.
All AscendC pods are host-side stubs so the build stays green before
real device code lands. The Ascend-side `gemm_w4a4` pod under
`csrc/kernels/gemm_w4a4/` keeps only the host header + stub; the
CUDA path moved out to `cute_kernels/gemm_w4a4/` (Python CuTe DSL,
JITed on-device).

## Build

`scripts/build.sh` drives CMake, which now only covers AscendC pods
(+ tests/bench when enabled). CUDA CuTe DSL and Triton pods don't go
through CMake — they're Python modules that JIT on first call.

```
# defaults: both backends on, tests/bench off
./scripts/build.sh

# CUDA only / Ascend only
CUDA=ON ASCEND=OFF ./scripts/build.sh
CUDA=OFF ASCEND=ON ./scripts/build.sh

# turn on tests/bench (empty today — opt-in adds the subdirs)
TESTS=ON BENCH=ON ./scripts/build.sh

# Debug build
BUILD_TYPE=Debug ./scripts/build.sh
```

The `CUDA=ON` branch is retained as a placeholder for any future C++
CUDA pod; with all current CUDA work living in `cute_kernels/`, a
CUDA-only CMake configure produces an effectively empty `build/`.

Before configuring on an Ascend machine: `source scripts/env_ascend.sh`
(sources CANN's `setenv.bash` / `set_env.sh`). Override the CANN
location with `ASCEND_HOME_PATH=...`.

Direct CMake flags if you need them:
`-DSVDQUANT_ENABLE_CUDA`, `-DSVDQUANT_ENABLE_ASCEND`,
`-DSVDQUANT_BUILD_TESTS`, `-DSVDQUANT_BUILD_BENCHMARKS`. The
`SVDQUANT_CUDA_ARCHS` CMake knob only affects any future C++ CUDA
pod; for CuTe DSL kernels the arch is set at JIT time via
`CUTE_DSL_ARCH` (see "Conventions" below).

No lint command yet. The Triton pod `quantize_w4a4_act_fuse_lora` has
a working smoke (`tmp/smoke_fused.py`); the CuTe DSL `gemm_w4a4` is
mid-implementation.

## Adding a new kernel pod

### CuTe DSL pod (CUDA)

No CMake line. Drop a directory under `cute_kernels/`:

```
cute_kernels/<op>/
  kernel.py             # @cute.jit kernel + torch-tensor host wrapper
  README.md             # op contract + ref anchor (usually a nunchaku or CUTLASS example)
```

`cutlass-dsl` JITs at first call; there's nothing to build ahead of
time. Already installed in the canonical venv and the Modal triton
image. On a non-B200 dev box (SM_120 here), export
`CUTE_DSL_ARCH=sm_100a` before running so NVFP4 / tcgen05 MMA atoms
pass the arch gate at trace time.

### AscendC pod (NPU)

One line in `csrc/kernels/CMakeLists.txt`:

```cmake
svdquant_add_kernel_pod(<op>)
```

Pod layout:

```
csrc/kernels/<op>/
  include/<op>.h        # struct <Op>Params; svdquant::ascend::<op> decl
  ascend/kernel.cpp     # host launcher
  ascend/kernel_device.cpp   # __aicore__ device code (ccec rule not wired yet)
  README.md
```

The helper still looks for `cuda/kernel.cu` for back-compat; leave that
unset for new pods — CUDA CuTe DSL goes under `cute_kernels/<op>/`.

### Triton pod (CUDA + Ascend)

No CMake line. Drop a directory under `triton_kernels/`:

```
triton_kernels/<op>/
  kernel.py             # @triton.jit kernel + torch-tensor host wrapper
  README.md             # op contract + ref anchor (usually a nunchaku source line)
```

Triton JITs at first call; there's nothing to build ahead of time.
`triton-ascend` has to be installed on the NPU host for the Ascend
path — not required for local edit / CUDA runs.

## Conventions that aren't obvious from the code

- **No shared signature between CUDA and Ascend.** The CUDA side is a
  Python host wrapper in `cute_kernels/<op>/kernel.py` (or
  `triton_kernels/<op>/kernel.py`) that takes torch tensors; the Ascend
  side is a C++ function declared in `csrc/kernels/<op>/include/<op>.h`
  taking a `<Op>Params` + `void* stream`. Keep the *math* and *tensor
  shapes* in sync — the languages differ by necessity.
- **`void* stream` on the Ascend C API.** `aclrtStream` is opaquely
  cast inside `kernel.cpp` so the public header stays free of CANN
  includes.
- **`TensorRef` is Ascend-only.** `data` is an opaque device pointer
  interpreted by the backend; strides are in elements, not bytes.
- **AscendC split:** `ascend/kernel.cpp` is the host launcher, compiled
  by the host C++ compiler. Real device code (`__aicore__`) goes into a
  sibling file (convention: `ascend/kernel_device.cpp`) compiled by
  `ccec`. That `ccec` rule is **not wired yet** — add it the first time
  a pod needs real device code, not before.
- **CUDA arch selection is runtime, not CMake.** `cutlass-dsl` probes
  the device at trace time; override with `CUTE_DSL_ARCH=sm_100a` or
  `sm_103a` for trace checks on a wrong-arch dev box (SM_120 locally)
  or for an AOT-compiled cubin. There is no `SVDQUANT_CUDA_ARCHS`
  equivalent on the CuTe DSL path.
- **`baseline/` is PyTorch-only and must stay importable without any
  kernel having built.** It's ground truth for tests — readability
  beats speed, no `torch.compile`, no fused ops, no backend tricks.
- **Python kernels in `cute_kernels/` and `triton_kernels/` are NOT
  tests or baselines.** They are real kernels that ship; the only
  difference from AscendC is the compiler path (MLIR or Triton → PTX
  instead of `ccec`). Treat them as production surface.

## Record gotchas in the repo, not just memory

When you hit a *silent* misbehavior — kernel compiles, runs,
produces deterministic-looking numbers that are wrong, OR a doc /
type system gate the docs don't cover — file the finding into the
repo before paging the context out. **Do not** rely on memory
notes alone for this; memory decays and is not visible to
collaborators or to a fresh `git clone`.

Where to put each finding:

- **CuTe DSL traps (CUDA path):** append to
  `docs/gotchas/cute_dsl.md`. Examples already there: Python `if`
  inside `@cute.jit` traces both branches; divide-API nesting
  differences; 2-CTA `cluster_layout_vmnk` axes;
  `TiledCopy.partition_D` rest-mode trap; `num_acc_stage` vs
  `tile_n`.
- **Ascend / AscendC / PTO ISA traps:** append to
  `docs/gotchas/ascend.md`. Examples already there: cube↔vec
  handoff is L2-resident; cube min addressable = 1 byte;
  `TLoad` of ColMajor `[N, 1]` from GM loads only the head;
  `TRowExpand` contaminates the vec mask; AIV K-loop reusing a
  partial UB region needs V→MTE2 cross-iter sync.
- **Backend-level scope or design decisions** (what's in /
  out of scope, what tradeoff was settled, why a path was
  abandoned): `docs/architecture.md` § "Scope decisions".
- **Perf-comparison context** (counter routing, reference
  implementation craft level, expected gaps): bottom of
  `docs/gpu.md` or `docs/npu.md`.
- **Per-kernel design state and bring-up history**:
  `docs/kernels/<op>*.md`. The gotcha files are for traps that
  apply broadly across the backend; one-off bring-up cycles
  belong in the per-kernel bring-up doc plus the `log/`
  artifact.

Entry shape: each gotcha is one ATX (`##`) heading with a clear
fact statement up front, then *why* (mechanism / hardware
constraint), then *apply* (fix template, alternatives, places
this could bite again), then optionally a *diagnostic* (how to
recognize the symptom). Faithful to the silent-bug symptom — name
what the wrong output looks like so the next person can pattern-match
quickly.

Memory remains useful for *cross-conversation indices* (what was
validated, what channel to use, current decision status) — but
the **technical content** of a trap belongs in the repo.

## The two ops (why this split)

A W4A4→W4A4 linear chain is two kernels, mirroring nunchaku's
public C++ API (`tmp/nunchaku/src/kernels/zgemm/gemm_w4a4.cu`):

```
[fp x]
  ↓ quantize_w4a4_act_fuse_lora(x, lora_down, smooth)               ← Triton
  ↓   → act_nvfp4 = quant(x / smooth)                                      [M, K/2]
  ↓   → lora_act_out = x @ lora_down                                       [M, R] fp32
  ↓   → ascales                                                            [K/16, M] fp8
[act, lora_act_in, ascales]
  ↓ gemm_w4a4(act, lora_act_in, wgt, lora_up, ascales, wscales,     ← CuTe DSL Python (CUDA)
  ↓           bias, wcscales, [qout, oscales, smooth_next])           ← AscendC (NPU)
  ↓   → y_fp = scaled_mma(act, wgt) · wcscale + bias
  ↓   → y_fp += lora_act_in @ lora_up
  ↓   → [optional] qout = quant(y_fp / smooth_next)                        for next layer
[y (, optional qout/oscales)]
```

Note: nunchaku's kernel has an extra `fuse_glu` flag that folds the
preceding SwiGLU (`gate ⊙ silu(value)`) into the load step. We do
**not** replicate it — see the "What this repo is" section above;
vLLM pipeline intrusion is out of scope.

- `gemm_w4a4` — compute-bound scaled-MMA + LoRA-up residual + bias +
  optional next-layer quantize. CUDA side is a CuTe DSL pod at
  `cute_kernels/gemm_w4a4/kernel.py`; Ascend side (future) is an
  AscendC pod. This is where `tcgen05` / `TMEM` / 2-CTA live.
- `quantize_w4a4_act_fuse_lora` (Triton pod) — memory-bound
  preprocessing: quantize input + do the small `x @ lora_down`
  down-projection. Measured AI 26–120 FLOP/B on ZImage Turbo
  shapes — well below the ~281 FLOP/B B200 tensor-core ridge. Not
  worth the CuTe DSL ceremony, and Ascend needs it too.

**Weight packing is not a pod.** Converting FP weights to INT4/NVFP4 +
block scales is offline and one-shot — run once per model at calibration
time. It lives as a pure-Python utility under `baseline/`. TMA on
SM_100/SM_103 re-tiles a contiguous packed layout cheaply enough at
load time that baking a GEMM-tile-specific layout into the packed file
buys nothing. Don't add `pack_weight_*` back as a CUDA/AscendC pod.

**Activation quantization is not its own pod either.** It only ever
runs fused with the LoRA-down projection (the Triton op above) or as
the epilogue of a previous `gemm_w4a4` (its optional `qout`/`oscales`
outputs). A standalone `quantize_act_*` op has no caller in nunchaku's
dataflow.
