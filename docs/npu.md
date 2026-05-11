# Ascend NPU backend

Huawei CANN / AscendC. Other NPU vendors (Cambricon, etc.) are out of
scope for this repo; add them as sibling directories
(`csrc/kernels/<op>/<vendor>/`) if needed later.

## Toolchain

- **CANN toolkit** at `${ASCEND_HOME_PATH}` (default
  `/usr/local/Ascend/ascend-toolkit/latest`).
- `ccec` — AscendC device-side compiler (for `__aicore__` code).
- Host-side ACL runtime (`libascendcl`, `libruntime`).
- **`triton-ascend`** (Python package) — required only for Triton
  kernels under `triton_kernels/`. Install on the Ascend host; not
  required for AscendC-only builds, and not required at all on the
  local cross-compile host.

Before configuring CMake:

```
source scripts/env_ascend.sh
```

This sources the CANN environment (`setenv.bash` / `set_env.sh`).
`cmake/FindCANN.cmake` then locates headers, libs, and `ccec`.

## Per-pod layout (AscendC pods)

```
csrc/kernels/<op>/ascend/
    kernel.cpp                # host launcher (plain C++)
    kernel_device.cpp         # (added later) __aicore__ kernel, compiled by ccec
```

`kernel.cpp` is compiled by the host C++ compiler and today is a stub.
When the first real AscendC kernel lands, we add a second file for the
device code and a custom build rule that feeds it to `ccec` and links
the resulting object into the pod's OBJECT library.

## Per-pod layout (Triton pods)

```
triton_kernels/<op>/
    kernel.py                 # same source as the CUDA path
    README.md
```

Triton pods don't go through CMake or `ccec`. `triton-ascend`
JIT-compiles `kernel.py` to AscendC at first call on the NPU; on
CUDA, upstream Triton JITs the same file to PTX. One source, two
backends.

## Build

```
source scripts/env_ascend.sh
CUDA=OFF ASCEND=ON ./scripts/build.sh
```

or directly:

```
cmake -S . -B build -G Ninja \
    -DSVDQUANT_ENABLE_CUDA=OFF \
    -DSVDQUANT_ENABLE_ASCEND=ON
cmake --build build
```

Triton pods are not part of this build — they aren't compiled ahead
of time, they just need to be importable at runtime on the host.

## Conventions

- AscendC launch signatures take `void* stream`; cast to
  `aclrtStream` inside `kernel.cpp`.
- `TensorRef::data` is a device address (what `aclrtMalloc` returns).
- AscendC kernels should use the tiling helpers rather than
  hand-rolling DMA; cube unit for GEMM-shaped math, vector unit for
  elementwise / reductions.

## When to pick Triton instead of AscendC

If an op is memory-bound (AI below ~90 FLOP/B in practice) AND the
same op needs to run on CUDA too, put it under `triton_kernels/<op>/`
instead of writing AscendC. One `kernel.py` saves having to maintain
parallel CuTe DSL + AscendC implementations. Compute-bound ops and
NPU-only ops still belong here.

## Gotchas (Ascend / PTO ISA traps)

Silent-misbehavior traps on the 910B (a2a3) cube + vec mix-mode
path — cube↔vec handoff is L2-resident not HBM, cube min
addressable is 1 byte (no INT4 tile dtype), `TLoad` of ColMajor
`[N, 1]` from GM only loads the head element, `TRowExpand`
leaves the vec mask register contaminated, AIV K-loop reusing a
partial UB region needs V→MTE2 cross-iter sync. Also the
hardware-level reasoning behind "W4A4 cube uses raw `mad_s4`
inside svdquant, not a PTO wrapper". See
[gotchas/ascend.md](./gotchas/ascend.md). Add new entries there
as you find them.
