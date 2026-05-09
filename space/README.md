# `space/` — GitCode Space deployment payload

This directory and the sibling `app.py` are the deployment payload for
the GitCode Space at <https://ai.gitcode.com/qq_42927189/svdquant-kernels>.
They live on the `gitcode-space` branch only — `main` keeps the
`*.o` / `*.bin` exclusions in `.gitignore` and never carries built
artifacts.

## Why a separate branch

The Space's NPU container is aarch64; building the AscendC kernel
there means running the full ascendc precompile + AIC + AIV + merge
pipeline through `ccec` natively, which is slow. Instead we
cross-compile locally on the x86_64 dev box and ship the resulting
object files as pre-built data.

## Layout

```
app.py                         Gradio frontend (lives at repo root —
                               GitCode Space convention)
space/
  link_smoke.sh                runs ON Space: links the .o files into
                               an aarch64 ELF using the container's
                               native CANN aarch64 libs
  smoke_main.cpp               source for the smoke driver (the only
                               new source on this branch; everything
                               else compiles from csrc/ or
                               build_ascend/auto_gen/)
  objects/
    host_stub.cpp.o            aarch64; CANN auto-gen kernel registry
                               stub WITH the device blob already
                               injected via objcopy --update-section
    kernel.cpp.o               aarch64; svdquant::ascend::gemm_w4a4
                               host launcher (aclrtMalloc + H2D copy
                               + aclrtlaunch_*)
    smoke_main.cpp.o           aarch64; aclInit + launch + sync
    blob.bin                   raw NPU code blob (1244 B), kept here
                               for re-injection if host_stub is
                               regenerated
```

## Local rebuild recipe (when the kernel changes)

```bash
# 1. Ensure x86_64 build is up to date — that's where the NPU device
#    blob comes from.
./scripts/build.sh CUDA=OFF ASCEND=ON

# 2. Re-extract the device blob and cross-build the 3 .o files.
#    See the section "Local cross-build" below; this is currently a
#    one-liner sequence rather than a wrapper script (we may bake it
#    into scripts/build.sh later if it churns).
```

## Local cross-build (what produced the .o files in this commit)

```bash
CANN=/usr/local/Ascend/cann-8.5.0/x86_64-linux

# Extract the NPU device blob from the x86 host_stub.o
objcopy -O binary \
    --only-section=.ascend.kernel.ascend910b1.svdquant_gemm_w4a4_device \
    /tmp/sq_extract/host_stub.cpp.o \
    space/objects/blob.bin

# Cross-compile host_stub.cpp (aarch64), then inject the blob
aarch64-linux-gnu-g++ -O2 -std=c++17 -I"${CANN}/include" \
    -c build_ascend/auto_gen/svdquant_gemm_w4a4_device/host_stub.cpp \
    -o space/objects/host_stub.cpp.o
aarch64-linux-gnu-objcopy \
    --update-section .ascend.kernel.ascend910b1.svdquant_gemm_w4a4_device=space/objects/blob.bin \
    space/objects/host_stub.cpp.o

# Cross-compile our host launcher
aarch64-linux-gnu-g++ -O2 -std=c++17 \
    -I"${CANN}/include" \
    -I csrc/kernels/gemm_w4a4/include \
    -I csrc/common/include \
    -I build_ascend/include/svdquant_gemm_w4a4_device \
    -c csrc/kernels/gemm_w4a4/ascend/kernel.cpp \
    -o space/objects/kernel.cpp.o

# Cross-compile the smoke driver
aarch64-linux-gnu-g++ -O2 -std=c++17 \
    -I"${CANN}/include" \
    -I csrc/kernels/gemm_w4a4/include \
    -I csrc/common/include \
    -c space/smoke_main.cpp \
    -o space/objects/smoke_main.cpp.o
```

`libascendc_runtime.a` (aarch64) is **not** shipped — we don't have it
on the local x86_64 dev box. `link_smoke.sh` resolves it from the
Space container's native CANN install at link time.
