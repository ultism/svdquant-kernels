#!/usr/bin/env bash
# Final aarch64 link step for the torch op extension — runs ON the
# GitCode Space 910B container, NOT during local cross-build.
#
# Inputs (locations in repo / container):
#   space/objects/host_stub.cpp.o     aarch64; auto-gen kernel registry stub
#                                     with device blob already injected
#                                     via objcopy --update-section
#                                     (cross-built locally, ships in repo)
#   space/objects/kernel.cpp.o        aarch64; svdquant::ascend::gemm_w4a4
#                                     host launcher  (cross-built locally,
#                                     ships in repo)
#   csrc/python/host/svdquant_w4a4_op.cpp   torch op wrapper (compiled here
#                                            against Space's torch+torch_npu
#                                            headers — they're aarch64-native
#                                            in the image)
#
# Output: space/objects/libop_extension.so (aarch64 shared library, loaded
#                                           by torch.ops.load_library).
#
# Why on the container, not pre-shipped: torch / torch_npu headers and
# .so files are aarch64 and tied to the Space image's exact Python +
# torch versions (currently py3.11 + torch 2.8 + torch_npu 2.8). Pulling
# those onto the dev box for a full cross-build would mean ~5 GB of
# sysroot setup that drifts every time the image refreshes. Linking on
# the container (~5 s for one .cpp) keeps the dev-box dependency to
# just aarch64-linux-gnu-g++ and the existing CANN + AscendC toolchain.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
ASCEND="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

if [[ ! -d "${ASCEND}" ]]; then
    echo "[link_op_extension] ASCEND_HOME_PATH (${ASCEND}) not found; please source set_env.sh first" >&2
    exit 10
fi

LIBDIR="${ASCEND}/lib64"
RUNTIME_A="${LIBDIR}/libascendc_runtime.a"
ASCENDCL_SO="${LIBDIR}/libascendcl.so"

for f in "${RUNTIME_A}" "${ASCENDCL_SO}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[link_op_extension] missing required CANN file: ${f}" >&2
        exit 11
    fi
done

# Discover torch + torch_npu install paths from the running interpreter.
# Querying via Python keeps the script independent of distro layout;
# Space's image carries both packages on the default python3 path.
PYTHON_BIN="${PYTHON_BIN:-python3}"

TORCH_PATH=$("${PYTHON_BIN}" -c "import os, torch; print(os.path.dirname(torch.__file__))")
TORCH_NPU_PATH=$("${PYTHON_BIN}" -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))")
CXX11_ABI=$("${PYTHON_BIN}" -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)")

if [[ -z "${TORCH_PATH}" || -z "${TORCH_NPU_PATH}" ]]; then
    echo "[link_op_extension] could not resolve torch/torch_npu install paths" >&2
    exit 12
fi

OUT="${HERE}/objects/libop_extension.so"
SRC="${REPO}/csrc/python/host/svdquant_w4a4_op.cpp"

set -x
g++ -O2 -std=c++17 -fPIC -shared \
    -D_GLIBCXX_USE_CXX11_ABI="${CXX11_ABI}" \
    -I"${HERE}/../csrc/python/host" \
    -I"${REPO}/csrc/kernels/gemm_w4a4/include" \
    -I"${REPO}/csrc/common/include" \
    -I"${ASCEND}/include" \
    -I"${TORCH_PATH}/include" \
    -I"${TORCH_PATH}/include/torch/csrc/api/include" \
    -I"${TORCH_NPU_PATH}/include" \
    "${SRC}" \
    "${HERE}/objects/host_stub.cpp.o" \
    "${HERE}/objects/kernel.cpp.o" \
    -L"${LIBDIR}" \
    -L"${TORCH_PATH}/lib" \
    -L"${TORCH_NPU_PATH}/lib" \
    -Wl,--copy-dt-needed-entries \
    -Wl,--whole-archive "${RUNTIME_A}" -Wl,--no-whole-archive \
    -ltorch -ltorch_cpu -lc10 -ltorch_python \
    -ltorch_npu \
    -lascendcl -lruntime -lerror_manager -lprofapi -lmmpa \
    -lascend_dump -lascendalog -lge_common_base -lc_sec \
    -lregister -lplatform -ltiling_api \
    -ldl -lpthread \
    -Wl,--allow-shlib-undefined \
    -Wl,-rpath,"${LIBDIR}" \
    -Wl,-rpath,"${TORCH_PATH}/lib" \
    -Wl,-rpath,"${TORCH_NPU_PATH}/lib" \
    -o "${OUT}"
set +x

echo "[link_op_extension] OK -> ${OUT}"
