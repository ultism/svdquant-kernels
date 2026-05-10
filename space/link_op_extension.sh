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

# GitCode Space container runs as uid=1000 with no /etc/passwd entry —
# torch._inductor.codecache calls getpass.getuser() at module init and
# crashes on KeyError. Set USER so getpass short-circuits to env before
# the pwd fallback. (app.py also sets this; defensive when the script
# is invoked standalone.)
export USER="${USER:-svdquant}"
export HOME="${HOME:-/tmp}"

# CANN's ascendalog/ascend_dump libraries write `[LOG_WARNING] can not
# create directory ...` *to stdout* (yes, stdout, not stderr) during
# torch_npu import — the warning gets concatenated with the path our
# `python -c "print(torch.__file__)"` substitution returns, and g++
# then sees garbage-prefixed -I/-L flags. Mute the CANN logger so the
# `$(python -c ...)` paths come back clean.
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3

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
#
# Use a sentinel-prefixed line + `${var##*=}` extraction so any extra
# stdout text (e.g. CANN logger spam that env vars above don't fully
# silence) gets stripped — the marker forces the path to the suffix
# of our captured variable regardless of what the library wrote first.
PYTHON_BIN="${PYTHON_BIN:-python3}"

_TP_RAW=$("${PYTHON_BIN}" -c "import os, torch; print('SVDQ_VAL=' + os.path.dirname(torch.__file__))" 2>/dev/null)
TORCH_PATH="${_TP_RAW##*SVDQ_VAL=}"

_TN_RAW=$("${PYTHON_BIN}" -c "import os, torch_npu; print('SVDQ_VAL=' + os.path.dirname(torch_npu.__file__))" 2>/dev/null)
TORCH_NPU_PATH="${_TN_RAW##*SVDQ_VAL=}"

_AB_RAW=$("${PYTHON_BIN}" -c "import torch; print('SVDQ_VAL=' + ('1' if torch.compiled_with_cxx11_abi() else '0'))" 2>/dev/null)
CXX11_ABI="${_AB_RAW##*SVDQ_VAL=}"

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
