#!/usr/bin/env bash
# Final aarch64 link step — runs ON the GitCode Space 910B container,
# NOT during local cross-build. The 3 .o files in space/objects/ are
# already aarch64 (cross-built locally); we just need libascendc_runtime.a
# (aarch64, from the Space container's native CANN install) for the
# kernel registry symbols, plus libascendcl.so for the public ACL API.
#
# Output: space/svdquant_gemm_w4a4_smoke (aarch64 ELF executable).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ASCEND="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

if [[ ! -d "${ASCEND}" ]]; then
    echo "[link_smoke] ASCEND_HOME_PATH (${ASCEND}) not found; please source set_env.sh first" >&2
    exit 10
fi

LIBDIR="${ASCEND}/lib64"
RUNTIME_A="${LIBDIR}/libascendc_runtime.a"
ASCENDCL_SO="${LIBDIR}/libascendcl.so"

for f in "${RUNTIME_A}" "${ASCENDCL_SO}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[link_smoke] missing required CANN file: ${f}" >&2
        exit 11
    fi
done

set -x
g++ -O2 \
    "${HERE}/objects/host_stub.cpp.o" \
    "${HERE}/objects/kernel.cpp.o" \
    "${HERE}/objects/smoke_main.cpp.o" \
    -Wl,--whole-archive "${RUNTIME_A}" -Wl,--no-whole-archive \
    -L"${LIBDIR}" \
    -lascendcl -lruntime -lerror_manager \
    -ldl -lpthread \
    -Wl,-rpath,"${LIBDIR}" \
    -o "${HERE}/svdquant_gemm_w4a4_smoke"
set +x

echo "[link_smoke] OK -> ${HERE}/svdquant_gemm_w4a4_smoke"
