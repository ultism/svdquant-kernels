"""svdquant op_extension — torch op binding loader.

Importing this package locates and dlopen's `lib/libop_extension.so`
into the current Python interpreter. After load, the ops registered
via `TORCH_LIBRARY_FRAGMENT(svdquant, m)` in
`csrc/python/host/svdquant_w4a4_op.cpp` are available as:

    torch.ops.svdquant.gemm_w4a4(act, wgt)

The .so itself is built by `space/link_op_extension.sh` on the Space
container at startup (Path C — see CLAUDE.md / Phase 2f notes). The
build does not use pip / setup.py / cmake; it only invokes g++ to
link a handful of pre-cross-built .o files plus this package's one
host wrapper source. When migrating to a setup.py-based vLLM-Ascend
production build, this loader stub is reused unchanged — only the
.so location convention may shift to `<package>/lib/libop_extension.so`
inside an installed wheel rather than the repo's source tree.
"""

from ._load import _load_opextension_so

_load_opextension_so()
