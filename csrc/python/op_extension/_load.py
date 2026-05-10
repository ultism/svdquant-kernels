"""Locate and load `libop_extension.so` into the Python interpreter.

Two lookup paths, in order:

1. Repo source tree (Phase 2f / Space deployment): the .so is
   produced by `space/link_op_extension.sh` and dropped at
   `<repo_root>/space/objects/libop_extension.so`. This is the path
   the GitCode Space container uses — link script writes there at
   container startup, then `import op_extension` finds it.

2. Installed wheel layout (future setup.py-based path): the .so
   ships inside the package itself at `op_extension/lib/...`. The
   PTO `gemm_basic` reference uses this layout (CMake's
   `CMAKE_LIBRARY_OUTPUT_DIRECTORY` puts it there at install time).

If neither exists, raise ImportError with a hint pointing at the
link script, since that's the most likely missing step in a fresh
checkout.
"""

import os
import pathlib

import torch
import torch_npu  # noqa: F401  — must be imported before load_library so the PrivateUse1 backend is registered before our op extension's TORCH_LIBRARY_IMPL fires.


_LIB_NAME = "libop_extension.so"


def _candidate_paths():
    pkg_dir = pathlib.Path(__file__).resolve().parent
    # (1) Sibling lib/ inside an installed wheel layout.
    yield pkg_dir / "lib" / _LIB_NAME
    # (2) Repo source tree — walk up to find space/objects/.
    for parent in pkg_dir.parents:
        candidate = parent / "space" / "objects" / _LIB_NAME
        if candidate.exists():
            yield candidate
            return


def _load_opextension_so():
    for path in _candidate_paths():
        if path.exists():
            torch.ops.load_library(str(path))
            return path
    raise ImportError(
        f"Could not find {_LIB_NAME}. "
        "Run space/link_op_extension.sh on a host with CANN + torch_npu "
        "to produce it, or install a packaged wheel."
    )
