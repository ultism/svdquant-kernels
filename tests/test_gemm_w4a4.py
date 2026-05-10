"""Phase 2f end-to-end test — torch.ops.svdquant.gemm_w4a4 on NPU.

Exercises the same numerics path Phase 2e validated through the smoke
binary, only now driven via the proper torch op extension:

  1. Build inputs (numpy fp16, deterministic seed) — same as Phase 2e
     `baseline.kernels.gemm_w4a4.ref_mock`.
  2. Move to NPU via .npu().
  3. Call torch.ops.svdquant.gemm_w4a4(act_npu, wgt_npu).
  4. Slice the vec_out tail off the over-allocated output, average
     across the kNumTiles repeat (every tile in mock mode produces
     the same output).
  5. Compare against numpy mock_gemm reference with torch.testing.assert_close.

Phase 3 collapses the cube-ring scratch out of `out` and adds
ascales/wscales/lora_*/bias to the op signature; the test layout
becomes simpler (no slicing, no tile averaging) at that point.

Uses stdlib `unittest` rather than `pytest` because the GitCode Space
image (ubuntu22-cann8.5-py311-torch2.8-gradio6.9) ships torch but not
pytest, and `pip install` may be blocked depending on container policy.
unittest is in stdlib so always available.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401  — registers PrivateUse1 backend

# csrc/python is on sys.path so we can `import op_extension`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "csrc" / "python"))
sys.path.insert(0, str(_REPO_ROOT))

import op_extension  # noqa: F401, E402  — loads libop_extension.so
from baseline.kernels.gemm_w4a4.ref_mock import make_inputs, mock_gemm  # noqa: E402


# Mock kernel constants — must match svdquant_w4a4_op.cpp + kernel_device.cpp.
PHASE2F_M         = 64
PHASE2F_K         = 128
PHASE2F_N         = 128
PHASE2F_RING      = 6
PHASE2F_NUM_TILES = 8
PHASE2F_SCALE     = 0.5


def _vec_out_tile(out_flat: torch.Tensor, tile_idx: int) -> torch.Tensor:
    """Slice the `tile_idx`-th vec_out tile out of the flat buffer.

    Layout (matches kernel_device.cpp):
      out[0 .. RING * M * N)            cube ring scratch (don't read)
      out[RING * M * N .. + NUM_TILES * M * N)
                                        vec_out, one M×N tile per slot.
    """
    base = (PHASE2F_RING + tile_idx) * PHASE2F_M * PHASE2F_N
    return out_flat.flatten()[base:base + PHASE2F_M * PHASE2F_N].view(
        PHASE2F_M, PHASE2F_N
    )


class TestGemmW4A4Phase2fMock(unittest.TestCase):

    def test_phase2f_mock(self):
        if not torch.npu.is_available():
            self.skipTest("Ascend NPU not available")

        act_np, wgt_np = make_inputs(PHASE2F_M, PHASE2F_K, PHASE2F_N)
        ref_np = mock_gemm(act_np, wgt_np, scale=PHASE2F_SCALE).astype(np.float32)
        ref = torch.from_numpy(ref_np)

        act = torch.from_numpy(act_np).npu()
        wgt = torch.from_numpy(wgt_np).npu()

        out = torch.ops.svdquant.gemm_w4a4(act, wgt)
        torch.npu.synchronize()

        # Every vec_out tile in mock mode produces the same value (kernel
        # loops the same input through num_tiles slots). Pull tile 0 and
        # also confirm tiles 1..N-1 match it bit-for-bit.
        out_cpu = out.cpu()
        tile_0 = _vec_out_tile(out_cpu, 0)
        for t in range(1, PHASE2F_NUM_TILES):
            tile_t = _vec_out_tile(out_cpu, t)
            torch.testing.assert_close(
                tile_t, tile_0, rtol=0.0, atol=0.0,
                msg=f"mock kernel produced different output at tile {t} vs 0",
            )

        torch.testing.assert_close(tile_0, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
