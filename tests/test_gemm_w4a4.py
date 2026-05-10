"""Phase 3a end-to-end test — torch.ops.svdquant.gemm_w4a4 INT4 main path.

Inputs: signed-INT4-packed act + wgt + per-64-K-block fp16 scales.
Output: fp16 [M, N].

Reference: `gemm_w4a4_ref_int4` with `lora_act_in = zeros, lora_up = zeros`
so the LoRA residual contributes nothing — Phase 3a only implements
the main GEMM path; LoRA is 3b. We don't expose a "no-lora" code path
in the op schema (Phase 3b will add the lora tensors as required args
and update both this test and the binding); zeros here is debug-only.

Tolerance: rtol=5e-2 atol=5e-2 — INT4 quant noise + dequant rounding
plus our K=2048 reduction stack up. Numeric fidelity at this looseness
is a sanity check, not a target metric. The real perf-vs-nunchaku eval
is on a different track (CUDA-side, B200).
"""

import sys
import unittest
from pathlib import Path

import torch
import torch_npu  # noqa: F401  — registers PrivateUse1 backend

# csrc/python is on sys.path so we can `import op_extension`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "csrc" / "python"))
sys.path.insert(0, str(_REPO_ROOT))

import op_extension  # noqa: F401, E402  — loads libop_extension.so
from baseline.kernels.gemm_w4a4.ref_int4 import (  # noqa: E402
    gemm_w4a4_ref_int4, make_int4_inputs,
)


# Phase 3a tile constants — must match svdquant_w4a4_op.cpp + kernel_device.cpp.
PHASE3A_M = 128
PHASE3A_K = 2048
PHASE3A_N = 256
PHASE3A_R = 1     # LoRA rank for the zero-stub residual (smallest valid).


class TestGemmW4A4Phase3aInt4(unittest.TestCase):

    def test_phase3a_int4_main_path(self):
        if not torch.npu.is_available():
            self.skipTest("Ascend NPU not available")

        act, wgt, ascales, wscales = make_int4_inputs(
            PHASE3A_M, PHASE3A_K, PHASE3A_N
        )

        # Reference math: full INT4 path with LoRA residual = 0.
        lora_act_in = torch.zeros(PHASE3A_M, PHASE3A_R, dtype=torch.float32)
        lora_up = torch.zeros(PHASE3A_N, PHASE3A_R, dtype=torch.float16)
        ref, _, _ = gemm_w4a4_ref_int4(
            act, wgt, ascales, wscales, lora_act_in, lora_up,
        )
        self.assertEqual(ref.shape, (PHASE3A_M, PHASE3A_N))
        self.assertEqual(ref.dtype, torch.float16)

        # NPU path.
        out = torch.ops.svdquant.gemm_w4a4(
            act.npu(), wgt.npu(), ascales.npu(), wscales.npu()
        )
        torch.npu.synchronize()
        self.assertEqual(out.shape, (PHASE3A_M, PHASE3A_N))
        self.assertEqual(out.dtype, torch.float16)

        torch.testing.assert_close(
            out.cpu(), ref, rtol=5e-2, atol=5e-2,
            msg="phase 3a int4 main-path output diverged from baseline ref",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
