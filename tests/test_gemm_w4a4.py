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
import traceback
import unittest
from pathlib import Path


def _step(msg: str) -> None:
    """Stream-print so the GitCode log panel sees progress before any
    SIGSEGV silences the process."""
    print(f"[3a-test] {msg}", flush=True)


_step("importing torch")
import torch
_step(f"  torch {torch.__version__}")

_step("importing torch_npu")
import torch_npu  # noqa: F401  — registers PrivateUse1 backend
_step(f"  torch_npu {torch_npu.__version__} npu_available={torch.npu.is_available()}")

# csrc/python is on sys.path so we can `import op_extension`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "csrc" / "python"))
sys.path.insert(0, str(_REPO_ROOT))

_step("importing op_extension (loads libop_extension.so + runs __register_kernels)")
try:
    import op_extension  # noqa: F401, E402  — loads libop_extension.so
    _step("  op_extension loaded OK")
except Exception:
    traceback.print_exc()
    raise

_step("importing baseline.kernels.gemm_w4a4.ref_int4")
from baseline.kernels.gemm_w4a4.ref_int4 import (  # noqa: E402
    gemm_w4a4_ref_int4, make_int4_inputs,
)
_step("  baseline ref_int4 loaded OK")


# Phase 3a tile constants — must match svdquant_w4a4_op.cpp + kernel_device.cpp.
PHASE3A_M = 128
PHASE3A_K = 2048
PHASE3A_N = 256
PHASE3A_R = 1     # LoRA rank for the zero-stub residual (smallest valid).


class TestGemmW4A4Phase3aInt4(unittest.TestCase):

    def test_phase3a_int4_main_path(self):
        _step("test_phase3a_int4_main_path: enter")
        if not torch.npu.is_available():
            self.skipTest("Ascend NPU not available")

        _step("  building INT4 inputs (CPU)")
        act, wgt, ascales, wscales = make_int4_inputs(
            PHASE3A_M, PHASE3A_K, PHASE3A_N
        )

        _step("  building reference (CPU)")
        lora_act_in = torch.zeros(PHASE3A_M, PHASE3A_R, dtype=torch.float32)
        lora_up = torch.zeros(PHASE3A_N, PHASE3A_R, dtype=torch.float16)
        ref, _, _ = gemm_w4a4_ref_int4(
            act, wgt, ascales, wscales, lora_act_in, lora_up,
        )
        self.assertEqual(ref.shape, (PHASE3A_M, PHASE3A_N))
        self.assertEqual(ref.dtype, torch.float16)
        _step(f"  ref shape {tuple(ref.shape)} max_abs={ref.abs().max().item():.3f}")

        _step("  moving inputs to NPU")
        act_npu = act.npu()
        wgt_npu = wgt.npu()
        ascales_npu = ascales.npu()
        wscales_npu = wscales.npu()
        torch.npu.synchronize()
        _step("  inputs on NPU OK")

        _step("  calling torch.ops.svdquant.gemm_w4a4")
        out = torch.ops.svdquant.gemm_w4a4(
            act_npu, wgt_npu, ascales_npu, wscales_npu
        )
        _step("  op returned, syncing")
        torch.npu.synchronize()
        _step(f"  sync OK, out shape {tuple(out.shape)} dtype {out.dtype}")
        self.assertEqual(out.shape, (PHASE3A_M, PHASE3A_N))
        self.assertEqual(out.dtype, torch.float16)

        _step("  comparing to ref")
        torch.testing.assert_close(
            out.cpu(), ref, rtol=5e-2, atol=5e-2,
            msg="phase 3a int4 main-path output diverged from baseline ref",
        )
        _step("test_phase3a_int4_main_path: pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
