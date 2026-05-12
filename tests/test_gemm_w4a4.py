"""Phase 3b end-to-end test — torch.ops.svdquant.gemm_w4a4 INT4 + LoRA-up.

Inputs: signed-INT4-packed act + wgt + per-64-K-block fp16 scales +
fp32 LoRA-down output + fp16 LoRA-up weight. Output: fp16 [M, N].

Reference: `gemm_w4a4_ref_int4`. Current state: LoRA tensors are zeros,
so only the main INT4 path is exercised (kernel doesn't read them
yet). Step 3b-3 wires up the LoRA cube pass; step 3b-4 then un-zeros
the test inputs to exercise both paths.

Tolerance: rtol=5e-2 atol=5e-2 — INT4 quant noise + dequant rounding.
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


# Phase 3b tile constants — must match svdquant_w4a4_op.cpp + kernel_device.cpp.
PHASE3B_M = 64
PHASE3B_K = 128
PHASE3B_N = 128
PHASE3B_R = 32     # LoRA rank — must match kPhase3bR in svdquant_w4a4_op.cpp


class TestGemmW4A4Phase3bInt4Lora(unittest.TestCase):

    def test_phase3b_int4_lora_path(self):
        _step("test_phase3b_int4_lora_path: enter")
        if not torch.npu.is_available():
            self.skipTest("Ascend NPU not available")

        _step("  building INT4 inputs (CPU)")
        act, wgt, ascales, wscales = make_int4_inputs(
            PHASE3B_M, PHASE3B_K, PHASE3B_N
        )

        _step("  building reference (CPU)")
        # Seeded random LoRA tensors — small amplitude so the LoRA-up
        # contribution sits in the same magnitude band as the main GEMM
        # output (~O(K * amax^2)). The kernel applies it pre-fp16-cast,
        # so even a tiny LoRA reaches the output bits.
        g = torch.Generator().manual_seed(0xB0BA)
        lora_act_in = (torch.rand(PHASE3B_M, PHASE3B_R, generator=g) * 2 - 1) * 0.1
        lora_up = ((torch.rand(PHASE3B_N, PHASE3B_R, generator=g) * 2 - 1) * 0.1).to(torch.float16)
        ref, _, _ = gemm_w4a4_ref_int4(
            act, wgt, ascales, wscales, lora_act_in, lora_up,
        )
        self.assertEqual(ref.shape, (PHASE3B_M, PHASE3B_N))
        self.assertEqual(ref.dtype, torch.float16)
        _step(f"  ref shape {tuple(ref.shape)} max_abs={ref.abs().max().item():.3f}")

        _step("  moving inputs to NPU")
        act_npu = act.npu()
        wgt_npu = wgt.npu()
        ascales_npu = ascales.npu()
        wscales_npu = wscales.npu()
        lora_act_in_npu = lora_act_in.npu()
        lora_up_npu = lora_up.npu()
        torch.npu.synchronize()
        _step("  inputs on NPU OK")

        _step("  calling torch.ops.svdquant.gemm_w4a4")
        out = torch.ops.svdquant.gemm_w4a4(
            act_npu, wgt_npu, ascales_npu, wscales_npu,
            lora_act_in_npu, lora_up_npu,
        )
        _step("  op returned, syncing")
        torch.npu.synchronize()
        _step(f"  sync OK, out shape {tuple(out.shape)} dtype {out.dtype}")
        self.assertEqual(out.shape, (PHASE3B_M, PHASE3B_N))
        self.assertEqual(out.dtype, torch.float16)

        out_cpu = out.cpu()
        diff = (out_cpu.float() - ref.float()).abs()
        _step(
            f"  diff vs ref: max_abs={diff.max().item():.4f} "
            f"mean_abs={diff.mean().item():.4f} "
            f"any_nan={out_cpu.isnan().any().item()}"
        )
        torch.testing.assert_close(
            out_cpu, ref, rtol=5e-2, atol=5e-2,
            msg="phase 3b int4+lora output diverged from baseline ref",
        )
        _step("test_phase3b_int4_lora_path: pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
