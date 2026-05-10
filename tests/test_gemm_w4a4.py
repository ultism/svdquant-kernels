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
PHASE3A_M = 64
PHASE3A_K = 128
PHASE3A_N = 128
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
        out, workspace = torch.ops.svdquant.gemm_w4a4(
            act_npu, wgt_npu, ascales_npu, wscales_npu
        )
        _step("  op returned, syncing")
        torch.npu.synchronize()
        _step(f"  sync OK, out shape {tuple(out.shape)} dtype {out.dtype}")
        _step(f"           workspace shape {tuple(workspace.shape)} dtype {workspace.dtype}")
        self.assertEqual(out.shape, (PHASE3A_M, PHASE3A_N))
        self.assertEqual(out.dtype, torch.float16)

        # ----- partial-dump bisect: did cube write sane int32 partials? -----
        # workspace[slot, m, n] is what cube TSTORE'd post-mad_s4 for the
        # kb=slot-th K-block. With M=64, K=128, N=128 and KS=64, slots 0
        # and 1 should each hold a [64, 128] int32 partial in the range
        # roughly [-3584, 3136] (= 64-elt sum of signed-INT4 products,
        # ±8 × ±8 × 64). Anything wildly outside that band means cube
        # produced garbage; all-zero means cube didn't write at all.
        _step("  inspecting cube partial via workspace[0]")
        ws_cpu = workspace.cpu()
        _step(f"  ws[0, 0, 0:8] = {ws_cpu[0, 0, 0:8].tolist()}")
        _step(f"  ws[0, 1, 0:8] = {ws_cpu[0, 1, 0:8].tolist()}")
        _step(f"  ws[0, 16, 0:8] = {ws_cpu[0, 16, 0:8].tolist()}  (start of 2nd m-fractal)")
        _step(f"  ws[0, 32, 0:8] = {ws_cpu[0, 32, 0:8].tolist()}  (subblock 1's first row)")
        _step(f"  ws[0, 63, 0:8] = {ws_cpu[0, 63, 0:8].tolist()}  (last row)")
        _step(
            f"  ws[0] stats: min={ws_cpu[0].min().item()} "
            f"max={ws_cpu[0].max().item()} "
            f"mean={ws_cpu[0].float().mean().item():.2f} "
            f"nonzero_rows={(ws_cpu[0].abs().sum(dim=1) > 0).sum().item()}"
        )
        _step(
            f"  ws[1] stats: min={ws_cpu[1].min().item()} "
            f"max={ws_cpu[1].max().item()} "
            f"mean={ws_cpu[1].float().mean().item():.2f} "
            f"nonzero_rows={(ws_cpu[1].abs().sum(dim=1) > 0).sum().item()}"
        )
        # Compute expected partial[0] from inputs to compare against.
        # ref_partial[m, n] = sum_{k in K-block 0} act_int4[m, k] * wgt_int4[n, k]
        from baseline.kernels._int4 import _unpack_signed_nibbles  # noqa: E402
        act_int = _unpack_signed_nibbles(act).to(torch.int32)
        wgt_int = _unpack_signed_nibbles(wgt).to(torch.int32)
        ref_partial0 = (act_int[:, :64] @ wgt_int[:, :64].T)
        _step(f"  ref_partial0[0, 0:8] = {ref_partial0[0, 0:8].tolist()}")
        _step(f"  ref_partial0[1, 0:8] = {ref_partial0[1, 0:8].tolist()}")
        _step(
            f"  ref_partial0 stats: min={ref_partial0.min().item()} "
            f"max={ref_partial0.max().item()} "
            f"mean={ref_partial0.float().mean().item():.2f}"
        )
        partial_diff = (ws_cpu[0] - ref_partial0).abs()
        _step(
            f"  ws[0] vs ref_partial0: max_abs={partial_diff.max().item()} "
            f"mean_abs={partial_diff.float().mean().item():.2f}"
        )

        _step("  comparing to ref")
        out_cpu = out.cpu()

        # Diagnostic prints — eyeball the mismatch class before assert_close
        # gives only the first failing entry. The shape of the disagreement
        # (factor-of-2 / sign / NaN / transposed / row-or-col-invariant) is
        # what tells layout bugs apart from scale-offset bugs.
        _step(f"  ref [0,:8] = {ref[0, :8].tolist()}")
        _step(f"  out [0,:8] = {out_cpu[0, :8].tolist()}")
        _step(f"  ref [1,:8] = {ref[1, :8].tolist()}")
        _step(f"  out [1,:8] = {out_cpu[1, :8].tolist()}")
        _step(f"  ref col0 [:8] = {ref[:8, 0].tolist()}")
        _step(f"  out col0 [:8] = {out_cpu[:8, 0].tolist()}")

        diff = (out_cpu.float() - ref.float())
        rel = diff.abs() / (ref.float().abs() + 1e-6)
        _step(
            f"  diff: max_abs={diff.abs().max().item():.4f} "
            f"mean_abs={diff.abs().mean().item():.4f} "
            f"max_rel={rel.max().item():.4f} "
            f"any_nan={out_cpu.isnan().any().item()}"
        )
        # Invariant probes:
        #   * if y[m, n] is essentially independent of n → cube is collapsing
        #     N axis (likely B-side fractal layout error or wscale broadcast
        #     hitting all N).
        #   * if y[m, n] is essentially independent of m → A-side analogous.
        out_n_var = out_cpu.float().std(dim=1).mean().item()
        out_m_var = out_cpu.float().std(dim=0).mean().item()
        _step(
            f"  out variability: along_n_per_row={out_n_var:.4f} "
            f"along_m_per_col={out_m_var:.4f}"
        )

        torch.testing.assert_close(
            out_cpu, ref, rtol=5e-2, atol=5e-2,
            msg="phase 3a int4 main-path output diverged from baseline ref",
        )
        _step("test_phase3a_int4_main_path: pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
