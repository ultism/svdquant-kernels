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

        # ----- 3a-bisect-3: vec-internal partial dump (post-TCVT i32→f32) -----
        # Kernel TSTORE'd `partF32` right after `TCVT(partF32, partI32)`
        # into workspace slots 2/3 (reinterpreted as fp32). Compare against
        # ws[0].float() / ws[1].float() — should match exactly modulo
        # CAST_RINT (no rounding for ints in [-3584, 3136], all exactly
        # representable in fp32). If mismatch → in-place TCVT corrupts.
        ws_f32 = workspace.cpu().view(torch.float32)  # bytewise reinterpret
        debug_part0 = ws_f32[2]    # post-TCVT for K-block 0
        debug_part1 = ws_f32[3]    # post-TCVT for K-block 1
        _step("  vec post-TCVT i32→f32 dump (workspace slot 2/3 viewed as fp32)")
        _step(f"  debug_part0[0, 0:8] = {debug_part0[0, 0:8].tolist()}")
        _step(f"  debug_part0[1, 0:8] = {debug_part0[1, 0:8].tolist()}")
        _step(f"  debug_part0[16, 0:8] = {debug_part0[16, 0:8].tolist()}")
        _step(f"  debug_part0[32, 0:8] = {debug_part0[32, 0:8].tolist()}  (subblock 1's first row)")
        ws0_f32 = ws_cpu[0].float()
        ws1_f32 = ws_cpu[1].float()
        d0_diff = (debug_part0 - ws0_f32).abs()
        d1_diff = (debug_part1 - ws1_f32).abs()
        _step(
            f"  debug_part0 vs ws[0].float(): max_abs={d0_diff.max().item():.4f} "
            f"mean_abs={d0_diff.mean().item():.4f} "
            f"any_nan={debug_part0.isnan().any().item()} "
            f"any_inf={debug_part0.isinf().any().item()}"
        )
        _step(
            f"  debug_part1 vs ws[1].float(): max_abs={d1_diff.max().item():.4f} "
            f"mean_abs={d1_diff.mean().item():.4f} "
            f"any_nan={debug_part1.isnan().any().item()} "
            f"any_inf={debug_part1.isinf().any().item()}"
        )
        _step(
            f"  debug_part0 stats: min={debug_part0.min().item():.2f} "
            f"max={debug_part0.max().item():.2f} "
            f"mean={debug_part0.mean().item():.2f}"
        )

        # ----- 3a-bisect-4: post-TROWEXPANDMUL pre-TCOLEXPANDMUL dump -----
        # Kernel TSTORE'd `partF32` after `TROWEXPANDMUL(partF32, ascaleF32)`
        # but before `TCOLEXPANDMUL` into ws slots 4/5 (reinterpreted as
        # fp32). Expected: ws_f32[4][m, n] = ws[0].float()[m, n] *
        # ascales[0, m].float() for all m in [0, 64). Day 4 finding:
        # out[0,:] EXACTLY matches ref[0,:] but rows 1+ are ~0. If
        # TROWEXPANDMUL only writes row 0 of its [32, 128] sub-tile,
        # debug_post_row[0,:] should match expected, debug_post_row[1:32, :]
        # should be either stale or = ws[0].float()[1:32, :] unscaled.
        # Symmetric check on AIV1's row band [32, 64).
        debug_post_row0 = ws_f32[4]
        debug_post_row1 = ws_f32[5]
        # Expected: per-row scaled partial.
        ascales_cpu = ascales.float()        # [2, M]
        wscales_cpu = wscales.float()        # [2, N]
        ws0_f = ws_cpu[0].float()
        ws1_f = ws_cpu[1].float()
        expected_post_row0 = ws0_f * ascales_cpu[0].unsqueeze(1)  # [M, N]
        expected_post_row1 = ws1_f * ascales_cpu[1].unsqueeze(1)
        d4_diff0 = (debug_post_row0 - expected_post_row0).abs()
        d4_diff1 = (debug_post_row1 - expected_post_row1).abs()
        _step("  vec post-TROWEXPANDMUL dump (workspace slot 4/5 viewed as fp32)")
        _step(f"  debug_post_row0[0, 0:4] = {debug_post_row0[0, 0:4].tolist()}")
        _step(f"  expected_post_row0[0, 0:4] = {expected_post_row0[0, 0:4].tolist()}")
        _step(f"  debug_post_row0[1, 0:4] = {debug_post_row0[1, 0:4].tolist()}")
        _step(f"  expected_post_row0[1, 0:4] = {expected_post_row0[1, 0:4].tolist()}")
        _step(f"  debug_post_row0[31, 0:4] = {debug_post_row0[31, 0:4].tolist()}  (last row of AIV0)")
        _step(f"  debug_post_row0[32, 0:4] = {debug_post_row0[32, 0:4].tolist()}  (AIV1's row 0)")
        _step(f"  expected_post_row0[32, 0:4] = {expected_post_row0[32, 0:4].tolist()}")
        _step(f"  debug_post_row0[33, 0:4] = {debug_post_row0[33, 0:4].tolist()}")
        _step(
            f"  debug_post_row0 vs expected: max_abs={d4_diff0.max().item():.4f} "
            f"mean_abs={d4_diff0.mean().item():.4f}"
        )
        # Per-row diagnosis — collapse to per-row max diff so we can see
        # which rows TROWEXPANDMUL actually wrote.
        per_row_diff0 = d4_diff0.max(dim=1).values
        _step(f"  debug_post_row0 per-row max_abs: {per_row_diff0.tolist()}")
        _step(
            f"  debug_post_row1 vs expected: max_abs={d4_diff1.max().item():.4f} "
            f"mean_abs={d4_diff1.mean().item():.4f}"
        )

        # ----- 3a-bisect-5a: ascaleF32 tile dump (post-TLOAD+TCVT) -----
        # Kernel TSTORE'd ascaleF32 [32, 1] ColMajor into ws slot 6 (kb=0
        # only). AIV0 writes at ws_f32[6, 0, 0:32], AIV1 at ws_f32[6, 0,
        # 32:64]. Compare against ascales[0, 0:64].float(). If matches →
        # TLOAD/TCVT correct, bug is downstream (in TROWEXPANDMUL itself,
        # most likely vbrcb).
        debug_ascale = ws_f32[6]                # [64, 128] fp32 view
        ascale_flat = ascales_cpu[0]            # [M=64]
        ascale_dumped = debug_ascale[0, :64]
        ascale_diff = (ascale_dumped - ascale_flat).abs()
        _step("  vec ascaleF32 dump (ws slot 6, kb=0)")
        _step(f"  ascale_dumped[0:8] = {ascale_dumped[0:8].tolist()}")
        _step(f"  ascale_ref   [0:8] = {ascale_flat[0:8].tolist()}")
        _step(f"  ascale_dumped[24:32] = {ascale_dumped[24:32].tolist()}")
        _step(f"  ascale_ref   [24:32] = {ascale_flat[24:32].tolist()}")
        _step(f"  ascale_dumped[32:40] = {ascale_dumped[32:40].tolist()}  (AIV1's row 0..7)")
        _step(f"  ascale_ref   [32:40] = {ascale_flat[32:40].tolist()}")
        _step(
            f"  ascale dump vs ref: max_abs={ascale_diff.max().item():.6f} "
            f"mean_abs={ascale_diff.mean().item():.6f}"
        )

        # ----- 3a-bisect-5b: vbrcb tmpbuf dump (post-TROWEXPANDMUL) -----
        # Kernel TSTORE'd tmpbuf @ TMP_UB_OFFSET as [32, 8] RowMajor into
        # ws slot 7 (kb=0). After vbrcb, tmpbuf should hold:
        #   block r (cols 0..7) = [s_r, s_r, ..., s_r]   ← 8 copies
        # AIV0 writes ws[7][0..31, 0..7], AIV1 writes ws[7][32..63, 0..7].
        # Verify: for r in [0, 32), tmpbuf[r, 0..7] should all equal
        # ascale[0, r]. If yes → vbrcb works, bug is in vmul. If no →
        # vbrcb is broken (row dispatch confused).
        debug_tmpbuf = ws_f32[7]                # [64, 128] fp32 view
        _step("  vec vbrcb tmpbuf dump (ws slot 7, kb=0, viewed as [32 blocks, 8 fp32])")
        for r in [0, 1, 2, 5, 16, 24, 31]:
            row = debug_tmpbuf[r, 0:8].tolist()
            expected_s = ascale_flat[r].item()
            _step(f"  tmpbuf[block {r}, :8] = {row}  (expected 8x {expected_s:.6f})")
        # Check uniformity within each block
        block_max = debug_tmpbuf[:32, 0:8].max(dim=1).values
        block_min = debug_tmpbuf[:32, 0:8].min(dim=1).values
        within_block_range = (block_max - block_min).abs()
        _step(f"  within-block max-min: max={within_block_range.max().item():.6f} mean={within_block_range.mean().item():.6f}")
        _step("    (should be 0 if vbrcb broadcasts each scalar into 8 contiguous fp32)")
        # Check that block r's value = ascale[r]
        block_vals = debug_tmpbuf[:32, 0]       # take first element of each block
        block_diff = (block_vals - ascale_flat[0:32]).abs()
        _step(f"  block[r, 0] vs ascale[r] for r in [0, 32): max_abs={block_diff.max().item():.6f} mean_abs={block_diff.mean().item():.6f}")

        _step("  comparing to ref")
        out_cpu = out.cpu()
        _step(f"  out [32,:8] = {out_cpu[32, :8].tolist()}  (AIV1's row 0)")
        _step(f"  ref [32,:8] = {ref[32, :8].tolist()}")
        _step(f"  out [33,:8] = {out_cpu[33, :8].tolist()}")
        _step(f"  ref [33,:8] = {ref[33, :8].tolist()}")

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
