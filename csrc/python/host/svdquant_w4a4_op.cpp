// torch op binding for `svdquant::gemm_w4a4` — Phase 2f scaffolding.
//
// Registers a single op into the `svdquant` namespace's PrivateUse1
// (NPU) dispatch table:
//
//   torch.ops.svdquant.gemm_w4a4(act, wgt) -> Tensor
//
// where `act` and `wgt` are NPU half tensors with the shape that
// Phase 2d's mock kernel hardcodes (act [64, 128], wgt [128, 128]).
// Output is fp32 and over-allocated to hold cube ring scratch
// (kRingSlots × M × N) + linear vec_out (kNumTiles × M × N), matching
// the exact layout the test harness expects.
//
// Phase 3 will swap this two-arg signature for the full SVDQuant op:
//
//   svdquant.gemm_w4a4(act_int4, wgt_int4, ascales, wscales,
//                      lora_act_in, lora_up, bias?, wcscales?,
//                      smooth_next?) -> (out, qout?, oscales?)
//
// At that point the host launcher in csrc/kernels/gemm_w4a4/ascend/
// also extends, but this file stays the only Python entry point.

#include "utils.h"

#include "aclrtlaunch_svdquant_gemm_w4a4_kernel.h"

namespace svdquant_op {

// Phase 2d / 2f mock kernel constants — must match
// `csrc/kernels/gemm_w4a4/ascend/kernel_device.cpp` constexpr block.
// These won't be hardcoded forever; Phase 3 tile-parameterizes the
// kernel and the binding then forwards M/K/N from the input tensor
// shapes rather than asserting them.
constexpr int64_t kPhase2dM        = 64;
constexpr int64_t kPhase2dK        = 128;
constexpr int64_t kPhase2dN        = 128;
constexpr int64_t kPhase2dRing     = 6;
constexpr int64_t kPhase2dNumTiles = 8;

at::Tensor run_gemm_w4a4(const at::Tensor& act, const at::Tensor& wgt)
{
    TORCH_CHECK(act.device().type() == kNpuDevice,
                "act must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(wgt.device().type() == kNpuDevice,
                "wgt must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(act.scalar_type() == at::kHalf, "act must be float16");
    TORCH_CHECK(wgt.scalar_type() == at::kHalf, "wgt must be float16");
    TORCH_CHECK(act.dim() == 2 && wgt.dim() == 2,
                "act and wgt must be 2D tensors");
    TORCH_CHECK(act.size(0) == kPhase2dM && act.size(1) == kPhase2dK,
                "act shape must be [", kPhase2dM, ", ", kPhase2dK, "] (Phase 2f mock)");
    TORCH_CHECK(wgt.size(0) == kPhase2dK && wgt.size(1) == kPhase2dN,
                "wgt shape must be [", kPhase2dK, ", ", kPhase2dN, "] (Phase 2f mock)");
    TORCH_CHECK(act.is_contiguous(), "act must be contiguous");
    TORCH_CHECK(wgt.is_contiguous(), "wgt must be contiguous");

    // Output is over-allocated to fit the cube ring scratch followed
    // by the linear vec_out slots. The caller (test) slices the
    // vec_out portion out by skipping the first kRingSlots × M × N
    // fp32 elements. Phase 3 separates the scratch into its own
    // workspace and `out` becomes [M_total, N] only.
    const int64_t total_slots = kPhase2dRing + kPhase2dNumTiles;
    auto out = at::empty({total_slots * kPhase2dM, kPhase2dN},
                         act.options().dtype(at::kFloat));

    constexpr uint32_t blockDim = 1;
    INVOKE_PTO_KERNEL(svdquant_gemm_w4a4_kernel, blockDim, act, wgt, out);
    return out;
}

}  // namespace svdquant_op

namespace {

TORCH_LIBRARY_FRAGMENT(svdquant, m)
{
    // Phase 2f schema — minimal, two NPU half tensors in, one fp32
    // tensor out. Phase 3 will append optional ascales/wscales/lora_*
    // and turn the return type into a tuple to surface qout/oscales.
    m.def("gemm_w4a4(Tensor act, Tensor wgt) -> Tensor");
}

TORCH_LIBRARY_IMPL(svdquant, PrivateUse1, m)
{
    m.impl("gemm_w4a4", TORCH_FN(svdquant_op::run_gemm_w4a4));
}

}  // namespace
