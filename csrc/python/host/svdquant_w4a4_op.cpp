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
// Implementation routes through our existing host launcher
// `svdquant::ascend::gemm_w4a4(act, wgt, out, stream)`, which internally
// packs a `DeviceParams` struct + H2D copies it onto the device — that
// extra step is required because Phase 2d's device kernel signature is
// `(GM_ADDR params_addr)` (a single typed-pointer struct in GM), not
// the raw N-pointer pattern PTO `gemm_basic` uses. We keep the device
// kernel signature unchanged for now (Phase 3 may flatten it), and
// just bridge tensor → pointer here.
//
// Phase 3 will extend the op signature to (act_int4, wgt_int4, ascales,
// wscales, lora_act_in, lora_up, bias?, wcscales?, smooth_next?) and
// return a tuple including optional qout / oscales; the binding layer
// stays at this level and only the host launcher signature grows.

#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "gemm_w4a4.h"

namespace svdquant_op {

constexpr auto kNpuDevice = c10::DeviceType::PrivateUse1;

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
    // workspace and `out` becomes [M, N] only.
    const int64_t total_slots = kPhase2dRing + kPhase2dNumTiles;
    auto out = at::empty({total_slots * kPhase2dM, kPhase2dN},
                         act.options().dtype(at::kFloat));

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    svdquant::ascend::gemm_w4a4(
        const_cast<void*>(act.storage().data()),
        const_cast<void*>(wgt.storage().data()),
        out.data_ptr(),
        static_cast<void*>(stream));
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
