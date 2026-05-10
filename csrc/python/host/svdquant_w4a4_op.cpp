// torch op binding for `svdquant::gemm_w4a4` — Phase 3a INT4 main path.
//
// Registers a single op into the `svdquant` namespace's PrivateUse1
// (NPU) dispatch table:
//
//   torch.ops.svdquant.gemm_w4a4(act, wgt, ascales, wscales) -> Tensor
//
// Inputs are packed signed-INT4 activation + weight + matching per-
// 64-K-block fp16 scales. Output is fp16 [M, N]. Workspace for the
// cube/vec int32 ring is allocated here (caching alloc, scoped to
// the call) and not exposed to the caller.
//
// Phase 3b/3c will extend to (act, wgt, ascales, wscales, lora_act_in,
// lora_up, bias?, wcscales?, smooth_next?) and return a tuple with
// optional qout / oscales; the binding layer pattern stays the same,
// only the host launcher signature grows.

#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "gemm_w4a4.h"

namespace svdquant_op {

constexpr auto kNpuDevice = c10::DeviceType::PrivateUse1;

// Phase 3a tile is hardcoded — must match
// `csrc/kernels/gemm_w4a4/ascend/kernel_device.cpp` constexpr block.
// Tile-parameterization comes in Phase 3b/3c.
constexpr int64_t kPhase3aM         = 64;
constexpr int64_t kPhase3aK         = 128;
constexpr int64_t kPhase3aN         = 128;
constexpr int64_t kPhase3aBlockSize = 64;        // K-block / mad_s4 KS
constexpr int64_t kPhase3aRingSlots = 6;

at::Tensor run_gemm_w4a4(const at::Tensor& act,
                         const at::Tensor& wgt,
                         const at::Tensor& ascales,
                         const at::Tensor& wscales)
{
    TORCH_CHECK(act.device().type() == kNpuDevice,
                "act must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(wgt.device().type() == kNpuDevice,
                "wgt must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(ascales.device().type() == kNpuDevice,
                "ascales must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(wscales.device().type() == kNpuDevice,
                "wscales must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(act.scalar_type() == at::kByte, "act must be uint8 (packed INT4)");
    TORCH_CHECK(wgt.scalar_type() == at::kByte, "wgt must be uint8 (packed INT4)");
    TORCH_CHECK(ascales.scalar_type() == at::kHalf, "ascales must be float16");
    TORCH_CHECK(wscales.scalar_type() == at::kHalf, "wscales must be float16");
    TORCH_CHECK(act.dim() == 2 && wgt.dim() == 2 && ascales.dim() == 2 && wscales.dim() == 2,
                "all tensors must be 2D");

    constexpr int64_t kK_packed = kPhase3aK / 2;
    constexpr int64_t kK_blocks = kPhase3aK / kPhase3aBlockSize;
    TORCH_CHECK(act.size(0) == kPhase3aM && act.size(1) == kK_packed,
                "act shape must be [", kPhase3aM, ", ", kK_packed, "] (Phase 3a)");
    TORCH_CHECK(wgt.size(0) == kPhase3aN && wgt.size(1) == kK_packed,
                "wgt shape must be [", kPhase3aN, ", ", kK_packed, "] (Phase 3a)");
    TORCH_CHECK(ascales.size(0) == kK_blocks && ascales.size(1) == kPhase3aM,
                "ascales shape must be [", kK_blocks, ", ", kPhase3aM, "] (Phase 3a)");
    TORCH_CHECK(wscales.size(0) == kK_blocks && wscales.size(1) == kPhase3aN,
                "wscales shape must be [", kK_blocks, ", ", kPhase3aN, "] (Phase 3a)");
    TORCH_CHECK(act.is_contiguous() && wgt.is_contiguous(),
                "act and wgt must be contiguous");
    TORCH_CHECK(ascales.is_contiguous() && wscales.is_contiguous(),
                "ascales and wscales must be contiguous");

    auto fp16_options = act.options().dtype(at::kHalf);
    auto i32_options  = act.options().dtype(at::kInt);

    // Workspace = cube/vec hand-off ring of int32 partials. Caching
    // allocator keeps re-alloc cost ~free across calls. Lifetime
    // ends with this `at::Tensor` going out of scope at function exit.
    // Use zeros (not empty) so stale garbage can't masquerade as a
    // cube-written partial if a slot is read before it's filled —
    // makes "vec read garbage" visible as zero output rather than inf.
    auto workspace = at::zeros(
        {kPhase3aRingSlots, kPhase3aM, kPhase3aN}, i32_options);
    auto out = at::empty({kPhase3aM, kPhase3aN}, fp16_options);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    svdquant::ascend::gemm_w4a4(
        const_cast<void*>(act.storage().data()),
        const_cast<void*>(wgt.storage().data()),
        const_cast<void*>(ascales.storage().data()),
        const_cast<void*>(wscales.storage().data()),
        workspace.data_ptr(),
        out.data_ptr(),
        static_cast<void*>(stream));
    return out;
}

}  // namespace svdquant_op

namespace {

TORCH_LIBRARY_FRAGMENT(svdquant, m)
{
    // Phase 3a schema — INT4 main path (no LoRA / bias / wcscales /
    // next-layer quant yet). Phase 3b/3c append optional Tensors and
    // turn the return type into a tuple to surface qout / oscales.
    m.def("gemm_w4a4(Tensor act, Tensor wgt, Tensor ascales, Tensor wscales) -> Tensor");
}

TORCH_LIBRARY_IMPL(svdquant, PrivateUse1, m)
{
    m.impl("gemm_w4a4", TORCH_FN(svdquant_op::run_gemm_w4a4));
}

}  // namespace
