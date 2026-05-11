// torch op binding for `svdquant::gemm_w4a4` — Phase 3b INT4 main + LoRA-up.
//
// Registers a single op into the `svdquant` namespace's PrivateUse1
// (NPU) dispatch table:
//
//   torch.ops.svdquant.gemm_w4a4(
//       act, wgt, ascales, wscales, lora_act_in, lora_up) -> Tensor
//
// Inputs are packed signed-INT4 activation + weight + matching per-
// 64-K-block fp16 scales + fp32 LoRA-down output + fp16 LoRA-up
// weight. Output is fp16 [M, N]. Cube/vec int32 ring and fp32 LoRA
// hand-off buffer are allocated as internal workspace here and freed
// after the launcher returns — not user-visible.
//
// Phase 3c will append optional `bias / wcscales` Tensors; the binding
// layer pattern stays the same, only the host launcher signature grows.

#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "gemm_w4a4.h"

namespace svdquant_op {

constexpr auto kNpuDevice = c10::DeviceType::PrivateUse1;

// Phase 3b tile is hardcoded — must match
// `csrc/kernels/gemm_w4a4/ascend/kernel_device.cpp` constexpr block.
// Tile-parameterization comes in Phase 3c.
constexpr int64_t kPhase3bM         = 64;
constexpr int64_t kPhase3bK         = 128;
constexpr int64_t kPhase3bN         = 128;
constexpr int64_t kPhase3bR         = 32;        // LoRA rank
constexpr int64_t kPhase3bBlockSize = 64;        // K-block / mad_s4 KS
constexpr int64_t kPhase3bRingSlots = 6;         // cube/vec int32 hand-off ring depth

at::Tensor
run_gemm_w4a4(const at::Tensor& act,
              const at::Tensor& wgt,
              const at::Tensor& ascales,
              const at::Tensor& wscales,
              const at::Tensor& lora_act_in,
              const at::Tensor& lora_up)
{
    TORCH_CHECK(act.device().type() == kNpuDevice,
                "act must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(wgt.device().type() == kNpuDevice,
                "wgt must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(ascales.device().type() == kNpuDevice,
                "ascales must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(wscales.device().type() == kNpuDevice,
                "wscales must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(lora_act_in.device().type() == kNpuDevice,
                "lora_act_in must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(lora_up.device().type() == kNpuDevice,
                "lora_up must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(act.scalar_type() == at::kByte, "act must be uint8 (packed INT4)");
    TORCH_CHECK(wgt.scalar_type() == at::kByte, "wgt must be uint8 (packed INT4)");
    TORCH_CHECK(ascales.scalar_type() == at::kHalf, "ascales must be float16");
    TORCH_CHECK(wscales.scalar_type() == at::kHalf, "wscales must be float16");
    TORCH_CHECK(lora_act_in.scalar_type() == at::kFloat, "lora_act_in must be float32");
    TORCH_CHECK(lora_up.scalar_type() == at::kHalf, "lora_up must be float16");
    TORCH_CHECK(act.dim() == 2 && wgt.dim() == 2
                && ascales.dim() == 2 && wscales.dim() == 2
                && lora_act_in.dim() == 2 && lora_up.dim() == 2,
                "all tensors must be 2D");

    constexpr int64_t kK_packed = kPhase3bK / 2;
    constexpr int64_t kK_blocks = kPhase3bK / kPhase3bBlockSize;
    TORCH_CHECK(act.size(0) == kPhase3bM && act.size(1) == kK_packed,
                "act shape must be [", kPhase3bM, ", ", kK_packed, "] (Phase 3b)");
    TORCH_CHECK(wgt.size(0) == kPhase3bN && wgt.size(1) == kK_packed,
                "wgt shape must be [", kPhase3bN, ", ", kK_packed, "] (Phase 3b)");
    TORCH_CHECK(ascales.size(0) == kK_blocks && ascales.size(1) == kPhase3bM,
                "ascales shape must be [", kK_blocks, ", ", kPhase3bM, "] (Phase 3b)");
    TORCH_CHECK(wscales.size(0) == kK_blocks && wscales.size(1) == kPhase3bN,
                "wscales shape must be [", kK_blocks, ", ", kPhase3bN, "] (Phase 3b)");
    TORCH_CHECK(lora_act_in.size(0) == kPhase3bM && lora_act_in.size(1) == kPhase3bR,
                "lora_act_in shape must be [", kPhase3bM, ", ", kPhase3bR, "] (Phase 3b)");
    TORCH_CHECK(lora_up.size(0) == kPhase3bN && lora_up.size(1) == kPhase3bR,
                "lora_up shape must be [", kPhase3bN, ", ", kPhase3bR, "] (Phase 3b)");
    TORCH_CHECK(act.is_contiguous() && wgt.is_contiguous(),
                "act and wgt must be contiguous");
    TORCH_CHECK(ascales.is_contiguous() && wscales.is_contiguous(),
                "ascales and wscales must be contiguous");
    TORCH_CHECK(lora_act_in.is_contiguous() && lora_up.is_contiguous(),
                "lora_act_in and lora_up must be contiguous");

    auto fp16_options = act.options().dtype(at::kHalf);
    auto fp32_options = act.options().dtype(at::kFloat);
    auto i32_options  = act.options().dtype(at::kInt);

    // Device cube path consumes fp16 inputs for the LoRA-up mad and
    // expects lora_up indexed K-first ([R, N]). Cast / transpose
    // here — both tensors are small (M*R*2 = 4 KB, R*N*2 = 8 KB at
    // R=32) and live until the launcher returns.
    auto la_fp16 = lora_act_in.to(at::kHalf);
    auto lu_T = lora_up.t().contiguous();

    // Internal scratch: cube/vec int32 ring + fp32 LoRA-up hand-off.
    auto workspace = at::empty(
        {kPhase3bRingSlots, kPhase3bM, kPhase3bN}, i32_options);
    auto lora_buf  = at::empty(
        {kPhase3bM, kPhase3bN}, fp32_options);
    auto out = at::empty({kPhase3bM, kPhase3bN}, fp16_options);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    svdquant::ascend::gemm_w4a4(
        const_cast<void*>(act.storage().data()),
        const_cast<void*>(wgt.storage().data()),
        const_cast<void*>(ascales.storage().data()),
        const_cast<void*>(wscales.storage().data()),
        la_fp16.data_ptr(),
        lu_T.data_ptr(),
        workspace.data_ptr(),
        lora_buf.data_ptr(),
        out.data_ptr(),
        static_cast<void*>(stream));
    return out;
}

}  // namespace svdquant_op

namespace {

TORCH_LIBRARY_FRAGMENT(svdquant, m)
{
    m.def("gemm_w4a4(Tensor act, Tensor wgt, Tensor ascales, Tensor wscales, "
          "Tensor lora_act_in, Tensor lora_up) -> Tensor");
}

TORCH_LIBRARY_IMPL(svdquant, PrivateUse1, m)
{
    m.impl("gemm_w4a4", TORCH_FN(svdquant_op::run_gemm_w4a4));
}

}  // namespace
