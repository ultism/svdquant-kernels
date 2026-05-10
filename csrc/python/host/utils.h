// torch_npu op extension adapter — extracts NPU device pointers and
// the current NPU stream from at::Tensor arguments, then dispatches
// the kernel through `at_npu::native::OpCommand::RunOpApi` so the
// torch_npu graph executor can track it (lazy / async dispatch
// remains intact).
//
// Layout mirrors PTO's reference op extension at
// `pto-isa/demos/baseline/gemm_basic/csrc/host/utils.h`. The macro
// `INVOKE_PTO_KERNEL(kernel_name, blk, args...)` does three things:
//   1. AdaptKernelArg unwraps each at::Tensor → `arg.storage().data()`
//      (the underlying NPU device pointer); non-Tensor args pass
//      through unchanged.
//   2. Captures the current NPU stream via `c10_npu::getCurrentNPUStream`.
//   3. Calls `ACLRT_LAUNCH_KERNEL(kernel_name)(blockDim, stream, args...)`
//      inside `RunOpApi(name, fn)` so the op shows up in profiling and
//      participates in graph capture if active.
//
// `ACLRT_LAUNCH_KERNEL(kernel_name)` is a macro provided by CANN's
// auto-generated `aclrtlaunch_<kernel>.h` (the same header our host
// launcher already includes). Don't rename — the macro is matched by
// preprocessor concatenation against the kernel symbol.

#ifndef SVDQUANT_PYTHON_HOST_UTILS_H
#define SVDQUANT_PYTHON_HOST_UTILS_H

#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace svdquant_op {

// torch_npu uses the PrivateUse1 dispatch key for NPU. Kept as a
// constexpr alias here so TORCH_CHECK call sites stay readable.
constexpr auto kNpuDevice = c10::DeviceType::PrivateUse1;

template <typename Arg>
decltype(auto) AdaptKernelArg(Arg&& arg)
{
    if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<Arg>>,
                                  at::Tensor>) {
        // `storage().data()` returns the underlying device pointer for
        // NPU tensors (PrivateUse1 backend) — see torch_npu's
        // NPUStorageImpl. Equivalent to `data_ptr()` for contiguous
        // tensors but works through views and offset slices too.
        return const_cast<void*>(arg.storage().data());
    } else {
        return std::forward<Arg>(arg);
    }
}

template <typename... Args>
auto AdaptKernelArgs(Args&&... args)
{
    return std::make_tuple(AdaptKernelArg(std::forward<Args>(args))...);
}

#define INVOKE_PTO_KERNEL(kernel_name, blk, ...)                                \
    do {                                                                        \
        auto __s = c10_npu::getCurrentNPUStream().stream(false);                \
        auto __p = ::svdquant_op::AdaptKernelArgs(__VA_ARGS__);                 \
        auto __fn = [__s, blk, __p]() -> int {                                  \
            uint32_t __rc = 0;                                                  \
            std::apply([&](auto&&... __a) {                                     \
                __rc = ACLRT_LAUNCH_KERNEL(kernel_name)(blk, __s, __a...);      \
            }, __p);                                                            \
            return __rc;                                                        \
        };                                                                      \
        at_npu::native::OpCommand::RunOpApi(#kernel_name, __fn);                \
    } while (false)

}  // namespace svdquant_op

#endif  // SVDQUANT_PYTHON_HOST_UTILS_H
