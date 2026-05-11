// Host launcher for the Ascend gemm_w4a4 pod.
//
// Phase 3b signature: 9 × device pointer + opaque stream — raw INT4
// inputs + per-K-block fp16 scales + LoRA-up inputs + int32 cube/vec
// hand-off ring + fp32 LoRA hand-off buffer + fp16 output. Caller (the
// torch op wrapper) is responsible for allocating workspace, lora_buf,
// and out from the NPU caching allocator before this call.
//
// The launcher repacks all 9 device pointers into a single device-side
// `DeviceParams` struct (72 B) and H2D-copies it to a small staging
// allocation, because the auto-gen `aclrtlaunch_*` wrapper takes a
// single `GM_ADDR params_addr` (it doesn't expand variadic tensor
// args).
//
// Synchronization: this launcher synchronizes after `aclrtlaunch_*`
// before freeing the staging `dev_params` to avoid use-after-free.
// Phase 3c+ will revisit this when the op participates in a torch_npu
// graph executor that owns its own stream lifetime.

#include "gemm_w4a4.h"

#include <acl/acl.h>
#include "aclrtlaunch_svdquant_gemm_w4a4_kernel.h"

namespace svdquant::ascend {

namespace {

// Mirrors the device-side `DeviceParams` in kernel_device.cpp by byte
// layout: 9 × 8 B = 72 B device pointers in this order. Host and
// device structs cannot share a header because the device file is
// `[aicore]` and ccec rejects dereferencing `void* __gm__` as a typed
// `__gm__ T*`. Keep the two field lists in sync.
struct DeviceParams {
    void* act;         // [M, K/2]                uint8
    void* wgt;         // [N, K/2]                uint8
    void* ascales;     // [K/64, M]               fp16
    void* wscales;     // [K/64, N]               fp16
    void* lora_act_in; // [M, R]                  fp32
    void* lora_up;     // [N, R]                  fp16
    void* workspace;   // [kRingSlots, M, N]      int32 cube/vec ring
    void* lora_buf;    // [M, N]                  fp32 LoRA hand-off
    void* out;         // [M, N]                  fp16 final
};

}  // namespace

void gemm_w4a4(void* act, void* wgt,
               void* ascales, void* wscales,
               void* lora_act_in, void* lora_up,
               void* workspace, void* lora_buf, void* out,
               void* stream) {
    auto raw_stream = static_cast<aclrtStream>(stream);

    DeviceParams dp{act, wgt, ascales, wscales,
                    lora_act_in, lora_up,
                    workspace, lora_buf, out};

    void* dev_params = nullptr;
    if (aclrtMalloc(&dev_params, sizeof(DeviceParams),
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return;
    }
    if (aclrtMemcpy(dev_params, sizeof(DeviceParams),
                    &dp, sizeof(DeviceParams),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(dev_params);
        return;
    }

    // Single-CTA launch for 3a (one BM=128 × BN=256 × BK_logical=2048
    // tile per call). Mix 1:2 → 1 cube + 2 vec subblocks. Phase 3b
    // multi-tile will scale blockDim with M/BM × N/BN.
    constexpr uint32_t blockDim = 1;
    aclrtlaunch_svdquant_gemm_w4a4_kernel(blockDim, raw_stream, dev_params);

    aclrtSynchronizeStream(raw_stream);
    aclrtFree(dev_params);
}

}  // namespace svdquant::ascend
