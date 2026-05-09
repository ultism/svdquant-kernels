// Phase 2 NPU launch smoke — runs on the GitCode Space 910B container.
//
// What it proves at each phase:
//   2a — both AIC and AIV launch from one aclrtlaunch call, and the
//        cube↔vec FFTS handshake unblocks (no deadlock).
//   2b — cube does a fp16 GEMM via pto_macro_matmul, FIX-pipe TSTOREs
//        to a 6-slot GM ring, and slot 0 contains `act @ wgt.T` with
//        all elements equal to Tile_K when act = wgt = ones.
//   2c+ — vec consumes from the ring and rescales/accumulates.
//
// Stdout is what the Gradio app pipes back to the page; the smoke
// emits one of:
//   smoke OK     — every checked element matches reference within
//                  fp32 epsilon (tile_k); validation included.
//   smoke FAILED <stage> — earlier ACL error or numeric mismatch.
//                  Stages: Init, NoDevice, SetDevice, Stream, Malloc,
//                  Memcpy, Memset, Sync, MemcpyD2H, Mismatch.

#include <acl/acl.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gemm_w4a4.h"

namespace {

constexpr int kTileM = 64;
constexpr int kTileK = 128;
constexpr int kTileN = 128;
constexpr int kRingSlots = 6;  // matches kernel_device.cpp; over-allocate ring

const char* AclErr() {
    const char* s = aclGetRecentErrMsg();
    return (s && *s) ? s : "(no aclGetRecentErrMsg)";
}

// IEEE 754 binary16 = 1 sign + 5 exp + 10 mantissa.
// fp16(1.0) = 0x3C00. Easier to bake the constant than to drag in a
// host-side fp32→fp16 conversion lib for a one-shot smoke seed.
uint16_t HalfOne() { return 0x3C00; }

}  // namespace

int main() {
    std::printf("[smoke] aclInit\n");
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclInit failed: %s\n", AclErr());
        std::printf("smoke FAILED Init\n");
        return 1;
    }

    int32_t deviceCount = 0;
    aclrtGetDeviceCount(reinterpret_cast<uint32_t*>(&deviceCount));
    std::printf("[smoke] device count: %d\n", deviceCount);
    if (deviceCount <= 0) {
        std::fprintf(stderr, "no NPU devices visible\n");
        aclFinalize();
        std::printf("smoke FAILED NoDevice\n");
        return 2;
    }

    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSetDevice(0) failed: %s\n", AclErr());
        aclFinalize();
        std::printf("smoke FAILED SetDevice\n");
        return 3;
    }

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtCreateStream failed: %s\n", AclErr());
        aclrtResetDevice(0);
        aclFinalize();
        std::printf("smoke FAILED Stream\n");
        return 4;
    }

    // ---- allocate device buffers + seed act/wgt = ones ----
    const size_t actBytes = sizeof(uint16_t) * kTileM * kTileK;
    const size_t wgtBytes = sizeof(uint16_t) * kTileN * kTileK;
    const size_t outBytes = sizeof(float) * kRingSlots * kTileM * kTileN;

    std::vector<uint16_t> hAct(kTileM * kTileK, HalfOne());
    std::vector<uint16_t> hWgt(kTileN * kTileK, HalfOne());

    void* dAct = nullptr;
    void* dWgt = nullptr;
    void* dOut = nullptr;
    auto cleanup = [&](int code, const char* stage) {
        if (dAct) aclrtFree(dAct);
        if (dWgt) aclrtFree(dWgt);
        if (dOut) aclrtFree(dOut);
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        std::printf("smoke FAILED %s\n", stage);
        return code;
    };

    if (aclrtMalloc(&dAct, actBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&dWgt, wgtBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&dOut, outBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtMalloc failed: %s\n", AclErr());
        return cleanup(5, "Malloc");
    }

    if (aclrtMemcpy(dAct, actBytes, hAct.data(), actBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(dWgt, wgtBytes, hWgt.data(), wgtBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::fprintf(stderr, "act/wgt H2D failed: %s\n", AclErr());
        return cleanup(6, "Memcpy");
    }
    // Zero `out` so a kernel that writes nothing can be distinguished
    // from a kernel that writes the right value.
    if (aclrtMemset(dOut, outBytes, 0, outBytes) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtMemset failed: %s\n", AclErr());
        return cleanup(7, "Memset");
    }

    // ---- pack params and launch ----
    svdquant::GemmW4A4Params p{};
    p.act.data = dAct;
    p.wgt.data = dWgt;
    p.out.data = dOut;

    std::printf("[smoke] launching gemm_w4a4 (Phase 2b cube fp16 mock, M=%d K=%d N=%d)\n",
                kTileM, kTileK, kTileN);
    svdquant::ascend::gemm_w4a4(p, stream);

    std::printf("[smoke] aclrtSynchronizeStream\n");
    aclError sync_ret = aclrtSynchronizeStream(stream);
    if (sync_ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSynchronizeStream failed (%d): %s\n",
                     static_cast<int>(sync_ret), AclErr());
        return cleanup(8, "Sync");
    }

    // ---- D2H slot 0 + validate ----
    std::vector<float> hOut(static_cast<size_t>(kRingSlots) * kTileM * kTileN, -1.0f);
    if (aclrtMemcpy(hOut.data(), outBytes, dOut, outBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::fprintf(stderr, "out D2H failed: %s\n", AclErr());
        return cleanup(9, "MemcpyD2H");
    }

    // act = wgt = ones(half) → out[m, n] = sum_k 1*1 = Tile_K = 128.
    const float expected = static_cast<float>(kTileK);
    int mismatches = 0;
    int first_bad_idx = -1;
    float first_bad_val = 0.0f;
    for (int m = 0; m < kTileM; ++m) {
        for (int n = 0; n < kTileN; ++n) {
            const int idx = m * kTileN + n;  // slot 0 is at offset 0
            const float v = hOut[idx];
            if (std::fabs(v - expected) > 1e-3f) {
                if (mismatches == 0) {
                    first_bad_idx = idx;
                    first_bad_val = v;
                }
                ++mismatches;
            }
        }
    }

    if (mismatches == 0) {
        std::printf("[smoke] slot 0 validated: %d x %d elements all = %.1f (= Tile_K)\n",
                    kTileM, kTileN, expected);
    } else {
        std::printf("[smoke] slot 0 mismatch: %d / %d elements off; first at idx %d = %.4f (expected %.1f)\n",
                    mismatches, kTileM * kTileN, first_bad_idx, first_bad_val, expected);
    }

    aclrtFree(dAct);
    aclrtFree(dWgt);
    aclrtFree(dOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    if (mismatches == 0) {
        std::printf("smoke OK\n");
        return 0;
    }
    std::printf("smoke FAILED Mismatch\n");
    return 10;
}
