// Phase 2 NPU launch smoke — runs on the GitCode Space 910B container.
//
// What it proves at each phase:
//   2a — both AIC and AIV launch from one aclrtlaunch call, and the
//        cube↔vec FFTS handshake unblocks (no deadlock).
//   2b — cube does a fp16 GEMM via pto_macro_matmul, FIX-pipe TSTOREs
//        to a 6-slot GM ring, and slot 0 contains `act @ wgt.T` with
//        all elements equal to Tile_K when act = wgt = ones.
//   2c — AIV (1:2 mix mode, two subblocks) consumes slot 0, applies
//        TROWEXPANDMUL by a per-row constant scale, TADDs the raw
//        cube tile back, and TSTOREs to a vec_out segment placed
//        right after the cube ring inside `out`. Each subblock owns
//        its own M/2 row stripe; both stripes get validated.
//        Mock math: vec_out[m,n] = K * scale + K = 128 * 0.5 + 128 = 192.
//
// Stdout is what the Gradio app pipes back to the page; the smoke
// emits one of:
//   smoke OK     — every checked element matches reference within
//                  fp32 epsilon (cube ring slot 0 + vec_out segment).
//   smoke FAILED <stage> — earlier ACL error or numeric mismatch.
//                  Stages: Init, NoDevice, SetDevice, Stream, Malloc,
//                  Memcpy, Memset, Sync, MemcpyD2H, MismatchSlot0,
//                  MismatchVecOut.

#include <acl/acl.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gemm_w4a4.h"

namespace {

constexpr int   kTileM     = 64;
constexpr int   kTileK     = 128;
constexpr int   kTileN     = 128;
constexpr int   kRingSlots = 6;     // cube ring (Phase 2b)
constexpr int   kVecOutSlots = 1;   // tail segment after the ring (Phase 2c)
constexpr float kVecScale  = 0.5f;  // matches kernel.cpp's launcher-side const

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
    // out covers cube ring + vec_out tail; smoke knows both are inside
    // the same allocation, kernel.cpp's DeviceParams.out points to its
    // base, and the device kernel splits via fixed offsets.
    const int    kTotalSlots = kRingSlots + kVecOutSlots;
    const size_t outBytes    = sizeof(float) * kTotalSlots * kTileM * kTileN;
    const size_t slotElems   = static_cast<size_t>(kTileM) * kTileN;
    const size_t vecOutBaseElems = static_cast<size_t>(kRingSlots) * slotElems;

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
    // from a kernel that writes the right value — covers both the
    // cube ring and the vec_out tail in one call.
    if (aclrtMemset(dOut, outBytes, 0, outBytes) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtMemset failed: %s\n", AclErr());
        return cleanup(7, "Memset");
    }

    // ---- pack params and launch ----
    svdquant::GemmW4A4Params p{};
    p.act.data = dAct;
    p.wgt.data = dWgt;
    p.out.data = dOut;

    std::printf("[smoke] launching gemm_w4a4 (Phase 2c cube+vec mock, M=%d K=%d N=%d, vec_scale=%.2f)\n",
                kTileM, kTileK, kTileN, kVecScale);
    svdquant::ascend::gemm_w4a4(p, stream);

    std::printf("[smoke] aclrtSynchronizeStream\n");
    aclError sync_ret = aclrtSynchronizeStream(stream);
    if (sync_ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSynchronizeStream failed (%d): %s\n",
                     static_cast<int>(sync_ret), AclErr());
        return cleanup(8, "Sync");
    }

    // ---- D2H whole buffer + validate ring slot 0 + vec_out ----
    std::vector<float> hOut(static_cast<size_t>(kTotalSlots) * slotElems, -1.0f);
    if (aclrtMemcpy(hOut.data(), outBytes, dOut, outBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::fprintf(stderr, "out D2H failed: %s\n", AclErr());
        return cleanup(9, "MemcpyD2H");
    }

    // (1) Cube ring slot 0: act = wgt = ones(half) → out[m,n] = K = 128.
    const float expectedRing = static_cast<float>(kTileK);
    int ringMismatches = 0;
    int firstRingBadIdx = -1;
    float firstRingBadVal = 0.0f;
    for (int m = 0; m < kTileM; ++m) {
        for (int n = 0; n < kTileN; ++n) {
            const int idx = m * kTileN + n;  // slot 0 starts at offset 0
            const float v = hOut[idx];
            if (std::fabs(v - expectedRing) > 1e-3f) {
                if (ringMismatches == 0) {
                    firstRingBadIdx = idx;
                    firstRingBadVal = v;
                }
                ++ringMismatches;
            }
        }
    }
    if (ringMismatches == 0) {
        std::printf("[smoke] cube ring slot 0 validated: %d x %d elements all = %.1f (= Tile_K)\n",
                    kTileM, kTileN, expectedRing);
    } else {
        std::printf("[smoke] cube ring slot 0 mismatch: %d / %d off; first at idx %d = %.4f (expected %.1f)\n",
                    ringMismatches, kTileM * kTileN, firstRingBadIdx, firstRingBadVal, expectedRing);
        return cleanup(10, "MismatchSlot0");
    }

    // (2) vec_out: vec_out[m,n] = K * scale + K = 128 * 0.5 + 128 = 192.
    const float expectedVec = expectedRing * kVecScale + expectedRing;
    int vecMismatches = 0;
    int firstVecBadIdx = -1;
    float firstVecBadVal = 0.0f;
    for (int m = 0; m < kTileM; ++m) {
        for (int n = 0; n < kTileN; ++n) {
            const size_t idx = vecOutBaseElems + static_cast<size_t>(m) * kTileN + n;
            const float v = hOut[idx];
            if (std::fabs(v - expectedVec) > 1e-3f) {
                if (vecMismatches == 0) {
                    firstVecBadIdx = static_cast<int>(idx - vecOutBaseElems);
                    firstVecBadVal = v;
                }
                ++vecMismatches;
            }
        }
    }
    if (vecMismatches == 0) {
        std::printf("[smoke] vec_out validated: %d x %d elements all = %.1f (= K*scale+K = %d*%.1f+%d)\n",
                    kTileM, kTileN, expectedVec, kTileK, kVecScale, kTileK);
    } else {
        std::printf("[smoke] vec_out mismatch: %d / %d off; first at idx %d (in vec_out) = %.4f (expected %.1f)\n",
                    vecMismatches, kTileM * kTileN, firstVecBadIdx, firstVecBadVal, expectedVec);
        return cleanup(11, "MismatchVecOut");
    }

    aclrtFree(dAct);
    aclrtFree(dWgt);
    aclrtFree(dOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::printf("smoke OK\n");
    return 0;
}
