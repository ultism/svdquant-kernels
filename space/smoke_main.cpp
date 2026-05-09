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
//   2d — Cube fires kNumTiles tiles into a kRingSlots ring with a
//        preload + main-loop double-stage driver and back-pressure
//        from vec via VEC_TILE_CONSUMED. Vec consumes each tile and
//        TSTOREs to a linear [kNumTiles, M, N] vec_out region after
//        the ring. Each tile reuses the same act/wgt (mock), so
//        every vec_out tile carries the same 192.0 sentinel; smoke
//        validates ALL kNumTiles segments, which exercises slot
//        reuse (tiles 6-7 land back in slots 0-1 reused after vec
//        consumed them).
//
// Stdout is what the Gradio app pipes back to the page; the smoke
// emits one of:
//   smoke OK     — every checked element matches reference within
//                  fp32 epsilon. Phase 2d validates the entire
//                  vec_out region (kNumTiles × M × N elements); the
//                  cube ring's final state is a function of slot
//                  reuse so we don't pin those values directly.
//   smoke FAILED <stage> — earlier ACL error or numeric mismatch.
//                  Stages: Init, NoDevice, SetDevice, Stream, Malloc,
//                  Memcpy, Memset, Sync, MemcpyD2H, MismatchVecOut.

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
constexpr int   kRingSlots = 6;     // cube ring (Phase 2b/2d)
constexpr int   kNumTiles  = 8;     // tiles produced per launch (Phase 2d)
                                    // — must match kernel_device.cpp.
constexpr float kVecScale  = 0.5f;  // matches device-side kVecScale

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
    // out covers cube ring + linear vec_out tail (kNumTiles slots);
    // kernel.cpp's DeviceParams.out points to its base, and the device
    // kernel splits via fixed offsets:
    //   [0 .. kRingSlots)                cube ring
    //   [kRingSlots .. kRingSlots+kNumTiles)
    //                                    vec_out, one slot per tile
    const int    kTotalSlots = kRingSlots + kNumTiles;
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

    std::printf("[smoke] launching gemm_w4a4 (Phase 2d preload+main, M=%d K=%d N=%d, ring=%d, num_tiles=%d, vec_scale=%.2f)\n",
                kTileM, kTileK, kTileN, kRingSlots, kNumTiles, kVecScale);
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

    // Validate every vec_out tile: each = K * scale + K = 128 * 0.5 + 128 = 192.
    // The cube ring's final state isn't directly validated because slots
    // get re-written as the kNumTiles loop wraps around the kRingSlots
    // ring; the vec_out tiles are the canonical evidence that every
    // produced cube tile was correctly consumed. Mismatches per tile
    // pinpoint which iteration of the preload/main-loop wave broke.
    const float expectedRing = static_cast<float>(kTileK);
    const float expectedVec  = expectedRing * kVecScale + expectedRing;
    int totalMismatches  = 0;
    int firstBadTile     = -1;
    int firstBadIdxInTile = -1;
    float firstBadVal    = 0.0f;

    for (int t = 0; t < kNumTiles; ++t) {
        const size_t tile_base = vecOutBaseElems + static_cast<size_t>(t) * slotElems;
        int tileMismatches = 0;
        for (int m = 0; m < kTileM; ++m) {
            for (int n = 0; n < kTileN; ++n) {
                const size_t idx = tile_base + static_cast<size_t>(m) * kTileN + n;
                const float v = hOut[idx];
                if (std::fabs(v - expectedVec) > 1e-3f) {
                    if (totalMismatches == 0) {
                        firstBadTile      = t;
                        firstBadIdxInTile = m * kTileN + n;
                        firstBadVal       = v;
                    }
                    ++tileMismatches;
                    ++totalMismatches;
                }
            }
        }
        std::printf("[smoke] vec_out tile %d/%d: %d mismatches\n",
                    t, kNumTiles, tileMismatches);
    }

    if (totalMismatches == 0) {
        std::printf("[smoke] vec_out validated: %d tiles x %d x %d = %d elements all = %.1f (= K*scale+K = %d*%.1f+%d)\n",
                    kNumTiles, kTileM, kTileN, kNumTiles * kTileM * kTileN,
                    expectedVec, kTileK, kVecScale, kTileK);
    } else {
        std::printf("[smoke] vec_out mismatch: %d total; first at tile %d, in-tile idx %d = %.4f (expected %.1f)\n",
                    totalMismatches, firstBadTile, firstBadIdxInTile, firstBadVal, expectedVec);
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
