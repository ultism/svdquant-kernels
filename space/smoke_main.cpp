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
//        cube tile back, and TSTOREs to a vec_out segment.
//   2d — Cube fires kNumTiles tiles into a kRingSlots ring with a
//        preload + main-loop double-stage driver and back-pressure
//        from vec via VEC_TILE_CONSUMED. Vec consumes each tile and
//        TSTOREs to a linear [kNumTiles, M, N] vec_out region after
//        the ring. All kNumTiles tiles reuse the same act/wgt, so
//        every vec_out tile must equal the same reference; sentinel
//        校验 (= 192 for all-ones) was the Phase 2d acceptance.
//   2e — Phase 2d numerics carried over to random fp16 act/wgt with
//        an externally-supplied PyTorch/numpy reference. Smoke takes
//        three argv paths (act.bin, wgt.bin, ref.bin), H2Ds act/wgt,
//        runs the kernel, D2Hs out, and diffs every vec_out tile
//        against ref element-wise within an fp16-scale tolerance.
//        Acceptance: all kNumTiles tiles match ref → "smoke OK".
//
// Stdout is what the Gradio app pipes back to the page; the smoke
// emits one of:
//   smoke OK     — every vec_out tile matches ref element-wise
//                  within tolerance.
//   smoke FAILED <stage> — earlier ACL error, missing argv, file IO,
//                  or numeric mismatch. Stages: Argv, ReadAct, ReadWgt,
//                  ReadRef, Init, NoDevice, SetDevice, Stream, Malloc,
//                  Memcpy, Memset, Sync, MemcpyD2H, MismatchVecOut.

#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

// Tolerance: cube's fp16 inputs go through a fp32 accumulator over K=128
// terms; numpy's fp32 reduction order differs from the cube intrinsic, so
// we compare with both a relative and absolute slack. fp16 quantization
// of the input alone bounds rel error ≈ K * 2^-10 ≈ 0.125; we double
// that for headroom. Magnitudes near zero need an abs floor.
constexpr float kRelTol    = 0.25f;
constexpr float kAbsTol    = 1e-2f;

const char* AclErr() {
    const char* s = aclGetRecentErrMsg();
    return (s && *s) ? s : "(no aclGetRecentErrMsg)";
}

// Read `expected_bytes` from a file into `dst`. Returns true on success.
// Dumps the failure reason to stderr.
bool ReadAll(const char* path, void* dst, size_t expected_bytes) {
    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "fopen('%s') failed: %s\n", path, std::strerror(errno));
        return false;
    }
    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (sz < 0 || static_cast<size_t>(sz) != expected_bytes) {
        std::fprintf(stderr, "size mismatch on '%s': got %ld, expected %zu\n",
                     path, sz, expected_bytes);
        std::fclose(fp);
        return false;
    }
    size_t n = std::fread(dst, 1, expected_bytes, fp);
    std::fclose(fp);
    if (n != expected_bytes) {
        std::fprintf(stderr, "short read on '%s': got %zu, expected %zu\n",
                     path, n, expected_bytes);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s <act.bin> <wgt.bin> <ref.bin>\n"
                     "  act.bin: fp16 [%d, %d] row-major\n"
                     "  wgt.bin: fp16 [%d, %d] row-major\n"
                     "  ref.bin: fp32 [%d, %d] row-major (single tile; smoke "
                     "diffs every vec_out tile against this)\n",
                     argv[0], kTileM, kTileK, kTileN, kTileK, kTileM, kTileN);
        std::printf("smoke FAILED Argv\n");
        return 1;
    }
    const char* actPath = argv[1];
    const char* wgtPath = argv[2];
    const char* refPath = argv[3];

    // ---- read host-side inputs + reference ----
    const size_t actBytes = sizeof(uint16_t) * kTileM * kTileK;
    const size_t wgtBytes = sizeof(uint16_t) * kTileN * kTileK;
    const size_t refBytes = sizeof(float)    * kTileM * kTileN;

    std::vector<uint16_t> hAct(static_cast<size_t>(kTileM) * kTileK);
    std::vector<uint16_t> hWgt(static_cast<size_t>(kTileN) * kTileK);
    std::vector<float>    hRef(static_cast<size_t>(kTileM) * kTileN);

    if (!ReadAll(actPath, hAct.data(), actBytes)) { std::printf("smoke FAILED ReadAct\n"); return 2; }
    if (!ReadAll(wgtPath, hWgt.data(), wgtBytes)) { std::printf("smoke FAILED ReadWgt\n"); return 3; }
    if (!ReadAll(refPath, hRef.data(), refBytes)) { std::printf("smoke FAILED ReadRef\n"); return 4; }

    std::printf("[smoke] aclInit\n");
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclInit failed: %s\n", AclErr());
        std::printf("smoke FAILED Init\n");
        return 5;
    }

    int32_t deviceCount = 0;
    aclrtGetDeviceCount(reinterpret_cast<uint32_t*>(&deviceCount));
    std::printf("[smoke] device count: %d\n", deviceCount);
    if (deviceCount <= 0) {
        std::fprintf(stderr, "no NPU devices visible\n");
        aclFinalize();
        std::printf("smoke FAILED NoDevice\n");
        return 6;
    }

    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSetDevice(0) failed: %s\n", AclErr());
        aclFinalize();
        std::printf("smoke FAILED SetDevice\n");
        return 7;
    }

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtCreateStream failed: %s\n", AclErr());
        aclrtResetDevice(0);
        aclFinalize();
        std::printf("smoke FAILED Stream\n");
        return 8;
    }

    // ---- allocate device buffers ----
    // out covers cube ring + linear vec_out tail (kNumTiles slots);
    // kernel.cpp's DeviceParams.out points to its base, the device kernel
    // splits via fixed offsets:
    //   [0 .. kRingSlots)                       cube ring
    //   [kRingSlots .. kRingSlots+kNumTiles)    vec_out, one slot per tile
    const int    kTotalSlots = kRingSlots + kNumTiles;
    const size_t outBytes    = sizeof(float) * kTotalSlots * kTileM * kTileN;
    const size_t slotElems   = static_cast<size_t>(kTileM) * kTileN;
    const size_t vecOutBaseElems = static_cast<size_t>(kRingSlots) * slotElems;

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
        return cleanup(9, "Malloc");
    }

    if (aclrtMemcpy(dAct, actBytes, hAct.data(), actBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(dWgt, wgtBytes, hWgt.data(), wgtBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::fprintf(stderr, "act/wgt H2D failed: %s\n", AclErr());
        return cleanup(10, "Memcpy");
    }
    // Zero `out` so a kernel that writes nothing can be distinguished
    // from a kernel that writes the right value.
    if (aclrtMemset(dOut, outBytes, 0, outBytes) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtMemset failed: %s\n", AclErr());
        return cleanup(11, "Memset");
    }

    // ---- pack params and launch ----
    svdquant::GemmW4A4Params p{};
    p.act.data = dAct;
    p.wgt.data = dWgt;
    p.out.data = dOut;

    std::printf("[smoke] launching gemm_w4a4 (Phase 2e ref-diff, M=%d K=%d N=%d, ring=%d, num_tiles=%d, vec_scale=%.2f)\n",
                kTileM, kTileK, kTileN, kRingSlots, kNumTiles, kVecScale);
    svdquant::ascend::gemm_w4a4(p, stream);

    std::printf("[smoke] aclrtSynchronizeStream\n");
    aclError sync_ret = aclrtSynchronizeStream(stream);
    if (sync_ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSynchronizeStream failed (%d): %s\n",
                     static_cast<int>(sync_ret), AclErr());
        return cleanup(12, "Sync");
    }

    // ---- D2H whole buffer + element-wise diff against ref ----
    std::vector<float> hOut(static_cast<size_t>(kTotalSlots) * slotElems, -1.0f);
    if (aclrtMemcpy(hOut.data(), outBytes, dOut, outBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::fprintf(stderr, "out D2H failed: %s\n", AclErr());
        return cleanup(13, "MemcpyD2H");
    }

    // Every vec_out tile reuses the same act/wgt → must match ref
    // element-wise. Mismatches per tile pinpoint which iteration of
    // the preload/main-loop wave broke.
    int    totalMismatches  = 0;
    int    firstBadTile     = -1;
    int    firstBadIdxInTile = -1;
    float  firstBadVal      = 0.0f;
    float  firstBadRef      = 0.0f;
    float  maxAbsErr        = 0.0f;
    float  maxRelErr        = 0.0f;

    for (int t = 0; t < kNumTiles; ++t) {
        const size_t tile_base = vecOutBaseElems + static_cast<size_t>(t) * slotElems;
        int tileMismatches = 0;
        for (int idx = 0; idx < kTileM * kTileN; ++idx) {
            const float v = hOut[tile_base + idx];
            const float r = hRef[idx];
            const float ae = std::fabs(v - r);
            const float allowed = std::max(kAbsTol, kRelTol * std::fabs(r));
            if (ae > allowed) {
                if (totalMismatches == 0) {
                    firstBadTile      = t;
                    firstBadIdxInTile = idx;
                    firstBadVal       = v;
                    firstBadRef       = r;
                }
                ++tileMismatches;
                ++totalMismatches;
            }
            if (ae > maxAbsErr) maxAbsErr = ae;
            const float denom = std::fabs(r) > 1e-6f ? std::fabs(r) : 1e-6f;
            const float re = ae / denom;
            if (re > maxRelErr) maxRelErr = re;
        }
        std::printf("[smoke] vec_out tile %d/%d: %d mismatches\n",
                    t, kNumTiles, tileMismatches);
    }

    std::printf("[smoke] err summary: max_abs=%.4f max_rel=%.4f (tol abs=%.4f rel=%.4f)\n",
                maxAbsErr, maxRelErr, kAbsTol, kRelTol);

    if (totalMismatches == 0) {
        std::printf("[smoke] vec_out validated: %d tiles x %d x %d = %d elements all match ref within tol\n",
                    kNumTiles, kTileM, kTileN, kNumTiles * kTileM * kTileN);
    } else {
        std::printf("[smoke] vec_out mismatch: %d total; first at tile %d, in-tile idx %d → got %.4f, ref %.4f\n",
                    totalMismatches, firstBadTile, firstBadIdxInTile, firstBadVal, firstBadRef);
        return cleanup(14, "MismatchVecOut");
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
