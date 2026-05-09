// gemm_w4a4 — Ascend __aicore__ device kernel.
//
// Phase 2d (current): preload + main loop double-stage cube driver,
// 6-slot ring buffer with cube ↔ vec back-pressure, num_tiles tiles
// per launch.
//
//   - AIC: kPreloadNum = kRingSlots tiles fired into ring slots 0..N-1
//     unconditionally (preload — no wait on vec because the slots
//     start empty). Then main loop: for each remaining tile, wait
//     VEC_TILE_CONSUMED (= "vec freed up at least one slot"), produce
//     into slot = tile_idx % kRingSlots, signal CUBE_TILE_READY.
//   - AIV: single loop — wait CUBE_TILE_READY, consume slot =
//     tile_idx % kRingSlots, TSTORE to vec_out[tile_idx], signal
//     VEC_TILE_CONSUMED.
//
// Why double-stage on cube and not single-stage with counter-prefill:
// FA pre-increments BUF1_SV_CONSUMED kRingSlots times via st_dev to
// represent "ring starts empty"; that needs the FFTS base address as
// an explicit kernel arg. Our auto-gen wrapper already calls
// set_ffts_base_addr internally and does NOT pass the ffts_addr GM
// pointer to the user kernel signature, so st_dev is unavailable. The
// FA double-stage form is the equivalent idiom that doesn't need
// ffts_addr — Phase A produces while the ring is known-empty,
// Phase B does the steady-state wait/produce dance. End result is
// the same producer/consumer wave shape.
//
// Cross-iter UB sync on the vec side: each iter does TLOAD →
// TROWEXPANDMUL/TADD → TSTORE on the same UB region (runningOTile).
// Need MTE3 → MTE2 sync between iters so the next TLOAD doesn't
// overwrite UB while the prev TSTORE is still reading it. We seed
// one set_flag(PIPE_MTE3, PIPE_MTE2) at vec entry, wait+set inside
// the loop, drain on exit.
//
// Cube ping-pong (PIPE_M → PIPE_MTE1) is per-tile self-contained:
// seed at iter top, drain at iter bottom. The flags are private to
// pto_macro_matmul's K-loop, not shared across tiles.
//
// Tile sizes are still mock (64×128×128) — Phase 3 will pick the
// real BM/BN/BK after we drop in the s4 path. Phase 2e adds a
// PyTorch reference and the smoke compares element-wise.
//
// `__enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_2`
// keeps the auto-gen wrapper in mix mode (1 cube : 2 vec) — see
// Phase 2a comment block in git log for the rationale.

#include "kernel_operator.h"
#include <pto/pto-inst.hpp>

#include "pto_macro_matmul.hpp"

constexpr KernelMetaType __enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_2;

namespace {

// FFTS flag IDs — same enum as Phase 2a so subsequent phases don't
// have to renumber. CUBE_TILE_READY / VEC_TILE_CONSUMED are the
// producer/consumer signals for the ring; HANDSHAKE_* slots are
// leftovers from Phase 2a, kept for future rerouting.
enum GemmFftsFlag : uint16_t {
    HANDSHAKE_CUBE_TO_VEC = 0,
    HANDSHAKE_VEC_TO_CUBE = 1,
    CUBE_TILE_READY      = 2,
    VEC_TILE_CONSUMED    = 3,
};

// Device-side params layout. ccec disallows casting `void* __gm__` to
// a typed `__gm__ T*` inside aicore code, so we can't read the public
// `svdquant::GemmW4A4Params` directly here. The host launcher packs
// typed pointers into this byte-compatible struct (24 B = 3 × 8 B
// pointers) and H2D-copies it to dev_params. Keep host (kernel.cpp)
// and device sides in sync.
struct DeviceParams {
    __gm__ half*  act;   // [Tile_M, Tile_K] half row-major
    __gm__ half*  wgt;   // [Tile_N, Tile_K] half row-major (NT-viewed)
    __gm__ float* out;   // [(kRingSlots + kNumTiles), Tile_M, Tile_N] fp32:
                         //   [0 .. kRingSlots)        cube ring
                         //   [kRingSlots .. +kNumTiles)
                         //                            vec_out (linear,
                         //                            one slot per tile,
                         //                            0..kNumTiles-1).
};

// Phase 2c-onwards mock dequant scale; baked into the kernel via
// TEXPANDS. Phase 3 derives the real per-row scale in-tile from
// ascales × wscales and TEXPANDS retires.
constexpr float kVecScale = 0.5f;

// Mock-stage cube tile shape. Phase 2d still uses 64×128×128 — the
// goal here is the ring + back-pressure plumbing, not throughput.
// Phase 3 picks the final BM/BN/BK once s4 lands.
constexpr uint32_t kTileM = 64;
constexpr uint32_t kTileK = 128;
constexpr uint32_t kTileN = 128;

// Ring buffer slot count and the number of tiles produced per launch.
// kPreloadNum = kRingSlots → cube can fill the entire ring once
// without back-pressure (the simplest two-stage form). With
// kNumTiles = 8 and kRingSlots = 6, the main-loop stage runs for
// kNumTiles - kRingSlots = 2 iterations, each gated by a
// VEC_TILE_CONSUMED signal — enough to actually exercise slot reuse
// (slots 0 and 1 get re-written by tiles 6 and 7).
constexpr uint32_t kRingSlots  = 6;
constexpr uint32_t kNumTiles   = 8;
constexpr uint32_t kPreloadNum = kRingSlots;

// Mix mode 1:2 — 1 cube + 2 vec subblocks per cluster.
constexpr uint16_t kAivPerAic = 2;

}  // namespace

extern "C" __global__ [aicore] void
svdquant_gemm_w4a4_kernel(GM_ADDR params_addr) {
    auto* p = (__gm__ const DeviceParams*)params_addr;

    if ASCEND_IS_AIC {
        auto* act_gm = p->act;
        auto* wgt_gm = p->wgt;
        auto* out_gm = p->out;

        // Tile + GM types are independent of tile_idx; declare once.
        // L1: A row-major bulk + row-major sub-fractal; B row-major
        // bulk + col-major sub-fractal — pto_macro_matmul deduces NT
        // (C = A · B^T) from this combination.
        using TileMatA = pto::Tile<pto::TileType::Mat, half, kTileM, kTileK,
                                    pto::BLayout::ColMajor, kTileM, kTileK,
                                    pto::SLayout::RowMajor, 512>;
        using TileMatB = pto::Tile<pto::TileType::Mat, half, kTileK, kTileN,
                                    pto::BLayout::RowMajor, kTileK, kTileN,
                                    pto::SLayout::ColMajor, 512>;
        using TileAccC = pto::TileAcc<float, kTileM, kTileN, kTileM, kTileN>;
        using GlobalA  = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, kTileM, kTileK>,
            pto::Stride<1, 1, 1, kTileK, 1>>;
        using GlobalB  = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, kTileK, kTileN>,
            pto::Stride<1, 1, 1, 1, kTileK>,
            pto::Layout::DN>;
        using GlobalOut = pto::GlobalTensor<float,
            pto::Shape<1, 1, 1, kTileM, kTileN>,
            pto::Stride<1, 1, 1, kTileN, 1>>;

        constexpr uint32_t kAByteOffset = 0;
        constexpr uint32_t kBByteOffset = kTileM * kTileK * 2;  // sizeof(half)

        // Single fused preload + main loop. Conceptually two stages:
        //   Phase A (i < kPreloadNum) — produce without back-pressure;
        //                               ring slots are known empty.
        //   Phase B (i >= kPreloadNum) — wait for vec to free a slot
        //                                via VEC_TILE_CONSUMED, then
        //                                produce.
        // Fused into one loop because ccec doesn't tag lambda bodies
        // [aicore] (PTO intrinsics rejected with "call to [aicore]
        // function from [host] function") so a lambda extraction breaks
        // the build. The `if (i >= kPreloadNum)` guard makes the
        // double-stage shape explicit at the source level.
        constexpr uint32_t kActualPreload =
            (kPreloadNum < kNumTiles) ? kPreloadNum : kNumTiles;

        for (uint32_t i = 0; i < kNumTiles; ++i) {
            if (i >= kActualPreload) {
                // Phase B back-pressure gate — only kicks in after the
                // ring's been filled once.
                wait_flag_dev(VEC_TILE_CONSUMED);
            }
            const uint32_t slot = i % kRingSlots;

            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

            TileMatA aMatTile;
            TileMatB bMatTile;
            TileAccC cAccTile;
            TASSIGN(aMatTile, kAByteOffset);
            TASSIGN(bMatTile, kBByteOffset);
            TASSIGN(cAccTile, 0u);

            GlobalA aGlobal(act_gm);
            GlobalB bGlobal(wgt_gm);
            GlobalOut cGlobal(out_gm + slot * kTileM * kTileN);

            TLOAD(aMatTile, aGlobal);
            TLOAD(bMatTile, bGlobal);

            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

            pto::pto_macro_matmul<kTileM, kTileK, kTileN>(aMatTile, bMatTile, cAccTile);

            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

            TSTORE(cGlobal, cAccTile);

            // mode 0x2 = subblock-broadcast; one cube signal unblocks
            // both vec subblocks in this cluster.
            ffts_cross_core_sync(PIPE_FIX, pto::getFFTSMsg(0x2, CUBE_TILE_READY));

            // Drain ping-pong flags so next iter can re-seed cleanly.
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        }

        // Drain the trailing VEC_TILE_CONSUMED signals so the FFTS
        // counter exits clean. Vec signals kNumTiles times total; the
        // loop above only waited (kNumTiles - kActualPreload) times,
        // leaving kActualPreload extra signals to absorb.
        for (uint32_t i = 0; i < kActualPreload; ++i) {
            wait_flag_dev(VEC_TILE_CONSUMED);
        }
    }

    if ASCEND_IS_AIV {
        constexpr uint32_t kVecM = kTileM / kAivPerAic;
        const uint32_t subblockid = get_subblockid();

        auto* out_gm = p->out;

        // out_gm layout (elements):
        //   [0 .. kRingSlots * M * N)         cube ring
        //   [kRingSlots * M * N .. above + kNumTiles * M * N)
        //                                     vec_out (linear by tile_idx)
        constexpr uint32_t kVecOutBaseElems = kRingSlots * kTileM * kTileN;
        const uint32_t row_off_elems = kVecM * kTileN * subblockid;

        using TileVecF = pto::Tile<pto::TileType::Vec, float, kVecM, kTileN,
                                    pto::BLayout::RowMajor, kVecM, kTileN>;
        using TileReduceF = pto::Tile<pto::TileType::Vec, float, kVecM, 1,
                                       pto::BLayout::ColMajor, kVecM, 1>;
        using GlobalSrc = pto::GlobalTensor<float,
            pto::Shape<1, 1, 1, kVecM, kTileN>,
            pto::Stride<1, 1, 1, kTileN, 1>>;

        // UB layout (private per subblock):
        //   [0 .. 16 KB)              runningOTile [Vec_M, Tile_N] fp32
        //   [16 KB .. 32 KB)          estTile      [Vec_M, Tile_N] fp32
        //   [32 KB .. 32 KB + 128 B)  scaleTile    [Vec_M, 1]      fp32
        constexpr uint32_t kRunningOff = 0;
        constexpr uint32_t kEstOff     = kVecM * kTileN * 4;
        constexpr uint32_t kScaleOff   = kEstOff + kVecM * kTileN * 4;

        // Seed MTE3 → MTE2 sync — the first iter has no prev TSTORE to
        // wait on, but the wait_flag inside the loop must see at least
        // one matched set_flag.
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        for (uint32_t tile_idx = 0; tile_idx < kNumTiles; ++tile_idx) {
            wait_flag_dev(CUBE_TILE_READY);

            const uint32_t slot       = tile_idx % kRingSlots;
            const uint32_t cube_off   = slot * kTileM * kTileN + row_off_elems;
            const uint32_t vec_off    = kVecOutBaseElems + tile_idx * kTileM * kTileN
                                        + row_off_elems;

            TileVecF runningOTile;
            TileVecF estTile;
            TileReduceF scaleTile;
            TASSIGN(runningOTile, kRunningOff);
            TASSIGN(estTile,      kEstOff);
            TASSIGN(scaleTile,    kScaleOff);

            GlobalSrc cubeSrc(out_gm + cube_off);
            GlobalSrc vecOutGlobal(out_gm + vec_off);

            // Wait for prev iter's TSTORE to release the UB region
            // before the next TLOAD overwrites it.
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

            TLOAD(runningOTile, cubeSrc);
            TLOAD(estTile,      cubeSrc);  // mock: same ring slot reloaded as residual

            // Constant scale on V pipe — see Phase 2c notes for why
            // TEXPANDS replaces a GM scale buffer.
            pto::TEXPANDS(scaleTile, kVecScale);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            pto::TROWEXPANDMUL(runningOTile, runningOTile, scaleTile);
            pto::TADD(runningOTile, runningOTile, estTile);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            TSTORE(vecOutGlobal, runningOTile);

            // Free up the ring slot. Cube's main-loop wait_flag_dev
            // matches this. PIPE_MTE2 = "after the most recent MTE2
            // ops" — we don't strictly need the sync semantic here
            // since TSTORE is on MTE3 not MTE2, but FA's compute_gu
            // uses PIPE_MTE2 here too and it's consistent across the
            // codebase.
            ffts_cross_core_sync(PIPE_MTE2, pto::getFFTSMsg(0x2, VEC_TILE_CONSUMED));

            // Set up MTE3 → MTE2 for next iter's TLOAD.
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }

        // Drain the trailing seed.
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}
