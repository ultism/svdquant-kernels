// gemm_w4a4 — Ascend __aicore__ device kernel.
//
// Phase 2b (current): cube fp16 mock GEMM into a 6-slot GM ring.
//   - AIC reads `act` ([Tile_M, Tile_K] half row-major) and `wgt`
//     ([Tile_N, Tile_K] half row-major, viewed as [Tile_K, Tile_N]
//     col-major for the NT-layout cube macro), runs the K-loop via
//     pto_macro_matmul (fp16 inputs → fp32 acc in L0C), TSTOREs the
//     fp32 result to slot 0 of `out` ([Tile_M, Tile_N] fp32 → backed
//     by an over-allocated [kRingSlots, Tile_M, Tile_N] buffer in
//     `params.out` so Phase 2d can rotate slots without reallocating).
//   - AIV exits silently. Phase 2c will turn it into a TROWEXPANDMUL +
//     TADD consumer that rescales each cube tile by a constant; for
//     now the validation strategy is "host reads slot 0 directly and
//     compares to `act @ wgt.T`".
//
// Phase 2a's cube↔vec FFTS handshake is intentionally removed at this
// stage: the cube does not need to notify the vec yet, and forcing
// the vec to wait for a non-existent producer signal would deadlock
// once the AIV body is filled in. The MIX_AIC_1_2 plumbing established
// in Phase 2a (auto-injected ffts_addr + set_ffts_base_addr in the
// auto-gen wrapper) stays — we'll rebuild on it in Phase 2c.
//
// Tile sizes are intentionally small (64×128×128) — the goal is to
// get cube K-loop + L0 ping-pong + FIX-pipe TSTORE plumbing right,
// not to push throughput. Phase 2d will scale up alongside the
// preload pipeline, and Phase 3 picks the final BM/BN/BK for s4.
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
// producer/consumer signals for the 6-slot ring; HANDSHAKE_* slots
// are leftovers from Phase 2a, kept for future rerouting.
enum GemmFftsFlag : uint16_t {
    HANDSHAKE_CUBE_TO_VEC = 0,
    HANDSHAKE_VEC_TO_CUBE = 1,
    CUBE_TILE_READY      = 2,
    VEC_TILE_CONSUMED    = 3,
};

// Device-side params layout. ccec disallows casting `void* __gm__` to
// a typed `__gm__ T*` inside aicore code, so we can't read the public
// `svdquant::GemmW4A4Params` (which holds `void*` inside TensorRef)
// directly here. Instead, the host launcher packs typed pointers into
// this byte-compatible struct (24 B = 3 × 8 B pointers) and H2D-copies
// it to dev_params. This file owns the device-side definition with
// `__gm__` qualifiers; `kernel.cpp` (host) packs the same byte layout
// using plain `void*`. Phase 2c will grow the struct as the vec
// consumer needs more pointers.
struct DeviceParams {
    __gm__ half*  act;   // [Tile_M, Tile_K] half row-major
    __gm__ half*  wgt;   // [Tile_N, Tile_K] half row-major (NT-viewed)
    __gm__ float* out;   // [kRingSlots, Tile_M, Tile_N] fp32 ring
};

// Mock-stage cube tile shape. Phase 2b is one iteration: AIC produces
// one [Tile_M, Tile_N] tile and writes it to slot 0 of the ring.
// Tile_K must be ≥ MEM_BUFFER_SIZE_BYTES / (Tile_M * 2) for the
// pto_macro_matmul calculateFittingCubeK heuristic to pick a Cube_K
// large enough that the K-loop has at least 2 iterations (so L0
// ping-pong actually toggles), but small enough that A/B fit in the
// 32KB L0A/L0B halves. 64 × 128 × 128 satisfies both: Cube_K=128
// → 1 K-loop iteration; for ping-pong demonstration we'd want
// Cube_K=64 → 2 iterations, which fits 64*64*2 = 8KB ≤ 32KB.
constexpr uint32_t kTileM = 64;
constexpr uint32_t kTileK = 128;
constexpr uint32_t kTileN = 128;

// Ring buffer slot count (matches Phase 2 plan in PLAN.md). Phase 2b
// only writes slot 0; Phase 2d will rotate.
constexpr uint32_t kRingSlots = 6;

// Mix mode 1:2 — 1 cube + 2 vec subblocks per cluster. Carry-over
// from Phase 2a; not exercised at Phase 2b but kept named for
// readability.
constexpr uint16_t kAivPerAic = 2;

}  // namespace

extern "C" __global__ [aicore] void
svdquant_gemm_w4a4_kernel(GM_ADDR params_addr) {
    // The auto-gen wrapper has already called set_ffts_base_addr for
    // us (mix-mode plumbing), so cube-side flags can be issued
    // immediately. Cube needs two pingpong flags seeded for the L0
    // K-loop wait_flag(PIPE_M, PIPE_MTE1, 0/1) inside pto_macro_matmul.
    if ASCEND_IS_AIC {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        // Read the device-side params struct from GM. Field types
        // are already typed `__gm__ T*` so no void→T cast is needed
        // — ccec rejects that cast inside aicore code, which is why
        // we don't read the public GemmW4A4Params (TensorRef::data
        // is `void*`) directly here. The host launcher packs this
        // exact 24-byte layout from the public params before H2D.
        auto* p = (__gm__ const DeviceParams*)params_addr;
        auto* act_gm = p->act;
        auto* wgt_gm = p->wgt;
        auto* out_gm = p->out;

        // L1 tile types. Mirror FA's QK pattern (`tfa_kernel.cpp:
        // 525-528`): A is row-major bulk + row-major sub-fractal,
        // B is row-major bulk + col-major sub-fractal, which the
        // pto_macro_matmul deduce_layout decodes as NT (i.e. C =
        // A · B^T). 512 is the L1 alignment / scratch byte count.
        using TileMatA = pto::Tile<pto::TileType::Mat, half, kTileM, kTileK,
                                    pto::BLayout::ColMajor, kTileM, kTileK,
                                    pto::SLayout::RowMajor, 512>;
        using TileMatB = pto::Tile<pto::TileType::Mat, half, kTileK, kTileN,
                                    pto::BLayout::RowMajor, kTileK, kTileN,
                                    pto::SLayout::ColMajor, 512>;
        using TileAccC = pto::TileAcc<float, kTileM, kTileN, kTileM, kTileN>;

        TileMatA aMatTile;
        TileMatB bMatTile;
        TileAccC cAccTile;

        // L1 layout for Phase 2b: A at offset 0, B right after.
        // L1 budget is 512 KB; A occupies 64*128*2 = 16 KB, B
        // occupies 128*128*2 = 32 KB → 48 KB total, plenty of slack.
        constexpr uint32_t kAByteOffset = 0;
        constexpr uint32_t kBByteOffset = kTileM * kTileK * 2;  // sizeof(half)
        TASSIGN(aMatTile, kAByteOffset);
        TASSIGN(bMatTile, kBByteOffset);
        TASSIGN(cAccTile, 0u);  // L0C base 0x0

        // GM descriptors. Mirror FA's compute_qk: act is contiguous
        // row-major [M, K], wgt is row-major [N, K] viewed as col-
        // major [K, N] via Layout::DN with stride pattern (1, K).
        using GlobalA = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, kTileM, kTileK>,
            pto::Stride<1, 1, 1, kTileK, 1>>;
        using GlobalB = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, kTileK, kTileN>,
            pto::Stride<1, 1, 1, 1, kTileK>,
            pto::Layout::DN>;
        GlobalA aGlobal(act_gm);
        GlobalB bGlobal(wgt_gm);

        // GM → L1. EVENT_ID0 owns the L1 read/write barrier; we
        // don't double-buffer at Phase 2b so a single event suffices.
        TLOAD(aMatTile, aGlobal);
        TLOAD(bMatTile, bGlobal);

        // MTE2 (data-load done) → MTE1 (L0 extract): the K-loop
        // inside pto_macro_matmul issues TEXTRACT, which reads from
        // L1; that read must observe the just-completed TLOADs.
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        pto::pto_macro_matmul<kTileM, kTileK, kTileN>(aMatTile, bMatTile, cAccTile);

        // M (matmul-pipe done) → FIX (FIX-pipe TSTORE): TSTORE on the
        // accumulator must observe the final TMATMUL_ACC retire.
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        // TSTORE the fp32 acc to slot 0 of the ring. `out_gm` is the
        // base of the [kRingSlots, Tile_M, Tile_N] buffer; offset 0
        // = slot 0. Phase 2d will rotate by `(slot_idx % kRingSlots)
        // * kTileM * kTileN`.
        using GlobalOut = pto::GlobalTensor<float,
            pto::Shape<1, 1, 1, kTileM, kTileN>,
            pto::Stride<1, 1, 1, kTileN, 1>>;
        GlobalOut cGlobal(out_gm);
        TSTORE(cGlobal, cAccTile);

        // Drain the L0 K-loop pingpong flags so the kernel exits
        // clean — the K-loop did one set_flag(PIPE_M, PIPE_MTE1, *)
        // per iteration, the seed loop above did two; balance.
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }

    // AIV branch is intentionally empty at Phase 2b — the validator
    // reads slot 0 of the ring back to host directly. Phase 2c will
    // turn this into a TROWEXPANDMUL + TADD consumer.
    if ASCEND_IS_AIV {
        (void)params_addr;
        (void)kAivPerAic;
    }
}
