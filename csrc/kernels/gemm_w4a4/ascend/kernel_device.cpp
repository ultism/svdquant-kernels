// gemm_w4a4 — Ascend __aicore__ device kernel.
//
// Phase 3a: real INT4 cube MMA + per-K-block dequant in vec.
//
// Tile shape (hardcoded for 3a; tile-parameterization is 3b/3c):
//   M = 128, K_logical = 2048, K_packed = 1024, N = 256
//   K-block = 64 logical = 32 packed bytes  (== mad_s4 KS == INT4_BLOCK_SIZE)
//   ⇒ 32 K-blocks per tile, 1 tile per launch.
//
// Buffers:
//   L1 (512 KB):
//     A tile [128, 1024] int8_t = 128 KB at offset 0
//     B tile [1024, 256] int8_t = 256 KB at offset 128 KB
//     Total 384 KB (no L1 ping-pong; whole BK loaded once per launch).
//   L0A / L0B (64 KB each):
//     ping-pong sub-tiles for one K-block (sizes set in
//     pto_macro_matmul_s4.hpp). 4 KB / 8 KB tiles → ample headroom.
//   L0C (256 KB):
//     single int32 [128, 256] = 128 KB at offset 0. No ping-pong —
//     each K-block writes init=true and is drained to GM workspace
//     before the next mad_s4 overwrites L0C.
//   GM workspace (caller-allocated):
//     int32 [kRingSlots=6, 128, 256] cube/vec hand-off ring (768 KB).
//   GM out (caller-allocated): fp16 [128, 256] = 64 KB final.
//
// Cube path (per K-block):
//   wait FIX→M  (prev TSTORE off L0C)         ← skipped on first iter
//   wait M→MTE1 (prev TEXTRACT on this pingpong drained)
//   slide L1 view to kb-th K-block via TASSIGN from saved bases
//   pto_macro_matmul_s4_block: TEXTRACT + mad_s4 (init=true)  → L0C
//   set M→FIX  ;  wait M→FIX (gate TSTORE on mad_s4 done)
//   TSTORE L0C → workspace[slot=kb%kRingSlots]
//   ffts_cross_core_sync(FIX, CUBE_TILE_READY)
//   set FIX→M   (next mad_s4 may overwrite L0C)
//   set M→MTE1  (next iter may reuse this pingpong slot)
//
// Back-pressure: kPreloadNum = kRingSlots = 6 K-blocks fired without
// vec gate (slots empty); from kb >= 6 onwards each iter waits one
// VEC_TILE_CONSUMED before producing. Drain trailing kRingSlots
// VEC_TILE_CONSUMED signals on exit so the FFTS counter ends clean
// (vec signals 32 times total; cube only waited 32-6=26 times in the
// loop).
//
// Vec path (per K-block):
//   wait_flag_dev(CUBE_TILE_READY)
//   TLOAD partial_i32 from workspace[slot] (vecM rows, BN cols)
//   TLOAD ascale fp16 row from ascales[kb, m_off:m_off+vecM]
//   TLOAD wscale fp16 col from wscales[kb, :]
//   TCVT i32→f32 (partial), fp16→f32 (ascale, wscale)
//   TROWEXPANDMUL (apply ascale per row)
//   TCOLEXPANDMUL (apply wscale per col)
//   TADD into running_f32 accumulator (or TMOV if kb==0)
//   ffts_cross_core_sync(MTE2, VEC_TILE_CONSUMED)   ← free ring slot
// After last K-block:
//   TCVT running_f32 → running_f16
//   TSTORE → out_gm[m_off:m_off+vecM, :]
//
// The TASSIGN-on-data() bug in pto_macro_matmul.hpp doesn't affect us
// because we save the L1 base ptr before the K-loop and recompute the
// per-K-block view from base every iter. (Phase 2d hit Cube_K=Tile_K
// so its loop iterated only once and the bug was masked.)
//
// `__enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_2`
// keeps the auto-gen wrapper in mix mode (1 cube : 2 vec).

#include "kernel_operator.h"
#include <pto/pto-inst.hpp>

#include "pto_macro_matmul_s4.hpp"

constexpr KernelMetaType __enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_2;

namespace {

// FFTS flag IDs — same enum as Phase 2a–2d so subsequent phases don't
// have to renumber. CUBE_TILE_READY / VEC_TILE_CONSUMED are now the
// per-K-block producer/consumer signals (each fires 32 times per
// launch, not 8 like Phase 2d).
enum GemmFftsFlag : uint16_t {
    HANDSHAKE_CUBE_TO_VEC = 0,
    HANDSHAKE_VEC_TO_CUBE = 1,
    CUBE_TILE_READY      = 2,
    VEC_TILE_CONSUMED    = 3,
    LORA_BUF_READY       = 4,
};

// Mirrors host-side DeviceParams in kernel.cpp. ccec disallows casting
// `void* __gm__` to a typed `__gm__ T*` inside aicore code, so we read
// these typed pointers from a host-packed struct on entry. Field
// order MUST match the host-side struct exactly.
struct DeviceParams {
    __gm__ uint8_t* act;         // [M, K/2]              packed INT4
    __gm__ uint8_t* wgt;         // [N, K/2]              packed INT4
    __gm__ half*    ascales;     // [K/64, M]             fp16 (K-block, M)
    __gm__ half*    wscales;     // [K/64, N]             fp16 (K-block, N)
    __gm__ half*    la_fp16;     // [M, R]                fp16 (cast from fp32 host-side)
    __gm__ half*    lu_T;        // [R, N]                fp16 (transposed host-side)
    __gm__ int32_t* workspace;   // [kRingSlots, M, N]    int32 cube/vec ring
    __gm__ float*   lora_buf;    // [M, N]                fp32 LoRA-up hand-off
    __gm__ half*    out;         // [M, N]                fp16 final
};

// Tile shape constants — pinned for 3a single-tile. Starting smaller
// (matching Phase 2d's mock shape M=64 K=128 N=128) to localize a
// runtime UB-OOB on AIV; once data path is correct, scale up to the
// production target M=128 K=2048 N=256 in 3a-cont.
constexpr uint32_t kBM         = 64;
constexpr uint32_t kBN         = 128;
constexpr uint32_t kBKLogical  = 128;
constexpr uint32_t kBKPacked   = kBKLogical / 2;
constexpr uint32_t kKSLogical  = 64;                   // mad_s4 K-block / scale block
constexpr uint32_t kKSPacked   = kKSLogical / 2;       // 32 packed bytes
constexpr uint32_t kNumKBlocks = kBKLogical / kKSLogical;  // 32

// LoRA rank (production R ≤ 128). 32 is a real shipping point and keeps
// the LoRA-up cube pass a single mad (kBM × kR × kBN fp16, fp32 acc).
constexpr uint32_t kR          = 32;

// Cube-vec ring. kPreloadNum = kRingSlots so cube fills the ring once
// without back-pressure, then steady-state waits VEC_TILE_CONSUMED.
constexpr uint32_t kRingSlots  = 6;
constexpr uint32_t kPreloadNum = kRingSlots;

// Mix mode 1:2 — 1 cube + 2 vec subblocks per cluster.
constexpr uint16_t kAivPerAic = 2;
constexpr uint32_t kVecM      = kBM / kAivPerAic;      // 64 rows per AIV subblock

}  // namespace

extern "C" __global__ [aicore] void
svdquant_gemm_w4a4_kernel(GM_ADDR params_addr) {
    auto* p = (__gm__ const DeviceParams*)params_addr;

    if ASCEND_IS_AIC {
        auto* act_gm = p->act;
        auto* wgt_gm = p->wgt;
        auto* ws_gm  = p->workspace;

        using TileMatA = pto::Tile<pto::TileType::Mat, int8_t, kBM, kBKPacked,
                                    pto::BLayout::ColMajor, kBM, kBKPacked,
                                    pto::SLayout::RowMajor, 512>;
        using TileMatB = pto::Tile<pto::TileType::Mat, int8_t, kBKPacked, kBN,
                                    pto::BLayout::RowMajor, kBKPacked, kBN,
                                    pto::SLayout::ColMajor, 512>;
        using TileAccC = pto::TileAcc<int32_t, kBM, kBN, kBM, kBN>;

        using GlobalA  = pto::GlobalTensor<int8_t,
            pto::Shape<1, 1, 1, kBM, kBKPacked>,
            pto::Stride<1, 1, 1, kBKPacked, 1>>;
        using GlobalB  = pto::GlobalTensor<int8_t,
            pto::Shape<1, 1, 1, kBKPacked, kBN>,
            pto::Stride<1, 1, 1, 1, kBKPacked>,
            pto::Layout::DN>;
        using GlobalRingSlot = pto::GlobalTensor<int32_t,
            pto::Shape<1, 1, 1, kBM, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;

        // L1 layout: A at offset 0 (128 KB), B at offset 128 KB (256 KB).
        // Total 384 KB / 512 KB.
        constexpr uint64_t kL1AByteOffset = 0;
        constexpr uint64_t kL1BByteOffset = (uint64_t)kBM * kBKPacked;

        TileMatA aMatTile;
        TileMatB bMatTile;
        TileAccC cAccTile;
        TASSIGN(aMatTile, kL1AByteOffset);
        TASSIGN(bMatTile, kL1BByteOffset);
        TASSIGN(cAccTile, 0u);  // L0C single buffer

        GlobalA aGlobal((__gm__ int8_t*)act_gm);
        GlobalB bGlobal((__gm__ int8_t*)wgt_gm);

        // One-shot TLOAD of the full BK into L1.
        TLOAD(aMatTile, aGlobal);
        TLOAD(bMatTile, bGlobal);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        // Save L1 base addresses for K-block sliding. The macro's
        // `aMatTile.data()` will return the most-recently-TASSIGN'd
        // value, which compounds if we rely on it; recompute each
        // iter from a fixed base.
        const uint64_t kL1A_base = (uint64_t)aMatTile.data();
        const uint64_t kL1B_base = (uint64_t)bMatTile.data();

        // Seed cross-pipe flags.
        // - PIPE_M → PIPE_MTE1 (×2 events for L0 ping-pong): so the
        //   first wait_flag inside the loop is satisfied on iter 0.
        // - PIPE_FIX → PIPE_M: so the first mad_s4 may overwrite L0C
        //   without waiting on a non-existent prior TSTORE.
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

        // When the K-block count is smaller than the ring depth, back-
        // pressure in the loop never triggers. The drain loop below
        // must then consume exactly the produced count, not kPreloadNum.
        // (Same idiom as Phase 2d's kActualPreload — without it, kNumKBlocks
        // < kPreloadNum deadlocks waiting on signals vec never sends.)
        constexpr uint32_t kActualPreload =
            (kPreloadNum < kNumKBlocks) ? kPreloadNum : kNumKBlocks;

        for (uint32_t kb = 0; kb < kNumKBlocks; ++kb) {
            const uint64_t pingpong = kb & 1;
            const uint32_t slot     = kb % kRingSlots;

            // Back-pressure: after the ring's filled once, vec must
            // free a slot for each subsequent producer iter.
            if (kb >= kActualPreload) {
                wait_flag_dev(VEC_TILE_CONSUMED);
            }

            // Slide L1 view to the kb-th K-block. Strides:
            //   A: kKSPacked * kBM bytes per K-block (M-fast ColMajor BLayout)
            //   B: kKSPacked * kBN bytes per K-block (K-fast RowMajor BLayout)
            TASSIGN(aMatTile, kL1A_base + (uint64_t)kb * kKSPacked * kBM);
            TASSIGN(bMatTile, kL1B_base + (uint64_t)kb * kKSPacked * kBN);

            // Wait L0C is free (prev TSTORE drained on FIX pipe).
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            // Wait this ping-pong's L0A/B slot is free (prev mad_s4 drained).
            wait_flag(PIPE_M, PIPE_MTE1, (event_t)pingpong);

            pto::pto_macro_matmul_s4_block<kBM, kBN, kKSLogical>(
                aMatTile, bMatTile, cAccTile, pingpong);

            // Gate the FIX-pipe TSTORE on mad_s4 done.
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

            // Drain int32 partial to the ring slot.
            GlobalRingSlot ringSlot(ws_gm + (uint64_t)slot * kBM * kBN);
            TSTORE(ringSlot, cAccTile);

            // Tell vec this K-block is consumable.
            ffts_cross_core_sync(PIPE_FIX, pto::getFFTSMsg(0x2, CUBE_TILE_READY));

            // L0C is again writable for the next mad_s4.
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            // This pingpong slot can be re-extracted into.
            set_flag(PIPE_M, PIPE_MTE1, (event_t)pingpong);
        }

        // Drain trailing per-pingpong ping-pong gates.
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        // Drain trailing FIX→M gate.
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

        // Drain trailing VEC_TILE_CONSUMED signals. Vec fires once per
        // K-block (kNumKBlocks total); cube only consumed
        // max(0, kNumKBlocks - kActualPreload) of them in the loop,
        // leaving kActualPreload pending here. With kNumKBlocks=2 and
        // kActualPreload=2, that's 2 drains — matches what vec sent.
        for (uint32_t i = 0; i < kActualPreload; ++i) {
            wait_flag_dev(VEC_TILE_CONSUMED);
        }

        // ===== LoRA-up cube pass =====
        // Single fp16×fp16 mad: la_fp16 [M, R] × lu_T [R, N] → fp32 acc → lora_buf [M, N].
        // L1 layout: place LA + LU_T after the main A/B occupancy. Main A occupies
        // [0, M*K_packed) = [0, 4 KB); main B occupies [4 KB, 4 KB + K_packed*N) =
        // [4 KB, 12 KB). LoRA LA goes at 16 KB (aligned), LU_T at 20 KB.
        constexpr uint64_t kL1LAOffset  = 16u * 1024;
        constexpr uint64_t kL1LUTOffset = kL1LAOffset + (uint64_t)kBM * kR * sizeof(half);

        // Both tiles use ND2NZ (BLayout=ColMajor + SLayout=RowMajor) because
        // the GM tensors for la_fp16 [M, R] and lu_T [R, N] are both row-major
        // (Layout::ND). The DN2ZN path used by the main-GEMM B side requires a
        // column-major-strided GM layout, which would force a host-side
        // transpose of lora_up before it reaches this kernel — wasteful for a
        // small rank-R hand-off.
        using TileMatLA  = pto::Tile<pto::TileType::Mat, half, kBM, kR,
                                      pto::BLayout::ColMajor, kBM, kR,
                                      pto::SLayout::RowMajor, 512>;
        using TileMatLUT = pto::Tile<pto::TileType::Mat, half, kR, kBN,
                                      pto::BLayout::ColMajor, kR, kBN,
                                      pto::SLayout::RowMajor, 512>;
        using TileAccLora = pto::TileAcc<float, kBM, kBN, kBM, kBN>;
        using LeftTileLora  = pto::TileLeft<half, kBM, kR, kBM, kR>;
        using RightTileLora = pto::TileRight<half, kR, kBN, kR, kBN>;

        using GlobalLA = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, kBM, kR>,
            pto::Stride<1, 1, 1, kR, 1>>;
        using GlobalLUT = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, kR, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;
        using GlobalLoraBuf = pto::GlobalTensor<float,
            pto::Shape<1, 1, 1, kBM, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;

        TileMatLA   laMatTile;
        TileMatLUT  lutMatTile;
        TileAccLora loraAccTile;
        TASSIGN(laMatTile,    kL1LAOffset);
        TASSIGN(lutMatTile,   kL1LUTOffset);
        TASSIGN(loraAccTile,  0u);  // reuse L0C BUF0 (main int32 acc is drained)

        GlobalLA  laGlobal((__gm__ half*)p->la_fp16);
        GlobalLUT lutGlobal((__gm__ half*)p->lu_T);

        TLOAD(laMatTile, laGlobal);
        TLOAD(lutMatTile, lutGlobal);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);

        LeftTileLora  aLoraL0;
        RightTileLora bLoraL0;
        TASSIGN(aLoraL0, 0u);  // L0A BUF0
        TASSIGN(bLoraL0, 0u);  // L0B BUF0

        TEXTRACT(aLoraL0, laMatTile,  0, 0);
        TEXTRACT(bLoraL0, lutMatTile, 0, 0);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);

        TMATMUL(loraAccTile, aLoraL0, bLoraL0);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);

        GlobalLoraBuf loraBufGm(p->lora_buf);
        TSTORE(loraBufGm, loraAccTile);

        // Signal vec that lora_buf is consumable.
        ffts_cross_core_sync(PIPE_FIX,
                              pto::getFFTSMsg(0x2, LORA_BUF_READY));
    }

    if ASCEND_IS_AIV {
        // PTO mix-mode AIV vector mask is NOT in a known reset state at
        // entry — TROWEXPANDMUL/TCOLEXPANDMUL/TRowMin etc. internally set
        // a per-line mask (e.g. `set_vector_mask(0, elementsPerLine)`)
        // and don't always restore it before the next op. Whatever
        // residue was left by a previous ASCEND_IS_AIV invocation, or
        // even by hardware power-on default, is what subsequent
        // TLOAD/TCVT/TADD pick up — which, when wrong, makes those ops
        // address a UB region they shouldn't, manifesting as either
        // VEC ub-out-of-bounds (subErrType:4) or, worse, silently-wrong
        // values that decode as ±inf/NaN if the mask happens to make
        // the load land somewhere accessible. PTO's own TColReduceOps.hpp:31
        // and TRowMin.hpp:94 follow exactly this idiom for the same
        // reason; reference: gitcode.com/cann/pto-isa/issues/218 (a vec
        // OOB with byte-identical signature was solved by zhangjian_hz11
        // adding precisely this two-line reset).
        // Both ops are CCE intrinsics (kernel_operator headers); on
        // dav_c220 they emit a single-instruction state write. Cost is
        // negligible (~2 cycles total).
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
        set_mask_norm();
        set_vector_mask(-1, -1);
#endif

        auto* ws_gm  = p->workspace;
        auto* as_gm  = p->ascales;
        auto* wsl_gm = p->wscales;
        auto* out_gm = p->out;

        const uint32_t subblockid = get_subblockid();
        const uint32_t row_off    = kVecM * subblockid;  // 0 or 64

        // UB layout (per AIV subblock; TOTAL_VEC_LOCAL_SIZE = 184 KB on
        // dav_c220 mix mode):
        //   partial_i32 / partial_f32   shared @ 0           = vecM*BN*4 bytes
        //   running_f32                 after partial        = vecM*BN*4 bytes
        //   ascale_f16/_f32             small                after running
        //   wscale_f16/_f32             small                after ascale
        //   out_f16                     overlap @ 0          (post-loop, partial dead)
        //
        // partial_i32/f32 share offset 0 on purpose: tried disjoint at
        // offset 0x4000 and triggered `VEC instruction error: ub address
        // out of bounds` mid-K-block, identical signature to 3a-fix-3
        // (commit 0334240) which suspected PTO internal scratch
        // (TMP_UB_OFFSET) using a fixed low UB offset. In-place TCVT
        // i32→f32 is safe at the kernel level (PIPE_V is element-wise);
        // the overlap stays.
        // Block size for vbrcb broadcast (32-byte block / sizeof(fp32) = 8).
        // TROWEXPAND([1, M] RowMajor → [M, 8] RowMajor) requires dst::Cols
        // == elemPerBlock = 8 (see pto::TROWEXPAND_IMPL isBroadcast check).
        constexpr uint32_t kBcastCols = 8;

        constexpr uint32_t kPartialOff      = 0;
        constexpr uint32_t kRunningOff      = kPartialOff + kVecM * kBN * 4;
        // ascale = per-row M scale. Loaded RowMajor [1, vecM] half (mirror
        // of wscale's known-working pattern), TCVT to RowMajor [1, vecM]
        // fp32, then expanded by pto::TROWEXPAND to RowMajor [vecM, 8]
        // broadcast tile (row r = [s_r] × 8). The broadcast tile is what
        // TROWEXPANDMUL consumes — feeding it as RowMajor src1 takes the
        // RowMajor code path that skips PTO's internal vbrcb scratch.
        // (3a-cycle-15 root cause: ColMajor [vecM, 1] TLOAD from GM only
        // loads the head element; switching to this load-row + expand
        // pattern bypasses the bug entirely. See memory note
        // pto_colmajor_n1_tload_broken.md.)
        constexpr uint32_t kAscaleF16Off    = kRunningOff + kVecM * kBN * 4;
        constexpr uint32_t kAscaleF32Off    = kAscaleF16Off + kVecM * 2;
        constexpr uint32_t kAscaleBcastOff  = kAscaleF32Off + kVecM * 4;
        constexpr uint32_t kWscaleF16Off    = kAscaleBcastOff + kVecM * kBcastCols * 4;
        constexpr uint32_t kWscaleF32Off    = kWscaleF16Off + kBN * 2;
        constexpr uint32_t kOutF16Off       = kPartialOff;  // overlap with partial post-loop

        using TilePartialI32 = pto::Tile<pto::TileType::Vec, int32_t, kVecM, kBN,
                                          pto::BLayout::RowMajor, kVecM, kBN>;
        using TilePartialF32 = pto::Tile<pto::TileType::Vec, float, kVecM, kBN,
                                          pto::BLayout::RowMajor, kVecM, kBN>;
        using TileRunningF32 = pto::Tile<pto::TileType::Vec, float, kVecM, kBN,
                                          pto::BLayout::RowMajor, kVecM, kBN>;
        using TileOutF16     = pto::Tile<pto::TileType::Vec, half, kVecM, kBN,
                                          pto::BLayout::RowMajor, kVecM, kBN>;

        // ascale RowMajor row tiles (mirror of wscale): 32 contiguous halfs
        // → 32 contiguous fp32s after TCVT. Then TROWEXPAND broadcasts each
        // scalar into a 32-byte block (= 8 fp32) along the row axis,
        // producing the [vecM, 8] tile that TROWEXPANDMUL takes as RowMajor
        // src1 (without invoking internal vbrcb).
        using TileAscaleF16    = pto::Tile<pto::TileType::Vec, half,  1, kVecM,
                                            pto::BLayout::RowMajor, 1, kVecM>;
        using TileAscaleF32    = pto::Tile<pto::TileType::Vec, float, 1, kVecM,
                                            pto::BLayout::RowMajor, 1, kVecM>;
        using TileAscaleBcastF32 = pto::Tile<pto::TileType::Vec, float, kVecM, kBcastCols,
                                              pto::BLayout::RowMajor, kVecM, kBcastCols>;
        // wscale = per-col N scale → RowMajor [1, BN] (TCOLEXPANDMUL src1).
        using TileWscaleF16 = pto::Tile<pto::TileType::Vec, half, 1, kBN,
                                         pto::BLayout::RowMajor, 1, kBN>;
        using TileWscaleF32 = pto::Tile<pto::TileType::Vec, float, 1, kBN,
                                         pto::BLayout::RowMajor, 1, kBN>;

        using GlobalRingSlot = pto::GlobalTensor<int32_t,
            pto::Shape<1, 1, 1, kVecM, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;
        // Standard RowMajor [1, vecM] half GM strip (same shape pattern as
        // wscale's GlobalWscaleRow). The previous ColMajor [vecM, 1] +
        // Layout::DN setup silently TLOAD'd only the head half — see the
        // memory note pto_colmajor_n1_tload_broken.md.
        using GlobalAscaleRow = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, 1, kVecM>,
            pto::Stride<1, 1, 1, kVecM, 1>>;
        using GlobalWscaleRow = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, 1, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;
        using GlobalOutTile  = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, kVecM, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;

        // Cross-iter UB sync — running_f32 reused across K-blocks.
        // Seed MTE3→MTE2 so first iter's TLOAD doesn't race a non-
        // existent prior TSTORE (drained at the end).
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        // Cross-iter V→MTE2: partial_i32/f32 UB region is reused every
        // iter; without this sync, iter N+1's TLOAD can race iter N's
        // PIPE_V writes (TROWEXPANDMUL/TCOLEXPANDMUL/TMOV land after
        // the load, corrupting the freshly-loaded int32 bytes). See
        // docs/gotchas/ascend.md "AIV K-loop reusing partial UB".
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);

        for (uint32_t kb = 0; kb < kNumKBlocks; ++kb) {
            wait_flag_dev(CUBE_TILE_READY);
            // Wait prior iter's PIPE_V (TROWEXPANDMUL/TCOLEXPANDMUL/TMOV)
            // to drain before this iter's TLOAD overwrites partial UB.
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            const uint32_t slot = kb % kRingSlots;

            // GM offsets:
            //   ring slot row band (this AIV subblock owns rows
            //     [row_off, row_off + kVecM)) into workspace[slot]
            //   ascales[kb, row_off:row_off+kVecM]
            //   wscales[kb, :]
            //   out[row_off:row_off+kVecM, :]
            const uint64_t partial_off =
                (uint64_t)slot * kBM * kBN + (uint64_t)row_off * kBN;
            const uint64_t ascale_off  = (uint64_t)kb * kBM + row_off;
            const uint64_t wscale_off  = (uint64_t)kb * kBN;
            const uint64_t out_off     = (uint64_t)row_off * kBN;

            TilePartialI32      partI32;
            TilePartialF32      partF32;
            TileRunningF32      running;
            TileAscaleF16       ascaleF16;
            TileAscaleF32       ascaleF32;
            TileAscaleBcastF32  ascaleBcast;
            TileWscaleF16       wscaleF16;
            TileWscaleF32       wscaleF32;
            TASSIGN(partI32,     kPartialOff);
            TASSIGN(partF32,     kPartialOff);  // in-place i32→f32 cast (see UB layout note above)
            TASSIGN(running,     kRunningOff);
            TASSIGN(ascaleF16,   kAscaleF16Off);
            TASSIGN(ascaleF32,   kAscaleF32Off);
            TASSIGN(ascaleBcast, kAscaleBcastOff);
            TASSIGN(wscaleF16,   kWscaleF16Off);
            TASSIGN(wscaleF32,   kWscaleF32Off);

            GlobalRingSlot  partGm  (p->workspace + partial_off);
            GlobalAscaleRow ascaleGm(p->ascales   + ascale_off);
            GlobalWscaleRow wscaleGm(p->wscales   + wscale_off);

            // Wait running_f32 region is free (prev TSTORE on out
            // tile finished, except on the first iter where this
            // is the seed).
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

            TLOAD(partI32,   partGm);
            TLOAD(ascaleF16, ascaleGm);
            TLOAD(wscaleF16, wscaleGm);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // Cast i32 → f32, fp16 → f32.
            pto::TCVT(partF32,   partI32,   pto::RoundMode::CAST_RINT);
            pto::TCVT(ascaleF32, ascaleF16, pto::RoundMode::CAST_RINT);
            pto::TCVT(wscaleF32, wscaleF16, pto::RoundMode::CAST_RINT);

            // Expand ascaleF32 [1, vecM] RowMajor → ascaleBcast [vecM, 8]
            // RowMajor where row r = [s_r] × 8. PTO's TROWEXPAND internally
            // calls vbrcb on the [1, M] flat row (which is well-defined
            // because the row tile was loaded via the canonical RowMajor
            // GM → RowMajor UB path). The resulting [vecM, 8] tile is then
            // a valid RowMajor src1 for TROWEXPANDMUL: its assertion
            // `RowMajor src1 && src1ValidCol == 32/sizeof(T) = 8` holds,
            // and the RowMajor code path skips the internal vbrcb scratch
            // entirely.
            pto::TROWEXPAND(ascaleBcast, ascaleF32);

            // Defensive: TROWEXPAND's internal vbrcb may leave the mask
            // register in a non-default state; TROWEXPANDMUL's RowMajor
            // path's NormModeTail else-branch does NOT call SetContMask
            // before vmul (it inherits the caller's mask). Cycle 16 Run H
            // showed first 4 rows per AIV silently skipped — symptom
            // consistent with stale mask. Reset to norm + full vec mask
            // before TROWEXPANDMUL to make the mask state explicit.
            pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask(-1, -1);

            // partF32[m,n] *= ascaleF32[m]  (via pre-broadcast ascaleBcast)
            pto::TROWEXPANDMUL(partF32, partF32, ascaleBcast);

            // partF32[m,n] *= wscaleF32[n]
            pto::TCOLEXPANDMUL(partF32, partF32, wscaleF32);

            // Accumulate into running_f32. On kb==0 there's no
            // prior value, so initialize via TMOV; subsequent iters
            // TADD.
            if (kb == 0) {
                pto::TMOV(running, partF32);
            } else {
                pto::TADD(running, running, partF32);
            }

            // Signal V→MTE2 for the next iter so its TLOAD won't overlap
            // with this iter's PIPE_V writes to partF32. See seed comment.
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);

            // Free the ring slot for the next cube K-block.
            ffts_cross_core_sync(PIPE_MTE2,
                                  pto::getFFTSMsg(0x2, VEC_TILE_CONSUMED));

            // Re-seed MTE3→MTE2 for the next iter's TLOAD region
            // dependency. (running and out_f16 occupy disjoint UB
            // regions, so this is mostly a tail flag bookkeeping
            // action — but consistent with Phase 2d's pattern.)
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }

        // Drain trailing seed before LoRA-add + final cast+store.
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);

        // ===== LoRA-up residual: running += lora_buf[row_off:row_off+vecM, :] =====
        // Reuse kPartialOff for the LoRA tile UB region — partial_i32/f32 is dead
        // after the K-loop (last iter's TADD has finished into running). loraTile
        // is also dead after the TADD below, so it cleanly aliases outF16's UB
        // region for the final TCVT.
        using TileLoraF32 = pto::Tile<pto::TileType::Vec, float, kVecM, kBN,
                                      pto::BLayout::RowMajor, kVecM, kBN>;
        using GlobalLoraSlice = pto::GlobalTensor<float,
            pto::Shape<1, 1, 1, kVecM, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;

        wait_flag_dev(LORA_BUF_READY);

        TileLoraF32 loraTile;
        TASSIGN(loraTile, kPartialOff);

        const uint64_t lora_off = (uint64_t)row_off * kBN;
        auto* lora_buf_gm = (__gm__ float*)p->lora_buf;
        GlobalLoraSlice loraGm(lora_buf_gm + lora_off);
        TLOAD(loraTile, loraGm);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

        TileRunningF32 runningForAdd;
        TASSIGN(runningForAdd, kRunningOff);
        pto::TADD(runningForAdd, runningForAdd, loraTile);

        // Final epilogue: f32 → fp16 then TSTORE the AIV's row band.
        TileRunningF32 runningFinal;
        TileOutF16     outF16Final;
        TASSIGN(runningFinal, kRunningOff);
        TASSIGN(outF16Final,  kOutF16Off);

        pto::TCVT(outF16Final, runningFinal, pto::RoundMode::CAST_RINT);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        const uint64_t out_off = (uint64_t)row_off * kBN;
        GlobalOutTile outGm(out_gm + out_off);
        TSTORE(outGm, outF16Final);
    }
}
