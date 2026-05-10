// gemm_w4a4 — INT4 cube K-block helper for Phase 3a.
//
// Encapsulates one K-block of signed-INT4 cube MMA: TEXTRACT a sub-
// tile from L1 into L0A/L0B, then issue a raw `mad_s4` on the L0
// fragments into a single L0C int32 accumulator.
//
// The caller owns the K-loop, the L1 base sliding, the FIX-pipe drain
// of the L0C int32 partial to GM, and the cube↔vec ring signals —
// because each K-block produces a *separate* int32 partial that vec
// dequants with the matching ascales[kb] / wscales[kb] (per-K-block
// scale → cannot accumulate across K-blocks in L0C). This helper
// just hides the per-block L0 setup boilerplate.
//
// Why not the fp16 `pto_macro_matmul.hpp` shape (K-loop inside,
// accumulating in L0C):
//   * fp16 path can accumulate freely in L0C because there's no per-
//     block scale to apply between iters. INT4 dequant requires the
//     int32 partial leave L0C every K-block (or the per-block scale
//     can no longer be applied separately).
//   * `mad_s4` takes 10 args; encoding "init vs accumulate" inside a
//     K-loop macro is awkward and would never be exercised because
//     init=true on every block is the only valid mode for our flow.
//
// Storage type is `int8_t` (not `uint8_t`) for both A/B fragments
// because the PTO TExtract.hpp:477-480 whitelist accepts int8_t for
// the L0 sub-tile dst. The byte pattern is identical and `mad_s4`
// reads via `void*`, so there's no semantic cost to picking either.
// See memory note `ascend_int4_addressability.md`.

#ifndef SVDQUANT_GEMM_W4A4_PTO_MACRO_MATMUL_S4_HPP
#define SVDQUANT_GEMM_W4A4_PTO_MACRO_MATMUL_S4_HPP

#include <pto/pto-inst.hpp>

namespace pto {

// L0 ping-pong byte offsets for INT4 cube tiles. The buffer size only
// has to be ≥ BM × KS_packed (A side) and ≥ KS_packed × BN (B side);
// L0A and L0B are 64 KB each, so we can give every ping-pong slot
// 8 KB (= 0x2000) of room and still fit 8 buffers — comfortably
// covering both the 3a starter tile (BM=64, BN=128) and the 3a-cont
// production tile (BM=128, BN=256). Different tile shapes can still
// override by re-#define'ing before include.
#ifndef SVDQ_S4_L0A_BYTES_PER_BUF
#define SVDQ_S4_L0A_BYTES_PER_BUF 0x2000  // 8 KB
#endif
#ifndef SVDQ_S4_L0B_BYTES_PER_BUF
#define SVDQ_S4_L0B_BYTES_PER_BUF 0x2000  // 8 KB
#endif

// Issue one K-block of INT4 cube MMA.
//
// `aMatTile` and `bMatTile` are L1 references that have already been
// TASSIGN'd to the current K-block's byte offset by the caller. The
// helper TEXTRACTs into per-pingpong L0A/L0B fragments, syncs MTE1→M,
// and runs the raw `mad_s4` intrinsic with `cmatrixInitVal=true`
// (overwrite L0C). After return, L0C holds the int32 partial; caller
// must drain it via FIX-pipe TSTORE before issuing the next K-block.
//
// Pre-conditions (caller's responsibility):
//   * `wait_flag(PIPE_M, PIPE_MTE1, pingpong)` already pending so this
//     ping-pong slot is safe to overwrite.
//   * `wait_flag(PIPE_FIX, PIPE_M, ...)` already issued if a previous
//     iter's TSTORE was still draining L0C.
//
// Post-conditions (caller does these next):
//   * `set_flag(PIPE_M, PIPE_FIX, ...)` to gate the TSTORE.
//   * `set_flag(PIPE_M, PIPE_MTE1, pingpong)` for the next L0
//     extraction reusing this ping-pong slot.
template <unsigned BM, unsigned BN, unsigned KS_logical,
          typename TileMatA, typename TileMatB, typename TileAccC>
[aicore] inline void pto_macro_matmul_s4_block(
    TileMatA& aMatTile,
    TileMatB& bMatTile,
    TileAccC& cAccTile,
    uint64_t pingpong)
{
    static_assert(KS_logical % 2 == 0, "KS_logical must be even (nibble pairs)");
    constexpr unsigned KS_packed = KS_logical / 2;

    using LeftTile  = TileLeft <int8_t, BM,        KS_packed, BM,        KS_packed>;
    using RightTile = TileRight<int8_t, KS_packed, BN,        KS_packed, BN>;

    LeftTile  alTile;
    RightTile blTile;
    TASSIGN(alTile, pingpong * (uint64_t)SVDQ_S4_L0A_BYTES_PER_BUF);
    TASSIGN(blTile, pingpong * (uint64_t)SVDQ_S4_L0B_BYTES_PER_BUF);

    // Both extract from the (0,0) sub-fractal — caller advances the
    // L1 view position (via TASSIGN on aMatTile/bMatTile) so the
    // helper always sees the current K-block at the origin.
    TEXTRACT(alTile, aMatTile, 0, 0);
    TEXTRACT(blTile, bMatTile, 0, 0);

    set_flag(PIPE_MTE1, PIPE_M, (event_t)pingpong);
    wait_flag(PIPE_MTE1, PIPE_M, (event_t)pingpong);

    // Raw mad_s4 — same call shape as `kernel_operator_mm_impl.h:330`.
    // cmatrixInitVal=true overwrites L0C with the new product because
    // each K-block's int32 partial needs to leave L0C separately for
    // per-K-block dequant in vec.
    mad_s4((__cc__ int32_t*)cAccTile.data(),
           (__ca__ void*)alTile.data(),
           (__cb__ void*)blTile.data(),
           BM, KS_logical, BN,
           /*unitFlag=*/0,
           /*kDirectionAlign=*/false,
           /*cmatrixSource=*/0,
           /*cmatrixInitVal=*/true);
}

}  // namespace pto

#endif  // SVDQUANT_GEMM_W4A4_PTO_MACRO_MATMUL_S4_HPP
