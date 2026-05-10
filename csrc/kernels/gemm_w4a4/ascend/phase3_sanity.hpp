// Phase 3a-min — uint8 storage Tile + raw `mad_s4` compile probe.
//
// This file exists to clear Phase 3 GAP #1: does the vendored a2a3 PTO
// snapshot accept the type plumbing we need for INT4 cube MMA without
// going through any TMatmul / TMatmulS4 wrapper? The path under test:
//
//   GM (int8 bytes, 2 nibbles each)  --TLOAD-->  L1 Mat<int8>
//   L1 Mat<int8>                     --TEXTRACT-->  L0A/L0B Left/Right<int8>
//   L0A, L0B                          --mad_s4-->  L0C Acc<int32>
//   L0C Acc<int32>                    --TSTORE-->  GM (int32)
//
// Storage choice — `int8_t` not `uint8_t`:
//   * TLoad whitelist (TLoad.hpp:424) accepts both int8_t/uint8_t.
//   * TExtract.hpp:477-480 only accepts int8_t/half/bf16/float — uint8_t
//     would fail the static_assert. Same byte width and the nibble bit-
//     pattern mad_s4 reads via `void*` doesn't care about host signedness,
//     so int8_t is the no-cost path through both gates.
//
// The probe is never launched at runtime. Phase 2d kernel_device.cpp
// calls it under a `volatile bool gate = false` so ccec instantiates
// templates and emits the actual mad_s4 op (volatile read prevents DCE)
// while the live execution path stays Phase 2d.
//
// Once this compiles green, Phase 3a proceeds: pull these template
// shapes into pto_macro_matmul_s4.hpp, K-loop them, replace Phase 2d's
// fp16 pto_macro_matmul in kernel_device.cpp's main path. See PLAN.md
// Phase 3 section for the rest.

#ifndef SVDQUANT_GEMM_W4A4_PHASE3_SANITY_HPP
#define SVDQUANT_GEMM_W4A4_PHASE3_SANITY_HPP

#include "kernel_operator.h"
#include <pto/pto-inst.hpp>

namespace svdquant_phase3 {

[aicore] inline void phase3_int4_compile_probe(__gm__ int8_t* a_gm,
                                                __gm__ int8_t* b_gm,
                                                __gm__ int32_t* c_gm)
{
    // Smallest one-shot mad_s4 shape consistent with PTO Tile inner-box
    // alignment for sizeof(DType)=1:
    //   InnerRow (SLayout::RowMajor, A side) = TileConfig::fixedRowSize = 16
    //   InnerCol (SLayout::RowMajor, A side) = TileConfig::alignedSize/sizeof = 32
    //   InnerRow (SLayout::ColMajor, B side) = TileConfig::alignedSize/sizeof = 32
    //   InnerCol (SLayout::ColMajor, B side) = TileConfig::fixedColSize = 16
    // → K_packed must be a multiple of max(32, 32) = 32 (one fractal box wide).
    //   M must be a multiple of 16, N a multiple of 16.
    // Smallest size that clears every static_assert: M=32, K_packed=32, N=32.
    // K_packed=32 means K_logical=64 nibbles per row.
    constexpr uint32_t M        = 32;
    constexpr uint32_t Klogical = 64;
    constexpr uint32_t Kpacked  = Klogical / 2;  // 32 bytes / row
    constexpr uint32_t N        = 32;

    // Mirror Phase 2d's Tile/Global template parameters exactly; only
    // the element type changes (half → int8_t for A/B, fp32 → int32 for
    // C). NT layout (C = A · B^T) deduced from SFractal pair.
    using TileMatA_S4 = pto::Tile<pto::TileType::Mat, int8_t, M, Kpacked,
                                   pto::BLayout::ColMajor, M, Kpacked,
                                   pto::SLayout::RowMajor, 512>;
    using TileMatB_S4 = pto::Tile<pto::TileType::Mat, int8_t, Kpacked, N,
                                   pto::BLayout::RowMajor, Kpacked, N,
                                   pto::SLayout::ColMajor, 512>;
    using TileLeftA_S4  = pto::TileLeft<int8_t, M, Kpacked, M, Kpacked>;
    using TileRightB_S4 = pto::TileRight<int8_t, Kpacked, N, Kpacked, N>;
    using TileAccC_I32  = pto::TileAcc<int32_t, M, N, M, N>;

    using GlobalA = pto::GlobalTensor<int8_t,
        pto::Shape<1, 1, 1, M, Kpacked>,
        pto::Stride<1, 1, 1, Kpacked, 1>>;
    using GlobalB = pto::GlobalTensor<int8_t,
        pto::Shape<1, 1, 1, Kpacked, N>,
        pto::Stride<1, 1, 1, 1, Kpacked>,
        pto::Layout::DN>;
    using GlobalC = pto::GlobalTensor<int32_t,
        pto::Shape<1, 1, 1, M, N>,
        pto::Stride<1, 1, 1, N, 1>>;

    // L1 byte offsets — A starts at 0, B follows. L0A/B/C all rooted
    // at offset 0 (single buffer, no ping-pong).
    constexpr uint32_t kAByteOffset = 0;
    constexpr uint32_t kBByteOffset = M * Kpacked;
    constexpr uint64_t kL0A_BUF     = 0x0;
    constexpr uint64_t kL0B_BUF     = 0x0;
    constexpr uint64_t kL0C_BUF     = 0x0;

    TileMatA_S4   aMatTile;
    TileMatB_S4   bMatTile;
    TileLeftA_S4  aLeft;
    TileRightB_S4 bRight;
    TileAccC_I32  cAccTile;
    TASSIGN(aMatTile,  kAByteOffset);
    TASSIGN(bMatTile,  kBByteOffset);
    TASSIGN(aLeft,     kL0A_BUF);
    TASSIGN(bRight,    kL0B_BUF);
    TASSIGN(cAccTile,  kL0C_BUF);

    GlobalA aGlobal(a_gm);
    GlobalB bGlobal(b_gm);
    GlobalC cGlobal(c_gm);

    TLOAD(aMatTile, aGlobal);
    TLOAD(bMatTile, bGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TEXTRACT(aLeft,  aMatTile, 0, 0);
    TEXTRACT(bRight, bMatTile, 0, 0);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // Raw CCE intrinsic — same call shape as
    // kernel_operator_mm_impl.h:330. Args (in order):
    //   c, a (void*), b (void*), m, k_logical, n,
    //   unitFlag, kDirectionAlign, cmatrixSource, cmatrixInitVal.
    // cmatrixInitVal=true overwrites C with the new product (no acc).
    mad_s4((__cc__ int32_t*)cAccTile.data(),
           (__ca__ void*)aLeft.data(),
           (__cb__ void*)bRight.data(),
           M, Klogical, N,
           /*unitFlag=*/0,
           /*kDirectionAlign=*/false,
           /*cmatrixSource=*/0,
           /*cmatrixInitVal=*/true);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(cGlobal, cAccTile);
}

}  // namespace svdquant_phase3

#endif  // SVDQUANT_GEMM_W4A4_PHASE3_SANITY_HPP
