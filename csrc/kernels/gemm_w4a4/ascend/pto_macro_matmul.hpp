// gemm_w4a4 — vendored fp16 cube K-loop macro (Phase 2b mock only).
//
// Verbatim copy of `pto-isa/tests/npu/a2a3/src/st/testcase/tfa/
// pto_macro_matmul.hpp` (see Apache 2.0 header below). Test-side
// helpers aren't in the public PTO include tree (`pto-isa/include/`)
// — vendoring keeps the device-side build self-contained without
// dragging tests/ into our include path. Phase 3's real mad_s4 cube
// loop will use a separately-written macro (different L0 buffer
// sizes, different tile element types), so this file is meant to be
// thrown away once the mock is no longer needed.
//
// One stylistic delta from upstream: the upstream file does not have
// header guards (it's only ever included once per TU in FA), so we
// don't need #pragma once for behavioural parity but include one
// anyway to be safe.

/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SVDQUANT_GEMM_W4A4_PTO_MACRO_MATMUL_HPP
#define SVDQUANT_GEMM_W4A4_PTO_MACRO_MATMUL_HPP

#include <pto/pto-inst.hpp>

namespace pto {

enum class layout_t
{
    NN,
    NT,
    TN,
    TT,
    NONE
};

#define L0A_BUF0 ((__ca__ half *)(__ca__ char *)0x0)
#define L0A_BUF1 ((__ca__ half *)(__ca__ char *)0x8000)
#define L0B_BUF0 ((__ca__ half *)(__ca__ char *)0x0)
#define L0B_BUF1 ((__ca__ half *)(__ca__ char *)0x8000)
#define L0C_BUF0 ((__ca__ half *)(__ca__ char *)0x0)
#define L0C_BUF1 ((__ca__ half *)(__ca__ char *)0x20000)

#define LAST_LOOP(x, n) ((x) == ((n)-1))
#define UNIT_FLAG_ENABLE(i, n) (LAST_LOOP(i, n) ? 3 : 2)

[aicore] inline uint64_t getPingPong(uint32_t flip)
{
    static uint64_t pingpong = 0;
    if (flip) {
        pingpong = 1 - pingpong;
    }
    return pingpong;
}

constexpr uint32_t MEM_BUFFER_SIZE_BYTES = 64 * 1024 / 2;
constexpr uint32_t HALF_SIZE_BYTES = 2;

[aicore] inline constexpr uint32_t calculateFittingCubeK(uint32_t Cube_M, uint32_t Cube_N)
{
    uint32_t bestCubeK = 32;

    if (Cube_M * 256 * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES &&
        256 * Cube_N * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES) {
        bestCubeK = 256;
    } else if (Cube_M * 128 * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES &&
               128 * Cube_N * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES) {
        bestCubeK = 128;
    } else if (Cube_M * 64 * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES &&
               64 * Cube_N * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES) {
        bestCubeK = 64;
    }

    return bestCubeK;
}

template <typename TileDataA, typename TileDataB>
[aicore] inline constexpr layout_t deduce_layout()
{
    if constexpr (TileDataA::SFractal == SLayout::RowMajor && TileDataB::SFractal == SLayout::RowMajor)
        return layout_t::NN;
    if constexpr (TileDataA::SFractal == SLayout::RowMajor && TileDataB::SFractal == SLayout::ColMajor)
        return layout_t::NT;
    if constexpr (TileDataA::SFractal == SLayout::ColMajor && TileDataB::SFractal == SLayout::RowMajor)
        return layout_t::TN;
    if constexpr (TileDataA::SFractal == SLayout::ColMajor && TileDataB::SFractal == SLayout::ColMajor)
        return layout_t::TT;
    return layout_t::NONE;
}

template <unsigned Cube_M, unsigned Tile_K, unsigned Cube_N, layout_t LAYOUT = layout_t::NONE, typename TileDataA,
          typename TileDataB, typename TileDataC>
[aicore] inline void pto_macro_matmul(TileDataA &aMatTile, TileDataB &bMatTile, TileDataC &cAccTile)
{
    constexpr layout_t layout = deduce_layout<TileDataA, TileDataB>();

    static_assert(layout != layout_t::NONE, "Deduced layout is NONE, check tile SLayouts");
    if constexpr (LAYOUT != layout_t::NONE) {
        static_assert(LAYOUT == layout,
                      "Layout mismatch: template LAYOUT does not match deduced layout from tile SLayouts. "
                      "Check SLayout of TileDataA and TileDataB.");
    }

    uint64_t pingpong = getPingPong(0);
    const uint64_t Cube_K = calculateFittingCubeK(Cube_M, Cube_N);
    for (uint64_t k = 0; k < (uint64_t)(Tile_K / Cube_K); k++) {
        using LeftTile = TileLeft<half, Cube_M, Cube_K, Cube_M, Cube_K>;
        LeftTile al0Tiles[2] = {LeftTile(), LeftTile()};
        using RightTile = TileRight<half, Cube_K, Cube_N, Cube_K, Cube_N>;
        RightTile bl0Tiles[2] = {RightTile(), RightTile()};

        TASSIGN(al0Tiles[0], (uint64_t)L0A_BUF0);
        TASSIGN(al0Tiles[1], (uint64_t)L0A_BUF1);
        TASSIGN(bl0Tiles[0], (uint64_t)L0B_BUF0);
        TASSIGN(bl0Tiles[1], (uint64_t)L0B_BUF1);

        wait_flag(PIPE_M, PIPE_MTE1, pingpong);

        if (layout == layout_t::NT) {
            TASSIGN(aMatTile, (uint64_t)aMatTile.data() + k * Cube_K * Cube_M * sizeof(typename TileDataA::DType));
            TASSIGN(bMatTile, (uint64_t)bMatTile.data() + k * Cube_K * Cube_N * sizeof(typename TileDataB::DType));
        }

        TEXTRACT(al0Tiles[pingpong], aMatTile, 0, 0);
        TEXTRACT(bl0Tiles[pingpong], bMatTile, 0, 0);

        set_flag(PIPE_MTE1, PIPE_M, pingpong);
        wait_flag(PIPE_MTE1, PIPE_M, pingpong);

        if (k == 0) {
            TMATMUL(cAccTile, al0Tiles[pingpong], bl0Tiles[pingpong]);
        } else {
            TMATMUL_ACC(cAccTile, cAccTile, al0Tiles[pingpong], bl0Tiles[pingpong]);
        }
        set_flag(PIPE_M, PIPE_MTE1, pingpong);
        pingpong = getPingPong(1);
    }
}

}  // namespace pto

#endif  // SVDQUANT_GEMM_W4A4_PTO_MACRO_MATMUL_HPP
