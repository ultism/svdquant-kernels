// gemm_w4a4 — Ascend __aicore__ device kernel.
//
// Phase 2a (current): cube/vec dispatch + FFTS handshake skeleton.
// Algorithm-free; goal is (a) verify both AIC and AIV cores actually
// launch from the aclrtlaunch auto-gen wrapper, (b) prove the FFTS
// sync table is wired and a cube↔vec round-trip handshake unblocks
// both sides cleanly. Real GEMM math arrives in Phase 2b/2c (cube
// fp16 mock + vec rescale) and Phase 3 (real mad_s4).
//
// `__enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_2`
// puts the auto-gen wrapper into mix-mode (1 cube : 2 vec, A2/A3
// canonical). The CANN 8.5 host-stub generator parses this marker out
// of the preprocessed source via regex and then:
//   - Injects a hidden `void* ffts_addr` as the first kernel arg.
//   - Calls `set_ffts_base_addr((uint64_t)ffts_addr)` from inside the
//     auto-generated `auto_gen_<name>_kernel` wrapper *before* it
//     transfers control to our `_origin`. So this user kernel does
//     not see ffts_addr explicitly, and does not need to set it.
// Without this marker the wrapper picks AIV-only mode and the AIC
// branch is never reached. See `extract_host_stub.py:547-556, 1888-1892`
// in `/usr/local/Ascend/cann-8.5.0/.../legacy_modules/util/`.

#include "kernel_operator.h"

constexpr KernelMetaType __enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_2;

namespace {

// FFTS flag IDs used by gemm_w4a4. Phase 2a only exercises the
// HANDSHAKE_* slots; CUBE_TILE_READY / VEC_TILE_CONSUMED are reserved
// for the 6-slot ring-buffer producer/consumer pattern landing in
// Phase 2b/2c — declared up front so the enum is stable across the
// kernel rewrite. FFTS msg encoding is `0x1 | (mode<<4) | (flag<<8)`
// (`GetffstMsg`); flag IDs share a 4-bit field, so values must stay
// in `[0, 15]`.
enum GemmFftsFlag : uint16_t {
    HANDSHAKE_CUBE_TO_VEC = 0,
    HANDSHAKE_VEC_TO_CUBE = 1,
    CUBE_TILE_READY      = 2,  // Phase 2b: cube → vec, GM-ring slot ready
    VEC_TILE_CONSUMED    = 3,  // Phase 2c: vec → cube, GM-ring slot free
};

// Mix mode 1:2 means each AIC has two paired AIV subblocks. Cube has
// to signal both before the cube-side wait can complete with one ack
// from a single nominated vec. Keep this as a named constant so the
// dependence on the AIC:AIV ratio is explicit at the call sites.
constexpr uint16_t kAivPerAic = 2;

}  // namespace

extern "C" __global__ [aicore] void
svdquant_gemm_w4a4_kernel(GM_ADDR params) {
    (void)params;

    // Cube → both Vec → Cube handshake. Counter semantics: each
    // `ffts_cross_core_sync` increments the FFTS slot once and each
    // `wait_flag_dev` consumes one increment, so cube must signal
    // once *per vec subblock* and only one vec sends the ack back
    // (the other vec exits silently after consuming its CUBE→VEC
    // signal). Pipes follow FA convention: PIPE_FIX is the cube tail
    // pipe, PIPE_MTE3 is the vec UB→GM tail pipe.
    if ASCEND_IS_AIC {
        for (uint16_t i = 0; i < kAivPerAic; ++i) {
            ffts_cross_core_sync(PIPE_FIX,
                AscendC::GetffstMsg(0x2, HANDSHAKE_CUBE_TO_VEC));
        }
        wait_flag_dev(HANDSHAKE_VEC_TO_CUBE);
    }

    if ASCEND_IS_AIV {
        wait_flag_dev(HANDSHAKE_CUBE_TO_VEC);
        if (get_subblockid() == 0) {
            ffts_cross_core_sync(PIPE_MTE3,
                AscendC::GetffstMsg(0x2, HANDSHAKE_VEC_TO_CUBE));
        }
    }
}
