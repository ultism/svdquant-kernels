#pragma once
// gemm_w4a4 — main SVDQuant W4A4 linear kernel (Ascend C-side header).
//
// This header declares the **Ascend-side** C++ entry point only. The
// CUDA path does not go through C++ at all — it is a CuTe DSL kernel
// under `cute_kernels/gemm_w4a4/kernel.py`, called directly from
// Python with torch tensors. Keep this header free of CUDA decls.
//
// Consumes packed INT4 activation + FP16 block scales, packed INT4
// weight + FP16 block scales, runs AscendC cube GEMM, accumulates the
// SVDQuant low-rank residual `lora_act_in @ lora_up` in the epilogue,
// optionally biases / rescales, and optionally re-quantizes the result
// for the next layer. (The CUDA / NVFP4 math is the same at the
// tensor-shape level; only the 4-bit format and the tensor-unit
// language differ — see CLAUDE.md "4-bit format splits by backend".)
//
// This is the compute-bound half of SVDQuant. The memory-bound
// preprocessing op that produces `act` / `ascales` / `lora_act_in`
// lives in `triton_kernels/quantize_w4a4_act_fuse_lora/` (Triton,
// shared across CUDA and Ascend).
//
// Logical shapes (Ascend INT4 packed layout; strides in elements):
//   act          [M,   K/2]   uint8     2 signed INT4 / byte
//   wgt          [N,   K/2]   uint8     2 signed INT4 / byte
//   ascales      [K/64, M]    fp16      per-64-K-block act scale
//   wscales      [K/64, N]    fp16      per-64-K-block weight scale
//   lora_act_in  [M,   R]     fp32      = fpsum @ lora_down from previous op
//   lora_up      [N,   R]     fp16/bf16
//   bias         [N]          fp16/bf16 (optional)
//   smooth       [N_next]     fp16/bf16 (optional; next layer's smooth factor)
//   out          [M, N]       fp16/bf16
//   qout         [M, N/2]     uint8     (optional) pre-quantized output for next layer
//   oscales      [N/64, M]    fp16      (optional) matching scales for `qout`
//
// Reference: `tmp/nunchaku/src/kernels/zgemm/gemm_w4a4.cu:34-105`.
//
// Phase 2f signature: raw device pointers + `void* stream`, no struct
// wrapper. The torch op extension (`csrc/python/host/svdquant_w4a4_op.cpp`)
// calls this directly with `at::Tensor.storage().data()` pointers and
// the current NPU stream from `c10_npu::getCurrentNPUStream`. Phase 2d
// device kernel still uses constexpr tile sizes (64×128×128, ring=6,
// num_tiles=8); the dims are not yet parameters here because the
// device kernel doesn't accept them yet — Phase 3 will add ascales /
// wscales / lora_* and parameterize.

namespace svdquant::ascend {

// Phase 2f — fp16 mock interface (matches Phase 2d kernel_device.cpp).
//   act:    [kTileM, kTileK]                        fp16  device pointer
//   wgt:    [kTileK, kTileN]                        fp16  device pointer
//   out:    [(kRingSlots + kNumTiles), kTileM, kTileN] fp32 device pointer
//             - first kRingSlots × M × N            cube ring scratch
//             - next  kNumTiles  × M × N            vec_out (final result)
//   stream: aclrtStream cast to void* (kept opaque so this header
//           doesn't drag in CANN headers; the .cpp casts back).
//
// Caller (the torch op wrapper) is responsible for allocating `out`
// large enough for both the cube ring and the linear vec_out region.
// Synchronization is the caller's responsibility — the launcher
// returns immediately after `aclrtlaunch_*` issues the kernel.
void gemm_w4a4(void* act, void* wgt, void* out, void* stream);

}  // namespace svdquant::ascend
