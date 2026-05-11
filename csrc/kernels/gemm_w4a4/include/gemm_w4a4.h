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
// Phase 3b signature: raw device pointers + opaque stream. The torch
// op extension (`csrc/python/host/svdquant_w4a4_op.cpp`) calls this
// directly with `at::Tensor.storage().data()` pointers and the current
// NPU stream from `c10_npu::getCurrentNPUStream`. Tile size is still
// constexpr-baked into the device kernel (M=64, K=128, N=128, R=16,
// one tile per launch); Phase 3c adds tile parameterization + bias /
// wcscales / multi-tile.
//
// LoRA-up residual is computed on cube in a separate pass after the
// main K-loop drains, then added to running_f32 on vec before final
// fp16 cast. See `csrc/kernels/gemm_w4a4/ascend/kernel_device.cpp`
// for the cube/vec hand-off layout.

namespace svdquant::ascend {

// Phase 3b — INT4 main path + LoRA-up residual.
//
//   act:         [M, K/2]                uint8   2 signed INT4 / byte
//   wgt:         [N, K/2]                uint8
//   ascales:     [K/64, M]               fp16    per-64-K-block act scale
//   wscales:     [K/64, N]               fp16    per-64-K-block wgt  scale
//   lora_act_in: [M, R]                  fp32    = prev-op output, R ≤ 128
//   lora_up:     [N, R]                  fp16
//   workspace:   [kRingSlots, M, N]      int32   cube/vec hand-off ring
//   lora_buf:    [M, N]                  fp32    LoRA-up cube → vec hand-off
//   out:         [M, N]                  fp16    final dequantized output
//   stream:      aclrtStream cast to void*.
//
// Caller (the torch op wrapper) allocates `workspace`, `lora_buf`, and
// `out`. Both `workspace` and `lora_buf` only persist for the duration
// of the call — their contents are not user-visible. Synchronization
// is the caller's responsibility; this launcher does the H2D pack of
// the params struct + `aclrtlaunch_*` and returns.
void gemm_w4a4(void* act, void* wgt,
               void* ascales, void* wscales,
               void* lora_act_in, void* lora_up,
               void* workspace, void* lora_buf, void* out,
               void* stream);

}  // namespace svdquant::ascend
