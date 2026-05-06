# Ascend `gemm_w4a4` 实施计划

跟踪任务：`#64 Implement Ascend gemm_w4a4 pod (uint8_t tile + raw mad_s4)`。

总目标：在 Ascend A2/A3 上实现 SVDQuant W4A4 GEMM —— 主路径用 PTO ISA 的 byte-typed Tile + 裸 `mad_s4` 调用做 s4 cube MMA，外加 fp16/bf16 LoRA-up cube MMA、vec 端 dequant + bias + 可选 next-layer quant。

## 上层决策（已固化在 memory）

- **Tile 路径**：`Tile<Mat, uint8_t, M, K/2>` + `TileLeft/TileRight<uint8_t, ...>` + 裸 `mad_s4(c, (__ca__ void*)a, (__cb__ void*)b, m, k_logical, n, unitFlag, false, src, init)`；epilogue 全部 PTO 抽象。参考 `ascend_pto_mad_s4_route.md`。
- **不复用 nunchaku** —— 它走 NVIDIA PTX `mma.sync` 路径，layout 跟 Ascend cube ABI 不通用。参考 `feedback_no_nunchaku_for_ascend.md`。
- **L2 视角**：cube↔vec 跨核握手通过 GM 地址但 hot-resident 在 L2，FA 的 `qkGlobalTensorNBuffers = 1 + qkPreloadNum = 6` 环形 buffer 是这个 pattern 的样板。参考 `ascend_cube_vec_l2_handoff.md`。
- **不 fuse 主 GEMM 和 LoRA-up 累加器** —— A2/A3 cube 没有 block-scaled mma，主路径出 int32、LoRA-up 出 fp32，dtype 不一致；L0C 只能串行复用，不能跨 dtype 合并。
- **Tile 起步参数（待 profiling 调）**：`BM=128, BN=256, BK_logical=2048, KS=64`（每发 mad_s4 的 K-block）；`acc_fifo_slots=6`；grid M-major（让相邻 cores 在 L2 共享 act 矩阵）。

## 参考代码定位

| 用途 | 文件 |
|---|---|
| ccec build 规则 + 双核占位骨架 | `pto-isa/demos/baseline/gemm_basic/{CMakeLists.txt, csrc/kernel/gemm_basic_custom.cpp}` |
| cube K-loop + L0 ping-pong + FIX-pipe TSTORE | 同上 (`ProcessKIteration`) |
| cube/vec 双身份 + FFTS 跨核同步 + 软流水 + L2 环形 buffer | `pto-isa/tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp` 全文 |
| `pto_macro_matmul`（cube K-loop 双 buffer 模板） | `pto-isa/tests/npu/a2a3/src/st/testcase/tfa/pto_macro_matmul.hpp` |
| `assign_running_acc_tile`（L0C 双 buffer toggle） | `tfa_kernel.cpp:168-179` |
| `pto_macro_fa_gu`（rescale prev × factor + add est —— 跟我们 dequant + accumulate 同款） | `tfa/pto_macro_fa_gu.hpp` |
| AscendC `mad_s4` ABI 真值（10 参数，void* 接 a/b） | `/usr/local/Ascend/cann-8.5.0/x86_64-linux/asc/impl/basic_api/dav_c220/kernel_operator_mm_impl.h:329-331` |

## 三阶段执行

### Phase 1 — Build skeleton（编译链路打通）

**目标**：cross-build 绿，host launcher 能起 cube + vec 空 kernel。

#### Phase 1a — `ascendc_library` 编译链路 ✅

- [x] `csrc/kernels/gemm_w4a4/ascend/kernel_device.cpp` —— **无条件** 一个 `extern "C" __global__ [aicore] void` 占位（不要套 `defined __CCE_AICORE__ == 220` 这种 gemm_basic 用的 buggy 测试，C 预处理器解析出来两支都为假，CANN 自动生成的 wrapper 里 `<sym>_origin` 找不到）。
- [x] `cmake/FindCANN.cmake` 增加 `CANN_ASCENDC_CMAKE` 路径探测。
- [x] 顶层 `CMakeLists.txt` 在 Ascend 使能时 `include(${CANN_ASCENDC_CMAKE})`，预设 `SOC_VERSION`、`ASCEND_CANN_PACKAGE_PATH`、`ASCEND_KERNEL_LAUNCH_ONLY=ON`、`ASCEND_PYTHON_EXECUTABLE` 指向我们的 wrapper。
- [x] `scripts/ascendc_python_wrapper.py` —— normpath 掉 argv 中的 `/./`，绕过 CANN 8.5 `extract_host_stub.py` 在 source 不在 sub-build CMAKE_SOURCE_DIR 之内时的 KeyError bug。
- [x] `scripts/build.sh` 把 CANN bin 路径加进 PATH（`/usr/local/Ascend/cann-8.5.0/x86_64-linux/{bin,ccec_compiler/bin}`），不然 `env -i` 之后 `llvm-objdump` 找不到。
- [x] `csrc/kernels/CMakeLists.txt` 的 `svdquant_add_kernel_pod` 检测 `kernel_device.cpp`，用 `ascendc_library(... STATIC ...)` + `ascendc_include_directories(... PTO_INCLUDE_DIR ...)` 编，最后 `target_link_libraries(host_obj PUBLIC <pod>_device)`。
- [x] `./scripts/build.sh CUDA=OFF ASCEND=ON` 产出 `lib/libsvdquant_gemm_w4a4_device.a` (~1.1MB，含 ascendc 运行时 + AIC/AIV merged device blob + host_stub)。

#### Phase 1b — host launcher 起空 kernel ✅（编译/链接层 + NPU 真机）

- [x] `ascend/kernel.cpp` 改造：`aclrtMalloc` device blob + `aclrtMemcpy` H2D + `aclrtlaunch_svdquant_gemm_w4a4_kernel(blockDim=1, stream, dev_params)` + `aclrtFree`。
- [x] auto-gen header `aclrtlaunch_svdquant_gemm_w4a4_kernel.h` 通过 device 静态库的 INTERFACE include propagation 自动可见。
- [x] `tmp/smoke_gemm_w4a4_link.cpp` 验证 host obj + device 静态库 + ascendcl/runtime/tiling_api/... 一组依赖能链通（产出 ELF）。
- [x] **GitCode Space 910B smoke 通过** (2026-05-07)：OpenI 不可用后改走 GitCode (AtomGit) AI 社区的免费 910B Space。`gitcode-space` 分支携带本地 cross-built 的 3 个 aarch64 .o（host_stub + kernel + smoke_main），Space 容器侧 `link_smoke.sh` 完成最终链接，`app.py`（Gradio 6.9.0）启动时执行：smoke 跑 `aclInit → aclrtSetDevice(0) → aclrtCreateStream → svdquant::ascend::gemm_w4a4(...) → aclrtSynchronizeStream → aclFinalize`，返回码 0。链接需要的全套 -l 参数 + `-Wl,--copy-dt-needed-entries` 已固化在 `space/link_smoke.sh`。

**当前状态**：编译/链接 + NPU 真机 launch 全绿。可以进 Phase 2。

### Phase 2 — Cube/Vec 协作骨架（通信不死锁）

**目标**：cube 和 vec 都跑真 launch，FFTS 跨核同步可用，软流水跑得起来；算法是 mock，但通信图完整。

- [ ] 把 FA 的 FFTS flag enum + `ffts_cross_core_sync` + `wait_flag_dev` pattern 搬进 `kernel_device.cpp`。
- [ ] 加 `assign_running_acc_tile`、`assign_tile_buffers`、`allocate_cube_tile_buffers` / `allocate_vec_tile_buffers` helper（直接抄）。
- [ ] cube 端 mock：跑 `pto_macro_matmul` 做 fp16 GEMM（不是 s4，便于先验证流水）；TSTORE 到 GM 环形 buffer（6 槽）。
- [ ] vec 端 mock：拿 cube 输出 + 一个常量缩放，TROWEXPANDMUL + TADD 到 `runningOTile`，最后 TSTORE。
- [ ] 加 preload loop + main loop 的双段结构（参考 `tfa_kernel.cpp:599-676`）。
- [ ] 跟 PyTorch fp16 reference 对一遍数（mock 算法等价于一个简单 GEMM × 常量）；目标是验证通信，不求最终算法。
- [ ] OpenI smoke：跑一组小 shape 不死锁，输出跟 reference 误差在 fp16 量级。

**完工标志**：cube 和 vec 都跑、6-slot 环形 buffer 工作、preload pipeline 不卡。

### Phase 3 — 真算法上（数值正确）

**目标**：替换 mock 为真 SVDQuant 算法，跟 `baseline/` PyTorch 参考过数。

- [ ] 主 cube 路径：`Tile<Mat, uint8_t, BM, BK/2>` + `TileLeft/TileRight<uint8_t, ...>` + 裸 `mad_s4(...)`；K-loop 每 KS=64 个 K nibble 一发，int32 累加在 L0C，结束后 FIX-pipe TSTORE 到 GM int32 ring。
- [ ] vec 端 per-K-block dequant：拿 ascale[kb] × wscale[kb]（`TROWEXPANDMUL` 两次），fp32 累加到 `runningOTile`。模式跟 `pto_macro_fa_gu` 几乎一样，只是因子来源不同。
- [ ] LoRA-up cube 阶段：`pto_macro_matmul` 跑 fp16/bf16 `lora_act_in @ lora_up`（fp32 acc），TSTORE 到第二条 GM ring。
- [ ] vec epilogue：load LoRA-up 结果 + bias，加到 `runningOTile`，cast → 主输出 TSTORE。
- [ ] 可选 next-layer quant 分支：跟 `triton_kernels/quantize_w4a4_act_fuse_lora` 的 quant 数学对齐，TSTORE qout / oscales（参考 `tquant/` ST 测例）。
- [ ] `baseline/` 加 PyTorch reference（如果还没有）；smoke 比对所有 shape 误差 ≤ 适当 tolerance（int4 路径常见 1e-2 量级）。
- [ ] Profile 一遍 cube/vec 占用率、L2 命中率（如果 cce profiler 能拿到），确认环形 buffer 槽数和 BM/BN 选型合理。

**完工标志**：所有 production shape 数值通过 + 一份性能基线数据。

## 不做（明确出 scope）

- ❌ vLLM pipeline 侵入式 fusion（`fuse_glu` 等）—— `vllm_consumer_scope.md`
- ❌ next-layer NVFP4 quant 集成到 CUDA 端 v3（已 drop）—— `gemm_w4a4_v3_scope_dropped.md`
- ❌ 给 PTO 上游加 dtype-aware TLoad/TMov —— 是 SIG 的活，不在 svdquant 范畴
- ❌ 把主 GEMM 和 LoRA-up 累加器在 L0C 上 fuse —— A2/A3 硬件层面做不到

## 当前位置

Phase 1 起步 —— 准备拷 gemm_basic 编译骨架，接 svdquant 现有 `svdquant_add_kernel_pod` helper。
