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

### Phase 2 — Cube/Vec 协作骨架（通信不死锁）✅

**目标**：cube 和 vec 都跑真 launch，FFTS 跨核同步可用，软流水跑得起来；算法是 mock，但通信图完整。

完工 2026-05-10。所有子项验证通过(见 `phase2[a-f]_npu_smoke_validated.md`)：
mix-mode 1:2 + FFTS handshake + 6-slot ring + preload/main pipeline + PyTorch fp16 reference + torch op extension (path C)。

### Phase 3a — INT4 main path（数值正确）✅

完工 2026-05-11。

- ✅ 主 cube 路径：uint8_t Tile + 裸 `mad_s4`；K-loop 每 KS=64 nibble 一发，int32 → L0C → FIX-TSTORE → 6-slot ring。
- ✅ vec per-K-block dequant：`TROWEXPANDMUL`(ascale) + `TCOLEXPANDMUL`(wscale)，fp32 累加到 `runningOTile`。
- ✅ vec epilogue：cast fp32→fp16 → 主输出 TSTORE。
- ✅ `baseline/kernels/gemm_w4a4/ref_int4.py` PyTorch reference。
- ✅ 910B Space smoke：M=64 K=128 N=128 vs ref_int4 **max_abs=0.0010** (cycle 17, 2026-05-11)。

期间踩到的硅级坑(已固化到 `docs/gotchas/ascend.md`)：cube↔vec L2 handoff、cube 1-byte min addressable、`mad_s4` 路径选型、`TLoad ColMajor [N,1]` 只载 head、`TRowExpand` 污染 vec mask、AIV K-loop 跨 iter V→MTE2 sync。

### Phase 3b — LoRA-up residual（代码完成，Space 端验证 pending）⏸

完工 2026-05-12 本地;Space 端因 GitCode 910B 资源池"系统错误"暂无法上机验证。

- ✅ op schema 扩展：`gemm_w4a4(act, wgt, ascales, wscales, lora_act_in, lora_up) -> Tensor`。R=32(production shipping point)。
- ✅ host op：LA fp32→fp16 cast + LU [N, R]→[R, N] transpose。
- ✅ device cube LoRA-up 第二 pass：单 mad fp16×fp16→fp32 → GM lora_buf。新 FFTS flag `LORA_BUF_READY`。
- ✅ device vec：`wait_flag_dev(LORA_BUF_READY)` → TLOAD lora_buf → TADD running → final TCVT/TSTORE。
- ✅ test 解开 lora=zeros，换 seeded random(amp 0.1)。
- ✅ 本地 x86 + aarch64 双 cross-build 绿。fixes 都已 land：
  - `7041864`: 重新 cross-build aarch64 .o(stale 7-arg 触发 undefined symbol)
  - `d437b31`: aarch64 `-fPIC`(R_AARCH64_ADR_PREL_PG_HI21 relink 失败)
- ⏸ Space 端 e2e validation：GitCode 激活 3× "系统错误"，等下次 retry。

**已知风险**(下次 Space 验证时重点看)：TileMatLUT 用 ND2NZ 而不是 DN2ZN — 编译过但 TEXTRACT 到 TileRight 的语义是否正确没有验过。若数值偏差大，先怀疑这个;若数值 OK，3b 收工。

### Phase 3c — Multi-tile + production shape + bias（pending）

- [ ] tile 参数化：把 kBM/kBN/kBKLogical/kR 从 constexpr 改成 launcher 传入。
- [ ] 上 production shape (M=128 K=2048 N=256)。
- [ ] bias、wcscales。
- [ ] grid M-major launch，cores 共享 act 矩阵在 L2。
- [ ] profile pass：cube/vec 占用率、L2 命中、ring 槽数调优。

## 不做（明确出 scope）

- ❌ vLLM pipeline 侵入式 fusion（`fuse_glu` 等）—— `vllm_consumer_scope.md`
- ❌ next-layer NVFP4 quant 集成（v3 已 drop）—— `docs/architecture.md` § Scope decisions
- ❌ 给 PTO 上游加 dtype-aware TLoad/TMov —— 是 SIG 的活，不在 svdquant 范畴
- ❌ 把主 GEMM 和 LoRA-up 累加器在 L0C 上 fuse —— A2/A3 硬件层面做不到

## 当前位置

Phase 3b 代码完成、Space 端 e2e 待验证。任务切换到 CUDA 路径(B200)。下次回到 NPU 路径时先 retry GitCode 激活，跑 `tests/test_gemm_w4a4.py`，若过则进 Phase 3c，若 fail 先看 TileMatLUT layout 选型。
