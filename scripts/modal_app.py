"""Run svdquant kernels on Modal.

Two execution paths:

1. **Native CUDA artifacts on B200** — `smoke` / `tests` below.
   Prereq: `./scripts/build.sh` to populate `./build/`. `build/` is
   mounted read-only at `/root/build` via `add_local_dir(copy=False)`,
   so each run picks up the latest local build.

2. **Triton kernels on RTX PRO 6000 Blackwell** — `triton_smoke` /
   `triton_bench`. SM_120, workstation Blackwell — not B200, but close
   enough to the local RTX 5060 Ti for apples-to-apples perf numbers
   with more bandwidth and SMs. No local build needed; only Python
   sources are shipped. Comparison baseline is nunchaku, installed from
   the upstream GitHub wheel (same version pinned in the local venv).

Usage:
    modal run scripts/modal_app.py                      # native smoke (B200)
    modal run scripts/modal_app.py::tests               # native ctest (B200)
    modal run scripts/modal_app.py::triton_smoke        # Triton correctness (RTX-PRO-6000)
    modal run scripts/modal_app.py::triton_bench        # Triton vs nunchaku bench (RTX-PRO-6000)
    modal run scripts/modal_app.py::gemm_smoke          # gemm_w4a4 ref smoke (B200)
    modal run scripts/modal_app.py::gemm_v0_smoke       # gemm_w4a4 v0 kernel smoke (B200, CuTe DSL)
    modal run scripts/modal_app.py::gemm_bench          # gemm_w4a4 vs fp16 torch bench (B200)
    modal run scripts/modal_app.py::lora_sat_prof_bench # LoRA kernel-pure timing via torch.profiler (B200)

Profiling on Modal: only the CUPTI Activity path works. `torch.profiler`
(activity buffer) and `nsys --trace=cuda,nvtx` (kernel trace) run; `ncu`
and `nsys --gpu-metrics-device` both hit perf-counter init that the
host blocks (`NVreg_RestrictProfilingToAdminUsers=1`), and no container
capability or env override changes that. For roofline / tensor-pipe
utilization we need a different GPU host.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
BUILD_DIR.mkdir(exist_ok=True)

app = modal.App("svdquant-kernels")

# --- Native CUDA image (B200) ---------------------------------------------
image = (
    modal.Image.from_registry("pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime")
    .add_local_dir(str(BUILD_DIR), remote_path="/root/build", copy=False)
)

# --- Torch + nunchaku image (Triton runs + gemm harness on B200) ----------
# Python 3.12 + torch 2.11 + cu13.0, paired with nunchaku's matching
# cu13.0/torch2.11/cp312 wheel from the v1.2.1 GitHub release. Keeping
# CUDA versions aligned across torch and nunchaku avoids the libcudart
# dlopen pitfall we hit when mixing cu13 torch with a cu12-linked
# nunchaku wheel. Triton 3.6 ships as a torch 2.11 dep, so no extra pin.
# Same image runs on RTX PRO 6000 (Triton kernels) and B200 (gemm_w4a4
# ref + nunchaku baseline); nunchaku's wheel includes SM_100 and SM_120.
_NUNCHAKU_WHL = (
    "https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/"
    "nunchaku-1.2.1+cu13.0torch2.11-cp312-cp312-linux_x86_64.whl"
)
triton_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(_NUNCHAKU_WHL)
    # CuTe DSL — the cute_kernels/ pods JIT against this. `cuda-python`
    # is a hard dep we use directly (cuda_drv.CUstream) in the host
    # wrapper; pin with nvidia-cutlass-dsl so the pair stays aligned.
    .pip_install("nvidia-cutlass-dsl", "cuda-python")
    .add_local_dir(
        str(ROOT / "triton_kernels"),
        remote_path="/root/svdquant-kernels/triton_kernels",
        copy=False,
    )
    .add_local_dir(
        str(ROOT / "cute_kernels"),
        remote_path="/root/svdquant-kernels/cute_kernels",
        copy=False,
    )
    .add_local_dir(
        str(ROOT / "baseline"),
        remote_path="/root/svdquant-kernels/baseline",
        copy=False,
    )
    .add_local_file(
        str(ROOT / "tmp" / "bench_fused.py"),
        remote_path="/root/svdquant-kernels/tmp/bench_fused.py",
        copy=False,
    )
    .add_local_file(
        str(ROOT / "tmp" / "smoke_fused.py"),
        remote_path="/root/svdquant-kernels/tmp/smoke_fused.py",
        copy=False,
    )
    .add_local_file(
        str(ROOT / "tmp" / "smoke_gemm.py"),
        remote_path="/root/svdquant-kernels/tmp/smoke_gemm.py",
        copy=False,
    )
    .add_local_file(
        str(ROOT / "tmp" / "smoke_gemm_v0.py"),
        remote_path="/root/svdquant-kernels/tmp/smoke_gemm_v0.py",
        copy=False,
    )
    .add_local_file(
        str(ROOT / "tmp" / "smoke_gemm_v1.py"),
        remote_path="/root/svdquant-kernels/tmp/smoke_gemm_v1.py",
        copy=False,
    )
    .add_local_file(
        str(ROOT / "tmp" / "bench_gemm.py"),
        remote_path="/root/svdquant-kernels/tmp/bench_gemm.py",
        copy=False,
    )
    .add_local_file(
        str(ROOT / "tmp" / "bench_gemm_v0.py"),
        remote_path="/root/svdquant-kernels/tmp/bench_gemm_v0.py",
        copy=False,
    )
    .add_local_file(
        str(ROOT / "tmp" / "bench_lora_saturation_prof.py"),
        remote_path="/root/svdquant-kernels/tmp/bench_lora_saturation_prof.py",
        copy=False,
    )
)


@app.function(gpu="B200", image=image)
def smoke() -> None:
    subprocess.run(["nvidia-smi"], check=True)
    import torch

    print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
    print(
        f"device: {torch.cuda.get_device_name(0)}, "
        f"sm{''.join(map(str, torch.cuda.get_device_capability(0)))}"
    )
    root = Path("/root/build")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"build/ contains {len(files)} files:")
    for p in files:
        print(f"  {p.relative_to(root)}  ({p.stat().st_size} B)")


@app.function(gpu="B200", image=image)
def tests() -> None:
    root = Path("/root/build")
    bins = [
        p for p in root.rglob("*")
        if p.is_file() and os.access(p, os.X_OK) and p.suffix == ""
    ]
    if not bins:
        print("no executables under build/ — rebuild with TESTS=ON?")
        return
    for b in bins:
        print(f"==> {b.relative_to(root)}")
        subprocess.run([str(b)], check=False)


@app.function(gpu="RTX-PRO-6000", image=triton_image, timeout=600)
def triton_smoke() -> None:
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        ["python", "/root/svdquant-kernels/tmp/smoke_fused.py"], check=True
    )


@app.function(gpu="RTX-PRO-6000", image=triton_image, timeout=600)
def triton_bench() -> None:
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        ["python", "/root/svdquant-kernels/tmp/bench_fused.py"], check=True
    )


@app.function(gpu="B200", image=triton_image, timeout=600)
def gemm_smoke() -> None:
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        ["python", "/root/svdquant-kernels/tmp/smoke_gemm.py"], check=True
    )


@app.function(gpu="B200", image=triton_image, timeout=900)
def gemm_v0_smoke() -> None:
    """gemm_w4a4 v0 (main NVFP4 MMA only) — CuTe DSL kernel vs fp32 ref.

    First real execution of the CuTe DSL pod — JIT compile is ~30s on
    first call, per-shape compile cache keys off (out_dtype, tiler).
    """
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        ["python", "/root/svdquant-kernels/tmp/smoke_gemm_v0.py"], check=True
    )


@app.function(gpu="B200", image=triton_image, timeout=900)
def gemm_v1_smoke() -> None:
    """gemm_w4a4 v1 (main NVFP4 + β-interleaved LoRA) — kernel vs fp32 ref.

    Shares the compile cache with v0 per (dtype, tiler, R); first v1
    call per (dtype, tiler, R) triple JITs, subsequent calls reuse.
    """
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        ["python", "/root/svdquant-kernels/tmp/smoke_gemm_v1.py"], check=True
    )


@app.function(gpu="B200", image=triton_image, timeout=900)
def gemm_v0_bench() -> None:
    """gemm_w4a4 v0 latency sweep across ZImage shapes.

    Baseline = fp16 torch.matmul, same (M, K, N). Reports t_ours /
    t_fp16 and speedup. Used to tune the MMA tiler before v1 LoRA
    interleave lands.
    """
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        ["python", "/root/svdquant-kernels/tmp/bench_gemm_v0.py"], check=True
    )


@app.function(gpu="B200", image=triton_image, timeout=600)
def gemm_bench() -> None:
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        ["python", "/root/svdquant-kernels/tmp/bench_gemm.py"], check=True
    )


@app.function(gpu="B200", image=triton_image, timeout=600)
def lora_sat_prof_bench() -> None:
    """Per-kernel device-side LoRA timing via torch.profiler CUPTI Activity.

    Strips kernel-launch overhead (first-order for the tiny LoRA shapes)
    and reports pure device-side kernel time. Uses the CUPTI activity
    buffer, which Modal's host permits; the counter-query path (what
    ncu and nsys --gpu-metrics-device need) is blocked at driver level.
    """
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        ["python", "/root/svdquant-kernels/tmp/bench_lora_saturation_prof.py"],
        check=True,
    )


@app.local_entrypoint()
def main() -> None:
    smoke.remote()
