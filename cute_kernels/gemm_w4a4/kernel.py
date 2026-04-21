"""gemm_w4a4 CuTe DSL kernel (Blackwell SM_100/SM_103).

Host-facing contract (v1 — main NVFP4 + β-interleaved LoRA; wcscales /
bias / next-layer quant still land in v2–v3):

    launch(act, wgt, ascales, wscales,
           lora_act_in=None, lora_up=None,  # both-or-neither; None → v0
           *, out_dtype=torch.float16) -> out

Inputs (NVFP4-packed, as produced by
`triton_kernels/quantize_w4a4_act_fuse_lora/`):

    act           [M, K // 2]  uint8           two E2M1 nibbles per byte
    wgt           [N, K // 2]  uint8           two E2M1 nibbles per byte
    ascales       [K // 16, M] fp8_e4m3fn      per-16-K-block act scale
    wscales       [K // 16, N] fp8_e4m3fn      per-16-K-block wgt scale
    lora_act_in   [M, R]       fp32 | out_dtype  (host pre-cast if fp32)
    lora_up       [N, R]       out_dtype       LoRA up-projection weight

Output:
    out       [M, N]         out_dtype (fp16 or bf16)

Device body ported from `tmp/cutlass/examples/python/CuTeDSL/blackwell/
dense_blockscaled_gemm_persistent.py` (stable API). Stripped:
persistent TileScheduler, clusters > 1, TMA multicast, overlapping_accum,
tile_n ∈ {64, 192} SFB-shift hacks. Uses 1SM MMA (cta_group=ONE) on
shape-adaptive tiler (128×128 small-M, 128×256 otherwise).

v1 adds: dense fp16/bf16 tiled_mma for LoRA, single-buffer LA/LU smem
with 1-stage TMA prolog, shared-tmem acc fragment (TV-layout match
verified at trace time by `tmp/verify_tmem_layout.py` for both tilers),
K-loop interleave per design §2 — `stride = K_atoms // R_atoms`
sprinkled in one MMA warp's issue stream.

Design doc: `docs/kernels/gemm_w4a4.md`.
"""
from __future__ import annotations

import os
from typing import Tuple, Type, Union

import torch

# Arch gate: probe returns SM_100/SM_103 on B200/B30x, SM_120 locally
# (consumer Blackwell — rejects tcgen05 NVFP4 atoms). Env override is
# for trace-level checks on a wrong-arch box.
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import cuda.bindings.driver as cuda_drv                  # noqa: E402
import cutlass                                           # noqa: E402
import cutlass.cute as cute                              # noqa: E402
import cutlass.pipeline as pipeline                      # noqa: E402
import cutlass.utils as utils                            # noqa: E402
import cutlass.utils.blackwell_helpers as sm100_utils    # noqa: E402
import cutlass.utils.blockscaled_layout as blockscaled_utils  # noqa: E402
from cutlass.cute.nvgpu import cpasync, tcgen05          # noqa: E402
from cutlass.cute.runtime import make_ptr                # noqa: E402
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait  # noqa: E402

# --- config ---------------------------------------------------------------
# v0 tune: two tilers, shape-adaptive.
#   (128, 128): num_acc_stage = 2, more CTAs.
#   (128, 256): num_acc_stage = 1 (auto by `_compute_stages`); 2× work
#               per CTA, fewer CTAs.
# Small M loses with 128×256: M=256 → 2 M-tiles × 12 N-tiles = 24 CTAs
# vs 148 SMs ⇒ -31% TFLOPS. Large M wins +3–12% from less launch
# overhead + longer MMA pipe run per CTA. Pick per-shape in `launch()`.
# Compile cache keys on (tiler, dtype) so both tilers JIT once.
# 2SM 256×256 is a later concern (needs cluster-aware barriers +
# cta_group = TWO + overlapping_accum).
_TILER_SMALL_M: Tuple[int, int] = (128, 128)
_TILER_DEFAULT: Tuple[int, int] = (128, 256)
# 2-CTA Phase 1: tile_m=256 (two CTAs pair, each owns 128 of M), tile_n=128
# (num_acc_stage=2 ⇒ no overlapping_accum). Phase 2 will add (256, 256)
# with overlapping_accum. Cluster must be (2, 1) for 2-CTA.
_TILER_2CTA: Tuple[int, int] = (256, 128)
# Empirically M ≤ 512 wants more CTAs over bigger tiles. Break-even
# on B200 + ZImage shape set lies between 256 (-31% at 128×256) and
# 4352 (+3–12%); no intermediate datapoint yet, so 512 is a guess
# that covers the known bad case without downgrading the good ones.
_TILER_SMALL_M_THRESHOLD: int = 512


def _pick_tiler(M: int, R: int = 0, use_2cta: bool = False) -> Tuple[int, int]:
    # 2-CTA opt-in bypasses the 1-CTA shape heuristics — see `_TILER_2CTA`.
    if use_2cta:
        return _TILER_2CTA
    # v1 LoRA smem budget: LA (BLOCK_M × R) + LU (BLOCK_N × R), operand
    # dtype fp16/bf16. At BLOCK_N = 256, R = 256, that's 128 KB for LU
    # alone — total LoRA 192 KB eats too much of the 228 KB SM smem
    # budget and `_compute_stages` rounds num_ab_stage down to 0. Fall
    # back to 128×128 for R ≥ 256 (halves LU to 64 KB). Design §4's
    # "double-stage LoRA through K-loop" fallback is the proper fix
    # post-v1 if we need to keep the 128×256 tile on high-R shapes.
    if R >= 256:
        return _TILER_SMALL_M
    return _TILER_SMALL_M if M <= _TILER_SMALL_M_THRESHOLD else _TILER_DEFAULT


def _pick_cluster(use_2cta: bool) -> Tuple[int, int]:
    return (2, 1) if use_2cta else (1, 1)


_CLUSTER_SHAPE_MN: Tuple[int, int] = (1, 1)   # 1-CTA default; see _pick_cluster
_SF_VEC_SIZE: int = 16                 # NVFP4 block size
# Dense fp16/bf16 tcgen05 MMA atom K-reduction size on SM_100. The LoRA
# MMA has `R_atoms = R // _LORA_INST_K` instructions to issue; this
# count is baked into the compile cache key (R varies per shape).
_LORA_INST_K: int = 16

# --- compile cache ---------------------------------------------------------
_COMPILED_CACHE: dict[tuple, object] = {}


# --- torch → cute bridging ------------------------------------------------
def _check_inputs(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    out_dtype: torch.dtype,
    lora_act_in: "torch.Tensor | None" = None,
    lora_up: "torch.Tensor | None" = None,
    use_2cta: bool = False,
) -> Tuple[int, int, int, int, Tuple[int, int]]:
    """Validate layout + dtype. Returns (M, N, K, R, picked_tiler).
    R = 0 when no LoRA; else R = lora_act_in.shape[1]."""
    assert act.is_cuda and wgt.is_cuda and ascales.is_cuda and wscales.is_cuda, \
        "all inputs must live on CUDA"
    assert act.dtype == torch.uint8 and wgt.dtype == torch.uint8, \
        f"act/wgt must be uint8 NVFP4-packed; got {act.dtype}/{wgt.dtype}"
    assert ascales.dtype == torch.float8_e4m3fn and wscales.dtype == torch.float8_e4m3fn, \
        f"scales must be fp8_e4m3fn; got {ascales.dtype}/{wscales.dtype}"
    assert out_dtype in (torch.float16, torch.bfloat16), \
        f"out_dtype must be fp16 or bf16; got {out_dtype}"

    M, K_half = act.shape
    N, K_half_w = wgt.shape
    assert K_half == K_half_w, f"act/wgt K disagree: {K_half} vs {K_half_w}"
    K = K_half * 2
    assert K % _SF_VEC_SIZE == 0, f"K ({K}) must be a multiple of {_SF_VEC_SIZE}"

    K_groups = K // _SF_VEC_SIZE
    assert ascales.shape == (K_groups, M), \
        f"ascales shape {tuple(ascales.shape)} != ({K_groups}, {M})"
    assert wscales.shape == (K_groups, N), \
        f"wscales shape {tuple(wscales.shape)} != ({K_groups}, {N})"

    assert (lora_act_in is None) == (lora_up is None), \
        "lora_act_in and lora_up must both be provided or both None"
    if lora_act_in is None:
        R = 0
    else:
        assert lora_act_in.is_cuda and lora_up.is_cuda, "LoRA inputs must live on CUDA"
        assert lora_act_in.shape[0] == M, \
            f"lora_act_in shape {tuple(lora_act_in.shape)} row != M ({M})"
        assert lora_up.shape[0] == N, \
            f"lora_up shape {tuple(lora_up.shape)} row != N ({N})"
        R = lora_act_in.shape[1]
        assert lora_up.shape[1] == R, \
            f"lora_act_in/lora_up R disagree: {lora_act_in.shape[1]} vs {lora_up.shape[1]}"
        assert R % _LORA_INST_K == 0, \
            f"R ({R}) must be a multiple of LoRA atom K ({_LORA_INST_K})"
        assert lora_up.dtype == out_dtype, \
            f"lora_up dtype {lora_up.dtype} must match out_dtype {out_dtype}"

    if use_2cta:
        assert lora_act_in is None, \
            "2-CTA Phase 1 does not support LoRA; set use_2cta=False when passing lora_*"
    tiler = _pick_tiler(M, R, use_2cta)
    assert M % tiler[0] == 0, f"M ({M}) must be a multiple of {tiler[0]} for tiler {tiler}"
    assert N % tiler[1] == 0, f"N ({N}) must be a multiple of {tiler[1]} for tiler {tiler}"
    assert K % 32 == 0, f"K ({K}) must be a multiple of 32 for NVFP4 alignment"
    return M, N, K, R, tiler


def _fp4_ptr(packed_uint8: torch.Tensor) -> "cute.Pointer":
    """uint8 tensor of shape [., K/2] → cute.Pointer typed Float4E2M1FN
    addressing the same bytes (no copy)."""
    return make_ptr(
        cutlass.Float4E2M1FN,
        packed_uint8.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )


def _sf_ptr(sf_uint8: torch.Tensor) -> "cute.Pointer":
    return make_ptr(
        cutlass.Float8E4M3FN,
        sf_uint8.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )


def _repack_scales_cutlass_atom(scales_km: torch.Tensor) -> torch.Tensor:
    """Repack `[K/16, M]` row-major scales (nunchaku convention) into the
    CUTLASS `BlockScaledBasicChunk` atom-tiled layout.

    The K-major basic chunk at `blockscaled_layout.py` declares
    stride `((16, 4), (0, 1))` over `((32, 4), (sf_vec_size, 4))`, meaning
    inside each 128-M × 4-K_group chunk the fastest-to-slowest storage
    dims are: `k_inner` (stride 1), `m_mid` (stride 4), `m_inner` (stride
    16). `tile_to_shape(chunk, (M, K, L), order=(2, 1, 3))` then extends
    with outer `(rest_m, rest_k, L)` tiles.

    Result: contiguous 6-D tensor with shape
    `(L, rest_m, rest_k, m_inner=32, m_mid=4, k_inner=4)` where the
    logical (m, k_group) at position (`m_outer*128 + m_mid*32 + m_inner`,
    `k_outer*4 + k_inner`) maps to `[0, m_outer, k_outer, m_inner, m_mid,
    k_inner]`.  Returns the tensor (caller grabs data_ptr).

    Copy happens on-device; scales are tiny (K/16 × M fp8 bytes, << act
    payload) so this is in the noise for bench.
    """
    sfk, M = scales_km.shape
    assert M % 128 == 0, f"CUTLASS SF atom: M ({M}) must be a multiple of 128"
    assert sfk % 4 == 0, f"CUTLASS SF atom: K/16 ({sfk}) must be a multiple of 4"
    rest_m = M // 128
    rest_k = sfk // 4
    # [K/16, M] → [M, K/16]
    mk = scales_km.T.contiguous()
    # Split: M = (rest_m, m_mid=4, m_inner=32)  K/16 = (rest_k, k_inner=4)
    # Storage order: (rest_m, rest_k, m_inner, m_mid, k_inner)
    mk5 = mk.reshape(rest_m, 4, 32, rest_k, 4)         # (rm, mmid, minn, rk, kinn)
    storage = mk5.permute(0, 3, 2, 1, 4).contiguous()  # (rm, rk, minn, mmid, kinn)
    return storage.unsqueeze(0)                        # L=1 front


def _c_ptr(out: torch.Tensor) -> "cute.Pointer":
    dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[out.dtype]
    return make_ptr(dtype, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def _lora_ptr(t: torch.Tensor) -> "cute.Pointer":
    """fp16/bf16 LoRA operand → cute.Pointer. LA/LU are K-major dense
    (K = R, contiguous) with row stride = R elements."""
    dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[t.dtype]
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def _cutlass_c_dtype(out_dtype: torch.dtype) -> Type[cutlass.Numeric]:
    return {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[out_dtype]


def launch(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: "torch.Tensor | None" = None,
    lora_up: "torch.Tensor | None" = None,
    *,
    out_dtype: torch.dtype = torch.float16,
    use_2cta: bool = False,
) -> torch.Tensor:
    """v0: y = scaled_mma(act_nvfp4, wgt_nvfp4) — no LoRA, no affine.
    v1 (when both lora_* are provided): y += lora_act_in @ lora_up.T
    via β-interleave into the main K-loop (design §2).

    `use_2cta=True` opts into the 2-CTA (CtaGroup.TWO, cluster=(2,1))
    path with tile (256, 128). Phase 1: v0-only (no LoRA), tile_n=128
    (num_acc_stage=2, no overlapping_accum). Phase 2 will add tile
    (256, 256) + overlapping_accum + LoRA re-enable."""
    M, N, K, R, tiler = _check_inputs(
        act, wgt, ascales, wscales, out_dtype, lora_act_in, lora_up, use_2cta)
    enable_lora = lora_act_in is not None
    cluster_shape_mn = _pick_cluster(use_2cta)
    out = torch.empty((M, N), dtype=out_dtype, device=act.device)

    cache_key = (
        _cutlass_c_dtype(out_dtype), tiler, _SF_VEC_SIZE, cluster_shape_mn,
        enable_lora, R,
    )
    compiled = _COMPILED_CACHE.get(cache_key)
    if compiled is None:
        compiled = _compile(
            c_dtype=_cutlass_c_dtype(out_dtype), tiler_mn=tiler,
            cluster_shape_mn=cluster_shape_mn,
            enable_lora=enable_lora, R=R,
        )
        _COMPILED_CACHE[cache_key] = compiled

    # Scale tensors arrive in nunchaku's K-major `[K/16, MN]` layout;
    # repack to CUTLASS's BlockScaled SF atom layout before handing off.
    ascales_atom = _repack_scales_cutlass_atom(ascales)
    wscales_atom = _repack_scales_cutlass_atom(wscales)

    # v1: LoRA `lora_act_in` wire dtype is fp32 (per fuse-lora op); the
    # dense MMA atom reads fp16/bf16 from smem. Design §4 option (a)
    # defers this cast to the load warp; v1 pre-casts at the host
    # boundary for simplicity (cost ≈ M·R·4 B copy, sub-µs at ZImage
    # scale). Revisit when the load warp is the bottleneck.
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    # v0 path: LA/LU are dummy 0-ptrs (the device body skips them when
    # self.enable_lora is False). Keeps one __call__ signature across
    # both paths — easier compile caching, no duplicated host plumbing.
    if enable_lora:
        la_cast = (
            lora_act_in.to(out_dtype).contiguous()
            if lora_act_in.dtype != out_dtype
            else lora_act_in.contiguous()
        )
        lu_cast = lora_up.contiguous()
        la_ptr = _lora_ptr(la_cast)
        lu_ptr = _lora_ptr(lu_cast)
    else:
        la_ptr = make_ptr(
            _cutlass_c_dtype(out_dtype), 0, cute.AddressSpace.gmem, assumed_align=16)
        lu_ptr = make_ptr(
            _cutlass_c_dtype(out_dtype), 0, cute.AddressSpace.gmem, assumed_align=16)
    compiled(
        _fp4_ptr(act),
        _fp4_ptr(wgt),
        _sf_ptr(ascales_atom),
        _sf_ptr(wscales_atom),
        la_ptr, lu_ptr,
        _c_ptr(out),
        (cutlass.Int32(M), cutlass.Int32(N), cutlass.Int32(K), cutlass.Int32(1)),
        stream,
    )
    return out


# --- JIT compile ----------------------------------------------------------


def _compile(
    *,
    c_dtype: Type[cutlass.Numeric],
    tiler_mn: Tuple[int, int] = _TILER_DEFAULT,
    cluster_shape_mn: Tuple[int, int] = _CLUSTER_SHAPE_MN,
    enable_lora: bool = False,
    R: int = 0,
):
    """AOT-compile the kernel; shapes (M/N/K) are runtime, R is
    compile-time (baked into LoRA atom unroll) when `enable_lora`."""
    kernel_obj = Sm100GemmW4A4(
        sf_vec_size=_SF_VEC_SIZE,
        mma_tiler_mn=tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        ab_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E4M3FN,
        c_dtype=c_dtype,
        enable_lora=enable_lora,
        R=R,
    )
    a_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=32)
    b_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfa_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfb_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    # LA/LU ptrs are always in the __call__ signature — dummy when
    # enable_lora=False, the device body gates on `self.enable_lora`.
    la_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    lu_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    dummy_stream = cuda_drv.CUstream(0)

    return cute.compile(
        kernel_obj,
        a_ptr, b_ptr, sfa_ptr, sfb_ptr, la_ptr, lu_ptr, c_ptr,
        (cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)),
        dummy_stream,
    )


# --- device kernel --------------------------------------------------------
class Sm100GemmW4A4:
    """Device-side kernel: main NVFP4 scaled-MMA + (optional) β-interleaved
    LoRA dense MMA. Non-persistent.

    Layout fixed: A K-major [M, K], B K-major [N, K], C row-major [M, N],
    all batch L = 1.  Cluster (1, 1), single-CTA 1SM MMA atom.
    Warp-spec split = 1 TMA load (warp 5) + 1 MMA (warp 4) + 4 epilogue
    (warps 0–3) = 6 warps = 192 threads.

    Porting base: dense_blockscaled_gemm_persistent.py (stable API).
    Stripped:
      - PersistentTileScheduler + outer `while work_tile.is_valid_tile`
      - cluster_shape > (1,1) and multicast masks
      - use_2cta_instrs path (cta_group = ONE hard-coded)
      - overlapping_accum (num_acc_stage = 2, cta_tile_n = 128)
      - tile_n ∈ {64, 192} SFB-shift hacks

    v1 (`enable_lora=True`) adds: dense tiled_mma for LoRA A/B (operand
    dtype = c_dtype), single-buffer LA/LU smem + 1-stage TMA prolog,
    shared-tmem acc fragment rebuilt through lora_mma partition, K-loop
    interleave per design §2.
    """

    def __init__(
        self,
        *,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        enable_lora: bool = False,
        R: int = 0,
    ):
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        # tile_m = 256 → 2-CTA (CtaGroup.TWO + cluster_m ≥ 2 + TMA
        # multicast). tile_m = 128 stays 1-CTA. `cluster_shape_mn`
        # should be (2, 1) for 2-CTA and (1, 1) for 1-CTA — caller's
        # responsibility. 2-CTA 256×256 (overlapping_accum) is Phase 2;
        # initial 2-CTA landing supports (256, 128) only.
        assert mma_tiler_mn[0] in (128, 256), "tile_m ∈ {128, 256}"
        assert mma_tiler_mn[1] in (128, 256), "tile_n ∈ {128, 256}"
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        if self.use_2cta_instrs:
            assert cluster_shape_mn == (2, 1), \
                f"2-CTA requires cluster_shape_mn=(2, 1); got {cluster_shape_mn}"
            assert mma_tiler_mn[1] == 128, \
                "2-CTA tile_n=256 (overlapping_accum) is Phase 2 — use (256, 128) for now"
        else:
            assert cluster_shape_mn == (1, 1), \
                f"1-CTA requires cluster_shape_mn=(1, 1); got {cluster_shape_mn}"
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)   # K filled in setup
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.ab_dtype = ab_dtype
        self.sf_dtype = sf_dtype
        self.c_dtype = c_dtype
        # LoRA config: R = 0 disables v1 path. R_atoms counts dense fp16/bf16
        # tcgen05 instructions (K=16 each). Baked compile-time so the
        # LoRA atom loop can `unroll_full`.
        self.enable_lora = enable_lora
        self.R = R
        if enable_lora:
            assert R > 0 and R % _LORA_INST_K == 0, \
                f"LoRA R ({R}) must be a positive multiple of {_LORA_INST_K}"
            self.R_atoms = R // _LORA_INST_K
            self.lora_ab_dtype = c_dtype   # LA/LU stored at output dtype
        else:
            self.R_atoms = 0
            self.lora_ab_dtype = None
        # A K-major, B K-major, C row-major (N-major in persistent.py's naming)
        self.a_major_mode = tcgen05.OperandMajorMode.K
        self.b_major_mode = tcgen05.OperandMajorMode.K
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * 6

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * (1 + len(self.epilog_warp_id)),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.buffer_align_bytes = 1024

    # -------- host-side setup  --------

    def _setup_attributes(self):
        # MMA instruction shapes. SFB atom always runs at 1-CTA width
        # (cta_group=ONE there), so when the main MMA is 2-CTA the SFB
        # M dim is halved per-CTA.
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # Cluster layout — (1, 1) degenerate for 1-CTA, (2, 1) for 2-CTA.
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )
        # Multicast CTA counts — for 2-CTA cluster (2, 1) with thr_id
        # shape 2, num_mcast_ctas_a/b = 1 (no along-axis broadcast; the
        # pair-sharing is implicit via CtaGroup.TWO). For future clusters
        # (>1 along M/N), the >1 result enables the classic A/B mcast.
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Stage counts. LoRA smem (LA + LU, single-buffer) is deducted
        # from the budget so num_ab_stage shrinks correctly.
        lora_smem_bytes = 0
        if cutlass.const_expr(self.enable_lora):
            la_bytes = (self.mma_inst_shape_mn[0] * self.R
                        * self.lora_ab_dtype.width // 8)
            lu_bytes = (self.mma_inst_shape_mn[1] * self.R
                        * self.lora_ab_dtype.width // 8)
            lora_smem_bytes = la_bytes + lu_bytes
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.ab_dtype,
            self.epi_tile, self.c_dtype, self.c_layout,
            self.sf_dtype, self.sf_vec_size,
            self.smem_capacity - lora_smem_bytes, self.occupancy,
        )
        assert self.num_acc_stage >= 1, f"num_acc_stage = {self.num_acc_stage}"
        # v0 forbids overlapping_accum (we chose tile_n = 128 explicitly)
        self.overlapping_accum = False

        # SMEM staged layouts
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage,
        )

        # TMEM column counts
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
        )

        # LoRA dense MMA + single-buffer LA/LU smem. K-dim of the tiler
        # = R (baked compile-time), giving `num_lora_kblocks = R // 16`
        # atoms per CTA-tile iteration. `_LORA_INST_K = 16` matches
        # Blackwell dense fp16/bf16 tcgen05 shape_mnk[2] (verified via
        # tmp/verify_tmem_layout.py). Acc TV-layout is verified to match
        # main NVFP4 MMA over this tiler, so the two share one tmem
        # region — no separate allocation needed.
        tiled_mma_lora = None
        if cutlass.const_expr(self.enable_lora):
            tiled_mma_lora = sm100_utils.make_trivial_tiled_mma(
                ab_dtype=self.lora_ab_dtype,
                a_leading_mode=self.a_major_mode,
                b_leading_mode=self.b_major_mode,
                acc_dtype=self.acc_dtype,
                cta_group=self.cta_group,
                mma_tiler_mn=self.mma_inst_shape_mn,
            )
            lora_inst_shape_k = cute.size(tiled_mma_lora.shape_mnk, mode=[2])
            assert lora_inst_shape_k == _LORA_INST_K, (
                f"LoRA dense MMA shape_mnk[2] ({lora_inst_shape_k}) != "
                f"expected _LORA_INST_K ({_LORA_INST_K}). Check Blackwell "
                f"dense tcgen05 for dtype {self.lora_ab_dtype}.")
            self.lora_mma_tiler = (
                self.mma_inst_shape_mn[0],
                self.mma_inst_shape_mn[1],
                self.R,
            )
            self.num_lora_stage = 1   # single-buffer prolog
            self.la_smem_layout_staged = sm100_utils.make_smem_layout_a(
                tiled_mma_lora, self.lora_mma_tiler, self.lora_ab_dtype,
                self.num_lora_stage,
            )
            self.lu_smem_layout_staged = sm100_utils.make_smem_layout_b(
                tiled_mma_lora, self.lora_mma_tiler, self.lora_ab_dtype,
                self.num_lora_stage,
            )

        return tiled_mma, tiled_mma_sfb, tiled_mma_lora

    @staticmethod
    def _compute_stages(
        tiled_mma, mma_tiler_mnk, a_dtype, b_dtype,
        epi_tile, c_dtype, c_layout, sf_dtype, sf_vec_size,
        smem_capacity, occupancy,
    ) -> Tuple[int, int, int]:
        """Heuristic stage count — picks num_ab_stage by fitting SMEM budget.
        Copied verbatim from persistent.py:_compute_stages except we pin
        num_acc_stage = 2 (tile_n = 128 never triggers overlapping_accum)."""
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2
        num_c_stage = 2
        a1 = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b1 = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa1 = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        sfb1 = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        c1 = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
        ab_bytes = (
            cute.size_in_bytes(a_dtype, a1)
            + cute.size_in_bytes(b_dtype, b1)
            + cute.size_in_bytes(sf_dtype, sfa1)
            + cute.size_in_bytes(sf_dtype, sfb1)
        )
        c_bytes_per = cute.size_in_bytes(c_dtype, c1)
        c_bytes = c_bytes_per * num_c_stage
        mbar_helpers = 1024
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers + c_bytes)
        ) // ab_bytes
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes * num_ab_stage
            - occupancy * (mbar_helpers + c_bytes)
        ) // (occupancy * c_bytes_per)
        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid_nonpersistent(
        c_tensor: cute.Tensor, cta_tile_shape_mnk: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        """Grid = one CTA per output tile. Non-persistent."""
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c_tensor, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        return (num_ctas_mnl[0], num_ctas_mnl[1], num_ctas_mnl[2])

    # -------- entrypoint --------

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        la_ptr: cute.Pointer,       # dummy when self.enable_lora == False
        lu_ptr: cute.Pointer,       # dummy when self.enable_lora == False
        c_ptr: cute.Pointer,
        problem_mnkl: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
        stream,
    ):
        m, n, k, l = problem_mnkl

        # Build GMEM tensors. A/B K-major → order = (1, 0, 2) ; C N-major
        # = row-major → order = (1, 0, 2) as well.
        a_layout = cute.make_ordered_layout(
            (cute.assume(m, 32), k, l), order=(1, 0, 2))
        b_layout = cute.make_ordered_layout(
            (cute.assume(n, 32), k, l), order=(1, 0, 2))
        c_layout = cute.make_ordered_layout(
            (m, cute.assume(n, 32), l), order=(1, 0, 2))
        a_tensor = cute.make_tensor(a_ptr, a_layout)
        b_tensor = cute.make_tensor(b_ptr, b_layout)
        c_tensor = cute.make_tensor(c_ptr, c_layout)

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        tiled_mma, tiled_mma_sfb, tiled_mma_lora = self._setup_attributes()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA atoms (no multicast — cluster = (1,1))
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id),
            a_tensor, a_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id),
            b_tensor, b_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id),
            sfa_tensor, sfa_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id),
            sfb_tensor, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.ab_dtype, a_smem_layout)
            + cute.size_in_bytes(self.ab_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        ) * atom_thr_size

        # TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor, epi_smem_layout, self.epi_tile,
        )

        # LoRA TMA atoms (when enabled). LA = [M, R] K-major, LU = [N, R]
        # K-major. Each tensor's row is one CTA-tile's LA/LU slab; single
        # K=R, L=1. Partitioned through tiled_mma_lora so the acc
        # fragment TV-layout matches main (shared tmem).
        tma_atom_la = None
        tma_tensor_la = None
        tma_atom_lu = None
        tma_tensor_lu = None
        num_lora_tma_bytes = 0
        if cutlass.const_expr(self.enable_lora):
            la_layout = cute.make_ordered_layout(
                (cute.assume(m, 32), self.R, l), order=(1, 0, 2))
            lu_layout = cute.make_ordered_layout(
                (cute.assume(n, 32), self.R, l), order=(1, 0, 2))
            la_tensor = cute.make_tensor(la_ptr, la_layout)
            lu_tensor = cute.make_tensor(lu_ptr, lu_layout)
            la_smem_layout = cute.slice_(
                self.la_smem_layout_staged, (None, None, None, 0))
            lu_smem_layout = cute.slice_(
                self.lu_smem_layout_staged, (None, None, None, 0))
            tma_atom_la, tma_tensor_la = cute.nvgpu.make_tiled_tma_atom_A(
                sm100_utils.cluster_shape_to_tma_atom_A(
                    self.cluster_shape_mn, tiled_mma_lora.thr_id),
                la_tensor, la_smem_layout, self.lora_mma_tiler, tiled_mma_lora,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_lu, tma_tensor_lu = cute.nvgpu.make_tiled_tma_atom_B(
                sm100_utils.cluster_shape_to_tma_atom_B(
                    self.cluster_shape_mn, tiled_mma_lora.thr_id),
                lu_tensor, lu_smem_layout, self.lora_mma_tiler, tiled_mma_lora,
                self.cluster_layout_vmnk.shape,
            )
            num_lora_tma_bytes = (
                cute.size_in_bytes(self.lora_ab_dtype, la_smem_layout)
                + cute.size_in_bytes(self.lora_ab_dtype, lu_smem_layout)
            ) * atom_thr_size
        self.num_lora_tma_bytes = num_lora_tma_bytes

        grid = self._compute_grid_nonpersistent(c_tensor, self.cta_tile_shape_mnk)

        # SharedStorage. LoRA fields are conditional; CuTe DSL allows
        # building the struct body at trace time so the fields only
        # materialize when enable_lora.
        if cutlass.const_expr(self.enable_lora):
            @cute.struct
            class SharedStorage:
                ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
                ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
                acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
                acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
                lora_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_lora_stage]
                lora_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_lora_stage]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                sC: cute.struct.Align[
                    cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)],
                    self.buffer_align_bytes,
                ]
                sA: cute.struct.Align[
                    cute.struct.MemRange[self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                    self.buffer_align_bytes,
                ]
                sSFA: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sSFB: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sLA: cute.struct.Align[
                    cute.struct.MemRange[self.lora_ab_dtype,
                                         cute.cosize(self.la_smem_layout_staged.outer)],
                    self.buffer_align_bytes,
                ]
                sLU: cute.struct.Align[
                    cute.struct.MemRange[self.lora_ab_dtype,
                                         cute.cosize(self.lu_smem_layout_staged.outer)],
                    self.buffer_align_bytes,
                ]
        else:
            @cute.struct
            class SharedStorage:
                ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
                ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
                acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
                acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                sC: cute.struct.Align[
                    cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)],
                    self.buffer_align_bytes,
                ]
                sA: cute.struct.Align[
                    cute.struct.MemRange[self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                    self.buffer_align_bytes,
                ]
                sSFA: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sSFB: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]

        self.shared_storage = SharedStorage

        if cutlass.const_expr(self.enable_lora):
            la_smem_layout_staged_arg = self.la_smem_layout_staged
            lu_smem_layout_staged_arg = self.lu_smem_layout_staged
        else:
            la_smem_layout_staged_arg = None
            lu_smem_layout_staged_arg = None

        self.kernel(
            tiled_mma, tiled_mma_sfb, tiled_mma_lora,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa,
            tma_atom_sfb, tma_tensor_sfb,
            tma_atom_la, tma_tensor_la,
            tma_atom_lu, tma_tensor_lu,
            tma_atom_c, tma_tensor_c,
            self.cluster_layout_vmnk, self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged,
            la_smem_layout_staged_arg, lu_smem_layout_staged_arg,
            self.c_smem_layout_staged,
            self.epi_tile,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # -------- device kernel --------

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tiled_mma_lora,    # cute.TiledMma | None
        tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom, mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom, mSFB_nkl: cute.Tensor,
        tma_atom_la, mLA_mkl,    # None when enable_lora=False
        tma_atom_lu, mLU_nkl,
        tma_atom_c: cute.CopyAtom, mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        la_smem_layout_staged,
        lu_smem_layout_staged,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            if cutlass.const_expr(self.enable_lora):
                cpasync.prefetch_descriptor(tma_atom_la)
                cpasync.prefetch_descriptor(tma_atom_lu)
            cpasync.prefetch_descriptor(tma_atom_c)

        # Non-persistent: one CTA per output tile. Block coord gives the tile.
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        # In cluster=(1,1), block_in_cluster_coord is degenerate
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # MMA tile coord (M, N, L); no persistent scheduler, take directly
        # from grid.
        mma_tile_coord_mnl = (
            bidx // cute.size(tiled_mma.thr_id.shape),
            bidy,
            bidz,
        )

        # Allocate shared storage and pipelines
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # In 2-CTA clusters the pair's TMA loads converge on the same
        # mbarrier — consumer thread count covers both CTAs when mcast
        # is active; for the degenerate (2,1) pair no mcast, thread
        # count stays 1.
        num_ab_tma_producers = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_ab_tma_producers,
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Epilogue consumers scale by 2 for 2-CTA (both CTAs in the pair
        # drain the acc tmem in parallel).
        num_acc_consumer_threads = (
            self.threads_per_warp * len(self.epilog_warp_id)
            * (2 if self.use_2cta_instrs else 1)
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # LoRA prolog pipeline: 1-stage, TMA (load warp) → UMMA (mma warp).
        # Producer = load warp, consumer = mma warp — mirrors ab_pipeline
        # wiring but single-use (one producer commit, one consumer wait).
        lora_pipeline = None
        if cutlass.const_expr(self.enable_lora):
            lora_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            lora_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 1,
            )
            lora_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.lora_full_mbar_ptr.data_ptr(),
                num_stages=self.num_lora_stage,
                producer_group=lora_producer_group,
                consumer_group=lora_consumer_group,
                tx_count=self.num_lora_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # SMEM tensors
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sLA = None
        sLU = None
        if cutlass.const_expr(self.enable_lora):
            sLA = storage.sLA.get_tensor(
                la_smem_layout_staged.outer,
                swizzle=la_smem_layout_staged.inner,
            )
            sLU = storage.sLU.get_tensor(
                lu_smem_layout_staged.outer,
                swizzle=lu_smem_layout_staged.inner,
            )

        # Multicast masks — active when 2-CTA or when cluster > 1 along
        # M / N. Mirrors `dense_blockscaled_gemm_persistent.py`. For
        # v0 Phase-1 2-CTA (cluster (2, 1), thr_id=2) the masks here
        # cover the CtaGroup.TWO pair's shared-smem convergence.
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(
            self.is_a_mcast or self.is_b_mcast or self.use_2cta_instrs
        ):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2,
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1,
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2,
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk,
                mcast_mode=1,
            )

        # Partition global tensors
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None))
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # TMA partitions
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout,
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
        )
        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa, block_in_cluster_coord_vmnk[2], sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # SMEM/TMEM fragments
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage))

        # LoRA-side partitions + fragments. Built off tiled_mma_lora
        # (dense fp16/bf16). Shared-tmem: acc fragment is sourced from
        # the SAME tmem ptr as main — invariant checked at trace time by
        # tmp/verify_tmem_layout.py (tv_layout_C_tiled match).
        tCrLA = None
        tCrLU = None
        tCtAcc_lora_fake = None
        tALsLA = None
        tALgLA = None
        tBLsLU = None
        tBLgLU = None
        if cutlass.const_expr(self.enable_lora):
            gLA_mkl = cute.local_tile(
                mLA_mkl, cute.slice_(self.lora_mma_tiler, (None, 0, None)),
                (None, None, None))
            gLU_nkl = cute.local_tile(
                mLU_nkl, cute.slice_(self.lora_mma_tiler, (0, None, None)),
                (None, None, None))
            thr_mma_lora = tiled_mma_lora.get_slice(mma_tile_coord_v)
            tCgLA = thr_mma_lora.partition_A(gLA_mkl)
            tCgLU = thr_mma_lora.partition_B(gLU_nkl)
            # LA TMA partition (same pattern as main A)
            la_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            tALsLA, tALgLA = cpasync.tma_partition(
                tma_atom_la, block_in_cluster_coord_vmnk[2], la_cta_layout,
                cute.group_modes(sLA, 0, 3), cute.group_modes(tCgLA, 0, 3),
            )
            lu_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            tBLsLU, tBLgLU = cpasync.tma_partition(
                tma_atom_lu, block_in_cluster_coord_vmnk[1], lu_cta_layout,
                cute.group_modes(sLU, 0, 3), cute.group_modes(tCgLU, 0, 3),
            )
            tCrLA = tiled_mma_lora.make_fragment_A(sLA)
            tCrLU = tiled_mma_lora.make_fragment_B(sLU)
            acc_shape_lora = tiled_mma_lora.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_lora_fake = tiled_mma_lora.make_fragment_C(
                cute.append(acc_shape_lora, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # =========================================================
        # TMA load warp (warp 5)
        # =========================================================
        if warp_idx == self.tma_warp_id:
            # LoRA prolog first — single-buffer TMA load of LA/LU, fires
            # before the main K-loop so the MMA warp can issue the first
            # interleaved LoRA atom as soon as k_atom=0 main atom retires.
            if cutlass.const_expr(self.enable_lora):
                lora_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_lora_stage)
                lora_pipeline.producer_acquire(lora_producer_state)
                tALgLA_slice = tALgLA[
                    (None, mma_tile_coord_mnl[0], 0, mma_tile_coord_mnl[2])]
                tBLgLU_slice = tBLgLU[
                    (None, mma_tile_coord_mnl[1], 0, mma_tile_coord_mnl[2])]
                cute.copy(
                    tma_atom_la, tALgLA_slice,
                    tALsLA[(None, lora_producer_state.index)],
                    tma_bar_ptr=lora_pipeline.producer_get_barrier(
                        lora_producer_state),
                    mcast_mask=None,
                )
                cute.copy(
                    tma_atom_lu, tBLgLU_slice,
                    tBLsLU[(None, lora_producer_state.index)],
                    tma_bar_ptr=lora_pipeline.producer_get_barrier(
                        lora_producer_state),
                    mcast_mask=None,
                )
                # producer_commit fires automatically via tx_count in the
                # mbarrier — no explicit advance/commit needed for a
                # single-stage 1-iteration pipeline.

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage)

            tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
            tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
            tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
            tBgSFB_slice = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

            peek_ab_empty = cutlass.Boolean(1)
            ab_producer_state.reset_count()
            if ab_producer_state.count < k_tile_cnt:
                peek_ab_empty = ab_pipeline.producer_try_acquire(ab_producer_state)

            for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty)

                cute.copy(
                    tma_atom_a, tAgA_slice[(None, ab_producer_state.count)],
                    tAsA[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    mcast_mask=a_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_b, tBgB_slice[(None, ab_producer_state.count)],
                    tBsB[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    mcast_mask=b_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_sfa, tAgSFA_slice[(None, ab_producer_state.count)],
                    tAsSFA[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    mcast_mask=sfa_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_sfb, tBgSFB_slice[(None, ab_producer_state.count)],
                    tBsSFB[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    mcast_mask=sfb_full_mcast_mask,
                )

                ab_producer_state.advance()
                peek_ab_empty = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty = ab_pipeline.producer_try_acquire(ab_producer_state)

            ab_pipeline.producer_tail(ab_producer_state)

        # =========================================================
        # MMA warp (warp 4)
        # =========================================================
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            (tiled_copy_s2t_sfa, tCsSFA_s2t, tCtSFA_s2t) = \
                self._mainloop_s2t_copy(sSFA, tCtSFA)
            (tiled_copy_s2t_sfb, tCsSFB_s2t, tCtSFB_s2t) = \
                self._mainloop_s2t_copy(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage)

            # Single tile per CTA (non-persistent).  Stage index = 0 since
            # each CTA uses acc_stage 0 — no rotation needed.
            acc_stage_index = acc_producer_state.index
            tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

            # Shared-tmem LoRA acc. Same ptr as main, TV-layout match
            # verified at trace time. Wait for LoRA prolog before first
            # LoRA issue — load warp fired the TMA right before the main
            # AB loop started, so typical latency is tens of cycles.
            tCtAcc_lora = None
            lora_consumer_state = None
            if cutlass.const_expr(self.enable_lora):
                tCtAcc_lora_base = cute.make_tensor(
                    acc_tmem_ptr, tCtAcc_lora_fake.layout)
                tCtAcc_lora = tCtAcc_lora_base[(None, None, None, acc_stage_index)]
                lora_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_lora_stage)
                if is_leader_cta:
                    lora_pipeline.consumer_wait(lora_consumer_state)
                tiled_mma_lora.set(tcgen05.Field.ACCUMULATE, True)

            peek_ab_full = cutlass.Boolean(1)
            ab_consumer_state.reset_count()
            if is_leader_cta and ab_consumer_state.count < k_tile_cnt:
                peek_ab_full = ab_pipeline.consumer_try_wait(ab_consumer_state)

            if is_leader_cta:
                acc_pipeline.producer_acquire(acc_producer_state)

            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

            num_kblocks = cute.size(tCrA, mode=[2])
            # β-interleave state (per design §2). K_atoms / R_atoms and
            # stride computed once up front — R_atoms is compile-time,
            # k_atom_total is dynamic (depends on K).
            r_next = cutlass.Int32(0)
            next_lora_at = cutlass.Int32(0)
            k_atom_flat = cutlass.Int32(0)
            if cutlass.const_expr(self.enable_lora):
                k_atoms_total = k_tile_cnt * num_kblocks
                stride = k_atoms_total // self.R_atoms

            for k_tile in range(k_tile_cnt):
                if is_leader_cta:
                    ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full)

                    s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                    cute.copy(
                        tiled_copy_s2t_sfa,
                        tCsSFA_s2t[s2t_stage_coord], tCtSFA_s2t,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfb,
                        tCsSFB_s2t[s2t_stage_coord], tCtSFB_s2t,
                    )

                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                        sf_kblock_coord = (None, None, kblock_idx)
                        tiled_mma.set(tcgen05.Field.SFA,
                                      tCtSFA[sf_kblock_coord].iterator)
                        tiled_mma.set(tcgen05.Field.SFB,
                                      tCtSFB[sf_kblock_coord].iterator)
                        cute.gemm(
                            tiled_mma, tCtAcc,
                            tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # β-interleave: sprinkle one LoRA atom every `stride`
                        # main atoms. tcgen05 issue queue is in-order per
                        # CTA, so the LoRA atom here sees the main atom's
                        # acc write.
                        if cutlass.const_expr(self.enable_lora):
                            if r_next < self.R_atoms and k_atom_flat == next_lora_at:
                                lora_kblock_coord = (None, None, r_next, 0)
                                cute.gemm(
                                    tiled_mma_lora, tCtAcc_lora,
                                    tCrLA[lora_kblock_coord],
                                    tCrLU[lora_kblock_coord],
                                    tCtAcc_lora,
                                )
                                r_next += 1
                                next_lora_at += stride
                            k_atom_flat += 1

                    ab_pipeline.consumer_release(ab_consumer_state)

                ab_consumer_state.advance()
                peek_ab_full = cutlass.Boolean(1)
                if is_leader_cta and ab_consumer_state.count < k_tile_cnt:
                    peek_ab_full = ab_pipeline.consumer_try_wait(ab_consumer_state)

            if is_leader_cta:
                acc_pipeline.producer_commit(acc_producer_state)
            acc_producer_state.advance()
            if is_leader_cta:
                acc_pipeline.producer_tail(acc_producer_state)
                if cutlass.const_expr(self.enable_lora):
                    lora_pipeline.consumer_release(lora_consumer_state)

        # =========================================================
        # Epilogue warps (warps 0–3)
        # =========================================================
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx
            (tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc) = \
                self._epilog_tmem_copy(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, self.use_2cta_instrs,
                )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            (tiled_copy_r2s, tRS_rC, tRS_sC) = \
                self._epilog_smem_copy(tiled_copy_t2r, tTR_rC, epi_tidx, sC)
            (tma_atom_c_, bSG_sC, bSG_gC_partitioned) = \
                self._epilog_gmem_copy(epi_tidx, tma_atom_c, tCgC, epi_tile, sC)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
            acc_stage_index = acc_consumer_state.index
            tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

            acc_pipeline.consumer_wait(acc_consumer_state)

            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            for subtile_idx in cutlass.range(subtile_cnt):
                tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                acc_vec = acc_vec.to(self.c_dtype)
                tRS_rC.store(acc_vec)

                c_buffer = subtile_idx % self.num_c_stage
                cute.copy(
                    tiled_copy_r2s, tRS_rC,
                    tRS_sC[(None, None, None, c_buffer)],
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epilog_sync_barrier.arrive_and_wait()

                if warp_idx == self.epilog_warp_id[0]:
                    cute.copy(
                        tma_atom_c_,
                        bSG_sC[(None, c_buffer)],
                        bSG_gC[(None, subtile_idx)],
                    )
                    c_pipeline.producer_commit()
                    c_pipeline.producer_acquire()
                self.epilog_sync_barrier.arrive_and_wait()

            acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            c_pipeline.producer_tail()

    # -------- helpers ported from persistent.py --------

    def _mainloop_s2t_copy(
        self, sSF: cute.Tensor, tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_s2t_)
        tCtSF_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_s2t, tCtSF_s2t

    def _epilog_tmem_copy(
        self, tidx: cutlass.Int32, tAcc: cute.Tensor, gC_mnl: cute.Tensor,
        epi_tile: cute.Tile, use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk, self.c_layout, self.c_dtype,
            self.acc_dtype, epi_tile, use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def _epilog_smem_copy(
        self, tiled_copy_t2r: cute.TiledCopy, tTR_rC: cute.Tensor,
        tidx: cutlass.Int32, sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def _epilog_gmem_copy(
        self, tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor, epi_tile: cute.Tile, sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tma_atom_c = atom
        sC_p = cute.group_modes(sC, 0, 2)
        gC_p = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1), sC_p, gC_p,
        )
        return tma_atom_c, bSG_sC, bSG_gC
