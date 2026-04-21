"""gemm_w4a4 CuTe DSL kernel (Blackwell SM_100/SM_103).

Host-facing contract (v0 — main NVFP4 only; LoRA / wcscales / bias /
next-layer quant land in v1–v3):

    launch(act, wgt, ascales, wscales, *, out_dtype=torch.float16) -> out

Inputs (NVFP4-packed, as produced by
`triton_kernels/quantize_w4a4_act_fuse_lora/`):

    act       [M, K // 2]    uint8           two E2M1 nibbles per byte
    wgt       [N, K // 2]    uint8           two E2M1 nibbles per byte
    ascales   [K // 16, M]   fp8_e4m3fn      per-16-K-block act scale
    wscales   [K // 16, N]   fp8_e4m3fn      per-16-K-block wgt scale

Output:
    out       [M, N]         out_dtype (fp16 or bf16)

Device body ported from `tmp/cutlass/examples/python/CuTeDSL/blackwell/
dense_blockscaled_gemm_persistent.py` (stable API). Stripped:
persistent TileScheduler, clusters > 1, TMA multicast, overlapping_accum,
tile_n ∈ {64, 192} SFB-shift hacks. v0 uses 128×128 tile (2 acc stages,
no overlap) on a single-CTA (1SM) MMA atom.

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
# v0: 1SM 128×128 MMA tile.  Promote to 128×256 or 2SM 256×256 in v2+ tune.
# 128×128 → num_acc_stage = 2, no overlapping_accum branch. Simplest
# correct first pass; plenty of room to tune once v1 (LoRA β) lands.
_MMA_INST_MN: Tuple[int, int] = (128, 128)
_CLUSTER_SHAPE_MN: Tuple[int, int] = (1, 1)
_SF_VEC_SIZE: int = 16                 # NVFP4 block size

# --- compile cache ---------------------------------------------------------
_COMPILED_CACHE: dict[tuple, object] = {}


# --- torch → cute bridging ------------------------------------------------
def _check_inputs(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    out_dtype: torch.dtype,
) -> Tuple[int, int, int]:
    """Validate layout + dtype. Returns (M, N, K)."""
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

    assert M % _MMA_INST_MN[0] == 0, f"M ({M}) must be a multiple of {_MMA_INST_MN[0]}"
    assert N % _MMA_INST_MN[1] == 0, f"N ({N}) must be a multiple of {_MMA_INST_MN[1]}"
    assert K % 32 == 0, f"K ({K}) must be a multiple of 32 for NVFP4 alignment"
    return M, N, K


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


def _c_ptr(out: torch.Tensor) -> "cute.Pointer":
    dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[out.dtype]
    return make_ptr(dtype, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def _cutlass_c_dtype(out_dtype: torch.dtype) -> Type[cutlass.Numeric]:
    return {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[out_dtype]


def launch(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """v0: y = scaled_mma(act_nvfp4, wgt_nvfp4) — no LoRA, no affine."""
    M, N, K = _check_inputs(act, wgt, ascales, wscales, out_dtype)
    out = torch.empty((M, N), dtype=out_dtype, device=act.device)

    cache_key = (_cutlass_c_dtype(out_dtype), _MMA_INST_MN, _SF_VEC_SIZE, _CLUSTER_SHAPE_MN)
    compiled = _COMPILED_CACHE.get(cache_key)
    if compiled is None:
        compiled = _compile_v0(c_dtype=_cutlass_c_dtype(out_dtype))
        _COMPILED_CACHE[cache_key] = compiled

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        _fp4_ptr(act),
        _fp4_ptr(wgt),
        _sf_ptr(ascales),
        _sf_ptr(wscales),
        _c_ptr(out),
        (cutlass.Int32(M), cutlass.Int32(N), cutlass.Int32(K), cutlass.Int32(1)),
        stream,
    )
    return out


# --- JIT compile ----------------------------------------------------------


def _compile_v0(*, c_dtype: Type[cutlass.Numeric]):
    """AOT-compile the v0 scaled-MMA kernel; shapes are runtime."""
    kernel_obj = Sm100GemmW4A4V0(
        sf_vec_size=_SF_VEC_SIZE,
        mma_tiler_mn=_MMA_INST_MN,
        cluster_shape_mn=_CLUSTER_SHAPE_MN,
        ab_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E4M3FN,
        c_dtype=c_dtype,
    )
    a_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=32)
    b_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfa_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfb_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    dummy_stream = cuda_drv.CUstream(0)

    return cute.compile(
        kernel_obj,
        a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr,
        (cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)),
        dummy_stream,
    )


# --- device kernel --------------------------------------------------------
class Sm100GemmW4A4V0:
    """v0 device-side kernel: main NVFP4 scaled-MMA only, non-persistent.

    Layout fixed: A K-major [M, K], B K-major [N, K], C row-major [M, N],
    all batch L = 1.  Cluster (1, 1), single-CTA 1SM 128×128 MMA atom.
    Warp-spec split = 1 TMA load (warp 5) + 1 MMA (warp 4) + 4 epilogue
    (warps 0–3) = 6 warps = 192 threads.

    Porting base: dense_blockscaled_gemm_persistent.py (stable API).
    Stripped:
      - PersistentTileScheduler + outer `while work_tile.is_valid_tile`
      - cluster_shape > (1,1) and multicast masks
      - use_2cta_instrs path (cta_group = ONE hard-coded)
      - overlapping_accum (num_acc_stage = 2, cta_tile_n = 128)
      - tile_n ∈ {64, 192} SFB-shift hacks (we're 128)
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
    ):
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        assert mma_tiler_mn[0] == 128, "v0 is 1SM-only; 2SM (tile_m=256) lands later"
        assert mma_tiler_mn[1] in (128, 256), "v0 supports tile_n ∈ {128, 256}"
        assert cluster_shape_mn == (1, 1), "v0 is non-clustered"
        self.use_2cta_instrs = False
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)   # K filled in setup
        self.cta_group = tcgen05.CtaGroup.ONE

        self.ab_dtype = ab_dtype
        self.sf_dtype = sf_dtype
        self.c_dtype = c_dtype
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
        # MMA instruction shapes
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0],   # 1SM, so no /2
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

        # No cluster, no multicast — degenerate vmnk = (v, 1, 1, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Stage counts
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.ab_dtype,
            self.epi_tile, self.c_dtype, self.c_layout,
            self.sf_dtype, self.sf_vec_size,
            self.smem_capacity, self.occupancy,
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

        return tiled_mma, tiled_mma_sfb

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

        tiled_mma, tiled_mma_sfb = self._setup_attributes()
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

        grid = self._compute_grid_nonpersistent(c_tensor, self.cta_tile_shape_mnk)

        # SharedStorage
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

        self.kernel(
            tiled_mma, tiled_mma_sfb,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa,
            tma_atom_sfb, tma_tensor_sfb,
            tma_atom_c, tma_tensor_c,
            self.cluster_layout_vmnk, self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged,
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
        tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom, mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom, mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom, mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
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
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 1  # single producer CTA, no mcast
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
        num_acc_consumer_threads = self.threads_per_warp * len(self.epilog_warp_id)
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

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=False,
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

        # Cluster = (1,1) → no mcast masks
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None

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

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # =========================================================
        # TMA load warp (warp 5)
        # =========================================================
        if warp_idx == self.tma_warp_id:
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

            peek_ab_full = cutlass.Boolean(1)
            ab_consumer_state.reset_count()
            if ab_consumer_state.count < k_tile_cnt:
                peek_ab_full = ab_pipeline.consumer_try_wait(ab_consumer_state)

            acc_pipeline.producer_acquire(acc_producer_state)

            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

            for k_tile in range(k_tile_cnt):
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

                num_kblocks = cute.size(tCrA, mode=[2])
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

                ab_pipeline.consumer_release(ab_consumer_state)

                ab_consumer_state.advance()
                peek_ab_full = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt:
                    peek_ab_full = ab_pipeline.consumer_try_wait(ab_consumer_state)

            acc_pipeline.producer_commit(acc_producer_state)
            acc_producer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

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
