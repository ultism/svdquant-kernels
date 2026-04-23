"""gemm_w4a4 v2 FA4 skeleton (Blackwell SM_100/SM_103).

Scope: main NVFP4 scaled-MMA + β-interleaved LoRA + per-column
`* wcscales + bias` epilogue. No next-layer requant (that's v3).

Forked from `kernel_v0_fa4.py` as the v1 snapshot freeze point — the
v0/v1 device body was stable (smoke 22/22 on 1-CTA + 2-CTA × fp16/bf16
× R ∈ {32,128,256} where smem allows). Adding `wcscales + bias` touches
the epilogue warp and changes the `pipeline_acc` consumer side, so the
v0/v1 file stays frozen as the reference implementation; all epilogue
changes land here.

Why a new file is kept: the previous `kernel.py` is pinned to
non-persistent 1-tile-per-CTA + stock `cutlass.pipeline.PipelineState`,
which is known to hang on 2-CTA persistent — our state space (pipeline
stages × 2CTA pair barriers × persistent tile loop × LoRA β × epilogue
chain) exceeds what the branching stock state can track cleanly across
tile boundaries. This module rebuilds the device body on the FA4 pattern:

- 3 warp-specialized roles (`load` / `mma` / `epilogue`) — each warp
  drives its own `PipelineStateSimple`, monotonically incremented, so
  pipeline state never resets at tile boundaries.
- 2 pipelines: `pipeline_aw` (TmaUmma, multi-stage, main K-loop; all
  four TMAs `act + wgt + sfa + sfb` share one barrier via one
  aggregate `tx_count` = `num_tma_load_bytes`) and `pipeline_acc`
  (UmmaAsync, single-stage for v0 — grows when v2 overlaps accum).
- `StaticPersistentTileScheduler` equivalent, inline — grid clamped to
  (sm_count aligned to cluster_m), each CTA advances by `grid_dim` (or
  `cluster_dim` under 2-CTA) until tile_idx exceeds the total.

Reference skeleton: `tmp/flash-attention/flash_attn/cute/` (FA4, Tri
Dao). See `README.md` → "Architecture (FA4-derived 3-pipeline /
3-warp)" and `AI/DEBUG_2CTA.md` for the 2-CTA pitfall hierarchy.

Host-facing contract (v2):

    launch_v2(act, wgt, ascales, wscales,
              lora_act_in=None, lora_up=None,
              wcscales=None, bias=None,
              *, out_dtype=torch.float16, use_2cta=False) -> out

v2 work (not yet implemented in this snapshot — file is a literal copy
of v1_fa4 with names changed; the wcscales/bias kwargs above are the
target API, not the current signature):

- Epilogue warp reads per-column `wcscales [N]` and optional `bias [N]`,
  multiplies acc then adds bias before `tmem → gmem` store.
- `pipeline_acc` may grow to 2-stage to overlap the epilogue MUL/ADD
  with the next tile's main MMA (measure first; v0/v1 had single-stage
  acc + bare `producer_acquire` tail).

NVFP4-packed inputs identical to `kernel.py::launch` — this module
reuses the same `_check_inputs`, pointer builders, and scale repacking.
"""
from __future__ import annotations

import os
from typing import Tuple, Type, Union

import torch

os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import cuda.bindings.driver as cuda_drv                  # noqa: E402
import cutlass                                           # noqa: E402
import cutlass.cute as cute                              # noqa: E402
import cutlass.pipeline as pipeline                      # noqa: E402
import cutlass.utils as utils                            # noqa: E402
import cutlass.utils.blackwell_helpers as sm100_utils    # noqa: E402
import cutlass.utils.blockscaled_layout as blockscaled_utils  # noqa: E402
from cutlass import Int32                                # noqa: E402
from cutlass.cute.nvgpu import cpasync, tcgen05          # noqa: E402
from cutlass.cute.runtime import make_ptr                # noqa: E402
from cutlass.pipeline import (                           # noqa: E402
    pipeline_init_arrive, pipeline_init_wait,
    PipelineUserType,
)

from cute_kernels.gemm_w4a4.kernel import (              # noqa: E402
    _SF_VEC_SIZE, _LORA_INST_K,
    _TILER_SMALL_M, _TILER_DEFAULT, _TILER_2CTA, _TILER_SMALL_M_THRESHOLD,
    _check_inputs, _fp4_ptr, _sf_ptr, _c_ptr, _lora_ptr,
    _repack_scales_cutlass_atom, _cutlass_c_dtype,
)
from cute_kernels.gemm_w4a4._pipeline_simple import (    # noqa: E402
    PipelineStateSimple, make_pipeline_state_simple,
    make_pipeline_state_from_index_phase,
)


# --- compile cache --------------------------------------------------------
_COMPILED_CACHE: dict[tuple, object] = {}


# 2-CTA tile_n=256 is also supported (num_acc_stage stays 1; fits tmem:
# 256 acc + 16 sfa + 32 sfb = 304 cols of 512). Same-session A/B on
# B200 showed ≤+4% MFU on big shapes and −3..11% on small M / small
# K·N — net not worth making default. Opt-in via `launch_v2(tiler_mn=
# (256, 256))`. The dominant MFU gap vs CUTLASS (~20pp at matched
# tile) is elsewhere (pipeline discipline / TMA / s2t) — task #41.
def _pick_tiler_v2(M: int, use_2cta: bool, R: int = 0) -> Tuple[int, int]:
    if use_2cta:
        return _TILER_2CTA
    # v1: high-R LoRA smem (LA + LU) overflows 228 KB at tile 128×256
    # (LU alone = 128 KB at R=256). Fall back to 128×128 on R ≥ 256 so
    # LU halves to 64 KB. Mirrors kernel.py::_pick_tiler; production
    # R ≤ 128 so this only fires on the stress-test shape.
    if R >= 256:
        return _TILER_SMALL_M
    return _TILER_SMALL_M if M <= _TILER_SMALL_M_THRESHOLD else _TILER_DEFAULT


def launch_v2(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: "torch.Tensor | None" = None,
    lora_up: "torch.Tensor | None" = None,
    *,
    out_dtype: torch.dtype = torch.float16,
    use_2cta: bool = False,
    tiler_mn: Tuple[int, int] | None = None,
) -> torch.Tensor:
    """v0: y = scaled_mma(act_nvfp4, wgt_nvfp4). No LoRA, no affine.
    v1 (both lora_* provided): y += lora_act_in @ lora_up.T, β-interleaved
    into the main K-loop per design §2.

    `tiler_mn` overrides the shape-adaptive pick — bench hook, do not use
    in production paths (no validation against smem / tmem budgets)."""
    M, N, K, R, _ = _check_inputs(
        act, wgt, ascales, wscales, out_dtype,
        lora_act_in=lora_act_in, lora_up=lora_up, use_2cta=use_2cta,
    )
    enable_lora = lora_act_in is not None
    tiler = tiler_mn if tiler_mn is not None else _pick_tiler_v2(M, use_2cta, R)
    assert M % tiler[0] == 0 and N % tiler[1] == 0
    cluster_shape_mn = (2, 1) if use_2cta else (1, 1)
    out = torch.empty((M, N), dtype=out_dtype, device=act.device)

    cache_key = (
        _cutlass_c_dtype(out_dtype), tiler, _SF_VEC_SIZE, cluster_shape_mn,
        enable_lora, R,
    )
    compiled = _COMPILED_CACHE.get(cache_key)
    if compiled is None:
        compiled = _compile_v2(
            c_dtype=_cutlass_c_dtype(out_dtype), tiler_mn=tiler,
            cluster_shape_mn=cluster_shape_mn,
            enable_lora=enable_lora, R=R,
        )
        _COMPILED_CACHE[cache_key] = compiled

    ascales_atom = _repack_scales_cutlass_atom(ascales)
    wscales_atom = _repack_scales_cutlass_atom(wscales)

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    # LoRA host prep: lora_act_in may arrive fp32; dense MMA atom reads
    # fp16/bf16 from smem, so pre-cast at the boundary (≤ M·R·4 B, sub-µs
    # for R ≤ 128). Kept for parity with kernel.py; revisit if load warp
    # becomes bottleneck.
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


def _compile_v2(
    *,
    c_dtype: Type[cutlass.Numeric],
    tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    enable_lora: bool = False,
    R: int = 0,
):
    kernel_obj = Sm100GemmW4A4V2FA4(
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
    la_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    lu_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    dummy_stream = cuda_drv.CUstream(0)

    return cute.compile(
        kernel_obj,
        a_ptr, b_ptr, sfa_ptr, sfb_ptr, la_ptr, lu_ptr, c_ptr,
        (cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)),
        dummy_stream,
    )


class Sm100GemmW4A4V2FA4:
    """v0 FA4-derived device kernel: main NVFP4 scaled-MMA only.

    Layout: A K-major [M, K], B K-major [N, K], C row-major [M, N], L=1.
    Warp-spec: 1 load (warp 5) + 1 MMA (warp 4) + 4 epilogue (warps 0–3)
    = 192 threads. Tile loop persistent (grid clamped to sm_count).

    Invariant: no tile-boundary state reset. Each warp's
    `PipelineStateSimple` just increments; `index`/`phase` are pure
    divmod. Stock `cutlass.pipeline.PipelineState` has a branching
    `advance()` that drifts under persistent iteration + 2-CTA clusters;
    we sidestep it.
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
        assert mma_tiler_mn[0] in (128, 256), "tile_m ∈ {128, 256}"
        assert mma_tiler_mn[1] in (128, 256), "tile_n ∈ {128, 256}"
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        if self.use_2cta_instrs:
            assert cluster_shape_mn == (2, 1), \
                f"2-CTA requires cluster_shape_mn=(2, 1); got {cluster_shape_mn}"
            # Phase 2: 2-CTA supports both tile_n=128 and tile_n=256.
            # num_acc_stage stays 1 (single-stage acc) — true overlapping_accum
            # (num_acc_stage=2 ping-pong) requires multi-stage acc state.
        else:
            assert cluster_shape_mn == (1, 1)
        self.cluster_shape_mn = cluster_shape_mn
        self.cta_group_size = 2 if self.use_2cta_instrs else 1
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.ab_dtype = ab_dtype
        self.sf_dtype = sf_dtype
        self.c_dtype = c_dtype
        # LoRA config: R = 0 disables v1 path. R_atoms = R // 16 (dense
        # tcgen05 fp16/bf16 K-instr). Baked compile-time so the LoRA
        # atom loop can unroll_full. 2-CTA LoRA TV-layout match verified
        # via tmp/verify_tmem_layout.py (shared-tmem acc viable).
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
        self.a_major_mode = tcgen05.OperandMajorMode.K
        self.b_major_mode = tcgen05.OperandMajorMode.K
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
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

    # -------- host-side setup (shape identical to kernel.py, LoRA pruned) --------

    def _setup_attributes(self):
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size,
            self.cta_group, self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size,
            tcgen05.CtaGroup.ONE, self.mma_inst_shape_mn_sfb,
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0], self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0], self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1], self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1], self.mma_tiler_sfb[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, self.use_2cta_instrs,
            self.c_layout, self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # LoRA smem (LA + LU, single-buffer) deducted from budget so
        # num_ab_stage shrinks. LA = [M, R] * dtype bytes, LU = [N, R] * dtype bytes.
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
        assert self.num_ab_stage >= 2, \
            f"num_ab_stage = {self.num_ab_stage} — smem budget too tight"
        # v0 single-stage acc (no overlapping_accum). v2/v3 may lift this.
        self.num_acc_stage = 1

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

        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
        )

        # LoRA dense MMA atom + LA/LU staged smem layouts. Shared-tmem acc:
        # LoRA MMA's partition_C writes to the same tmem region as main —
        # TV-layout match verified under 2-CTA via tmp/verify_tmem_layout.py.
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
            self.num_lora_stage = 1   # single-buffer prolog (README §Pipelines)
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

    # -------- persistent grid --------

    def _compute_grid_persistent(
        self, c_tensor: cute.Tensor,
    ) -> Tuple[Int32, Int32, Int32, Int32]:
        """Grid clamped to `sm_count` (aligned to `cluster_shape_m`).
        Returns `(grid_x, 1, 1, total_tiles_cluster)` — grid_x launched,
        `total_tiles_cluster` handed to the device so `is_valid_tile`
        can be evaluated per-iteration.

        `total_tiles_cluster` counts clusters (M_tiles / cluster_m × N_tiles ×
        L), not individual CTAs — the advance stride is `cluster_dim`
        under 2-CTA, so the while-condition compares cluster indices.
        Tile by `mma_tiler` (spans the whole cluster in M) so the count
        matches the device-side `gC_mnl.num_m`."""
        mma_c_shape = cute.slice_(self.mma_tiler, (None, None, 0))
        gc = cute.zipped_divide(c_tensor, tiler=mma_c_shape)
        num_m_cluster, num_n, num_l = gc[(0, (None, None, None))].shape
        total_tiles_cluster = num_m_cluster * num_n * num_l
        cluster_m = self.cluster_shape_mn[0]
        hw = utils.HardwareInfo()
        sm_count = hw.get_device_multiprocessor_count()
        max_ctas = (sm_count // cluster_m) * cluster_m
        grid_x = cutlass.min(max_ctas, total_tiles_cluster * cluster_m)
        return (Int32(grid_x), Int32(1), Int32(1),
                Int32(total_tiles_cluster))

    # -------- entrypoint --------

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        la_ptr: cute.Pointer,
        lu_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_mnkl: Tuple[Int32, Int32, Int32, Int32],
        stream,
    ):
        m, n, k, l = problem_mnkl

        a_layout = cute.make_ordered_layout(
            (cute.assume(m, 32), k, l), order=(1, 0, 2))
        b_layout = cute.make_ordered_layout(
            (cute.assume(n, 32), k, l), order=(1, 0, 2))
        c_layout = cute.make_ordered_layout(
            (m, cute.assume(n, 32), l), order=(1, 0, 2))
        a_tensor = cute.make_tensor(a_ptr, a_layout)
        b_tensor = cute.make_tensor(b_ptr, b_layout)
        c_tensor = cute.make_tensor(c_ptr, c_layout)

        sfa_tensor = cute.make_tensor(
            sfa_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size),
        )
        sfb_tensor = cute.make_tensor(
            sfb_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size),
        )

        tiled_mma, tiled_mma_sfb, tiled_mma_lora = self._setup_attributes()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

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
        # Total TMA bytes per aw barrier: all 4 tensors share one barrier.
        # Multiply by `atom_thr_size` (== 2 under 2-CTA): the pair's two
        # CTAs both sign the same cluster barrier, so tx_count covers
        # both contributions — bake the ×cta_group_size in here at
        # pipeline creation, never at runtime (DEBUG_2CTA step 5).
        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.ab_dtype, a_smem_layout)
            + cute.size_in_bytes(self.ab_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        ) * atom_thr_size

        # LoRA TMA atoms (when enabled). LA = [M, R] K-major, LU = [N, R]
        # K-major; L=1, K=R (baked compile-time). Partitioned via
        # tiled_mma_lora so acc fragment TV-layout matches main. Tx_count
        # carries the same ×cta_group_size multiplier as main A/B.
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

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor, epi_smem_layout, self.epi_tile,
        )

        grid_x, grid_y, grid_z, total_tiles_cluster = \
            self._compute_grid_persistent(c_tensor)

        # SharedStorage. LoRA fields conditional; CuTe DSL builds the
        # struct body at trace time so fields only materialize when
        # enable_lora. Mirrors kernel.py:798-867.
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
            total_tiles_cluster,
        ).launch(
            grid=(grid_x, grid_y, grid_z),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # -------- device kernel (FA4 3-warp split) --------

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
        total_tiles_cluster: Int32,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            if cutlass.const_expr(self.enable_lora):
                cpasync.prefetch_descriptor(tma_atom_la)
                cpasync.prefetch_descriptor(tma_atom_lu)
            cpasync.prefetch_descriptor(tma_atom_c)

        # Cluster coordinate. Under 2-CTA, block_idx within the cluster
        # (0 or 1) splits the M-tile into two halves handled by the pair.
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # Initial cluster tile index. Under cluster_m==1, that's just
        # block_idx[0]; under cluster_m>1, it's cluster_idx[0] (all CTAs
        # in a cluster share the tile coord).
        if cutlass.const_expr(self.cluster_shape_mn[0] == 1):
            initial_tile_idx = cute.arch.block_idx()[0]
        else:
            initial_tile_idx = cute.arch.cluster_idx()[0]
        initial_tile_idx = cute.arch.make_warp_uniform(Int32(initial_tile_idx))

        mma_tile_coord_v = cta_rank_in_cluster % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Allocate shared storage.
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipelines. `pipeline_aw` is multi-stage TmaUmma; `pipeline_acc`
        # is single-stage UmmaAsync (v0 — grows when v2 overlaps accum).
        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_ab_tma_producers = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_ab_tma_producers,
        )
        pipeline_aw = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_producer_group,
            consumer_group=ab_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        acc_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = (
            self.threads_per_warp * len(self.epilog_warp_id) * self.cta_group_size
        )
        acc_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads,
        )
        pipeline_acc = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_producer_group,
            consumer_group=acc_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # LoRA prolog pipeline: 1-stage PipelineTmaUmma, TMA (load warp) →
        # UMMA (mma warp). Per-tile fire + consume (LA/LU depend on the
        # tile's m/n coords). Under persistent iteration phase flips each
        # tile. `tx_count * atom_thr_size` bakes in the 2-CTA cluster
        # contribution — same pattern as pipeline_aw.
        pipeline_lora = None
        if cutlass.const_expr(self.enable_lora):
            lora_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            lora_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 1,
            )
            pipeline_lora = pipeline.PipelineTmaUmma.create(
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

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage))

        # LoRA-side partitions + fragments. Built off tiled_mma_lora (dense
        # fp16/bf16). Shared-tmem acc: tCtAcc_lora_fake's layout is used
        # later in the MMA warp to alias the same tmem as the main acc —
        # TV-layout match verified by tmp/verify_tmem_layout.py.
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

        # Per-MMA-tile count for the `tile_idx → (m_tile, n_tile, l_tile)`
        # decode. `gC_mnl` below is tiled by `mma_tiler` (= 256×128 under
        # 2-CTA), so its `num_m` axis counts *mma tiles*, not cta tiles;
        # each mma tile spans both CTAs in a 2-CTA cluster and
        # `partition_C` on the mma slot already hands each CTA its half.
        # Both CTAs in a cluster therefore use the **same** `m_tile`.
        #
        # Two divide-API traps rolled into one. (1) `local_tile.shape` is
        # flat `(tile_m, tile_n, num_m, num_n, num_l)`, not nested — use
        # `zipped_divide` and unpack the second mode for positional coord
        # access. (2) Tile by `mma_tiler[0]`, not `cta_tile_shape_mnk[0]`:
        # an earlier version divided by cta-tile (128) and added a V
        # offset to `m_tile`, which fired 2× out-of-range indices on any
        # 2-CTA shape with `num_m_cluster > 1` (= any shape beyond the
        # single-cluster M=256 smoke).
        _c_tile = cute.slice_(self.mma_tiler, (None, None, 0))
        _gc_zipped = cute.zipped_divide(mC_mnl, tiler=_c_tile)
        num_m_cluster, num_n, num_l = _gc_zipped[(0, (None, None, None))].shape
        tiles_per_l = num_m_cluster * num_n

        # =========================================================
        # LOAD warp — fires all four TMAs per K-block into pipeline_aw.
        # v1: also fires LoRA LA/LU prolog TMA per tile into pipeline_lora
        # (1-stage, phase flips each tile). README §Pipeline state.
        # =========================================================
        if warp_idx == self.load_warp_id:
            aw_producer_state = make_pipeline_state_simple(
                PipelineUserType.Producer, self.num_ab_stage)
            # Single-stage LoRA producer: phase starts at 1 (pre-armed
            # empty by pipeline_init_arrive). Flips each tile.
            lora_producer_phase = Int32(1)

            tile_idx = initial_tile_idx
            grid_stride = self._grid_stride()
            while tile_idx < total_tiles_cluster:
                l_tile = tile_idx // tiles_per_l
                mn_rem = tile_idx % tiles_per_l
                m_tile = mn_rem // num_n
                n_tile = mn_rem % num_n

                # LoRA prolog: fire LA/LU TMA once per tile before main K.
                # Uses pipeline_aw's A/B mcast masks (LA is A-like, LU is
                # B-like) so under 2-CTA both cluster CTAs see the data.
                if cutlass.const_expr(self.enable_lora):
                    pipeline_lora.producer_acquire(
                        make_pipeline_state_from_index_phase(
                            self.num_lora_stage, Int32(0), lora_producer_phase,
                        )
                    )
                    lora_bar = pipeline_lora.producer_get_barrier(
                        make_pipeline_state_from_index_phase(
                            self.num_lora_stage, Int32(0), lora_producer_phase,
                        )
                    )
                    tALgLA_slice = tALgLA[(None, m_tile, 0, l_tile)]
                    tBLgLU_slice = tBLgLU[(None, n_tile, 0, l_tile)]
                    cute.copy(
                        tma_atom_la, tALgLA_slice,
                        tALsLA[(None, 0)],
                        tma_bar_ptr=lora_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_lu, tBLgLU_slice,
                        tBLsLU[(None, 0)],
                        tma_bar_ptr=lora_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                tAgA_slice = tAgA[(None, m_tile, None, l_tile)]
                tBgB_slice = tBgB[(None, n_tile, None, l_tile)]
                tAgSFA_slice = tAgSFA[(None, m_tile, None, l_tile)]
                tBgSFB_slice = tBgSFB[(None, n_tile, None, l_tile)]

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    pipeline_aw.producer_acquire(aw_producer_state)
                    tma_bar = pipeline_aw.producer_get_barrier(aw_producer_state)
                    cute.copy(
                        tma_atom_a, tAgA_slice[(None, k_tile)],
                        tAsA[(None, aw_producer_state.index)],
                        tma_bar_ptr=tma_bar, mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b, tBgB_slice[(None, k_tile)],
                        tBsB[(None, aw_producer_state.index)],
                        tma_bar_ptr=tma_bar, mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa, tAgSFA_slice[(None, k_tile)],
                        tAsSFA[(None, aw_producer_state.index)],
                        tma_bar_ptr=tma_bar, mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb, tBgSFB_slice[(None, k_tile)],
                        tBsSFB[(None, aw_producer_state.index)],
                        tma_bar_ptr=tma_bar, mcast_mask=sfb_full_mcast_mask,
                    )
                    aw_producer_state.advance()

                if cutlass.const_expr(self.enable_lora):
                    lora_producer_phase ^= 1

                tile_idx += grid_stride

            # Multi-stage producer_tail: drain last `num_ab_stage - 1`
            # buffers. Safe because pipeline_aw has >= 2 stages (asserted
            # in _setup_attributes); under 2-CTA the drain is leader-CTA
            # gated by PipelineTmaUmma's internal is_leader_cta check.
            _producer_tail_simple(pipeline_aw, aw_producer_state)
            # Single-stage LoRA tail: bare acquire on the next phase per
            # README §2CTA conventions and FA4 load():1505-1506 pattern.
            # producer_tail would try to drain a non-existent slot and
            # deadlock under 2-CTA (DEBUG_2CTA 2CTA-2).
            if cutlass.const_expr(self.enable_lora):
                pipeline_lora.producer_acquire(
                    make_pipeline_state_from_index_phase(
                        self.num_lora_stage, Int32(0), lora_producer_phase,
                    )
                )

        # =========================================================
        # MMA warp — issues one NVFP4 scaled-MMA per K-block.
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

            aw_consumer_state = make_pipeline_state_simple(
                PipelineUserType.Consumer, self.num_ab_stage)
            # Single-stage pipeline_acc — phase bit only (XOR toggle).
            # Producer starts at phase=1: `pipeline_init_arrive` pre-arms
            # the empty barrier to parity=1, so the first `producer_acquire`
            # with phase=1 returns immediately. Starting at 0 blocks forever
            # (consumer never flips full, kernel hangs — was the 9-min hang
            # on first smoke). Mirrors stock `make_pipeline_state(Producer)`.
            acc_producer_phase = Int32(1)
            # LoRA consumer: single-stage, mma warp is the consumer. Phase
            # starts at 0 (producer starts at 1; see README §Pipeline state).
            lora_consumer_phase = Int32(0)

            tCtAcc = tCtAcc_base[(None, None, None, 0)]
            # Shared-tmem LoRA acc: same ptr as main, TV-layout match
            # verified at trace time (tmp/verify_tmem_layout.py). Both
            # MMAs' partition_C writes to identical tmem cells, so the β
            # interleave += into the same accumulator.
            tCtAcc_lora = None
            if cutlass.const_expr(self.enable_lora):
                tCtAcc_lora_base = cute.make_tensor(
                    acc_tmem_ptr, tCtAcc_lora_fake.layout)
                tCtAcc_lora = tCtAcc_lora_base[(None, None, None, 0)]

            num_kblocks = cute.size(tCrA, mode=[2])

            tile_idx = initial_tile_idx
            grid_stride = self._grid_stride()
            while tile_idx < total_tiles_cluster:
                # Acquire acc buffer (single-stage, phase-only).
                if is_leader_cta:
                    pipeline_acc.producer_acquire(
                        make_pipeline_state_from_index_phase(
                            self.num_acc_stage, Int32(0), acc_producer_phase,
                        )
                    )

                # LoRA prolog wait: block until load warp's LA/LU TMA
                # commits. Only leader CTA issues MMAs, so only leader
                # waits on the consumer side.
                if cutlass.const_expr(self.enable_lora):
                    if is_leader_cta:
                        pipeline_lora.consumer_wait(
                            make_pipeline_state_from_index_phase(
                                self.num_lora_stage, Int32(0),
                                lora_consumer_phase,
                            )
                        )
                    # LoRA MMA += into shared acc from the first atom.
                    tiled_mma_lora.set(tcgen05.Field.ACCUMULATE, True)

                # `tiled_mma.set(ACCUMULATE, …)` is a trace-time Python
                # mutation — state is captured at each `cute.gemm` site
                # when the MLIR is built, not when the device executes.
                # So we need `range()` (fully unrolled at trace) rather
                # than `cutlass.range(..., unroll=1)` (body traced once,
                # reused at runtime): the latter would capture ACC=False
                # at the single kblock=0 site and re-run it every k_tile,
                # wiping the acc between K-tiles. With `range()`, MLIR
                # holds exactly one gemm(ACC=False) at the very first
                # unrolled position — every other site is gemm(ACC=True).
                # The outer persistent `while tile_idx` re-runs the whole
                # unrolled block once per tile, which is the per-tile
                # "zero then accumulate" shape we want. Mirrors kernel.py.
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                # β-interleave state (design §2). `stride` = # main atoms
                # between LoRA atoms; `R_atoms` total LoRA atoms scheduled
                # evenly across all main K atoms. tcgen05 issue queue is
                # per-CTA in-order, so the LoRA atom sees the preceding
                # main atom's tmem write. Leader CTA only (2-CTA: leader
                # issues both main and LoRA, cluster MMAs propagate).
                r_next = cutlass.Int32(0)
                next_lora_at = cutlass.Int32(0)
                k_atom_flat = cutlass.Int32(0)
                if cutlass.const_expr(self.enable_lora):
                    k_atoms_total = k_tile_cnt * num_kblocks
                    stride = k_atoms_total // self.R_atoms

                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        pipeline_aw.consumer_wait(aw_consumer_state)
                        s2t_stage_coord = (None, None, None, None, aw_consumer_state.index)
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_s2t[s2t_stage_coord], tCtSFA_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_s2t[s2t_stage_coord], tCtSFB_s2t,
                        )
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None, None, kblock_idx, aw_consumer_state.index,
                            )
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

                            # β-interleave: inject LoRA atom every `stride`
                            # main atoms. Shared-tmem acc means the LoRA
                            # gemm += into the same tCtAcc cells (verified).
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
                        pipeline_aw.consumer_release(aw_consumer_state)
                    aw_consumer_state.advance()

                # LoRA consumer release (per tile; LA/LU smem freed).
                if cutlass.const_expr(self.enable_lora):
                    if is_leader_cta:
                        pipeline_lora.consumer_release(
                            make_pipeline_state_from_index_phase(
                                self.num_lora_stage, Int32(0),
                                lora_consumer_phase,
                            )
                        )
                    lora_consumer_phase ^= 1

                # Commit acc and hand off to epilogue. UmmaAsync commit
                # uses state.index only — single-stage, index is always 0.
                if is_leader_cta:
                    pipeline_acc.producer_commit(
                        make_pipeline_state_from_index_phase(
                            self.num_acc_stage, Int32(0), acc_producer_phase,
                        )
                    )
                acc_producer_phase ^= 1

                tile_idx += grid_stride

            # 2-CTA-safe tail: single-stage pipeline_acc. `producer_tail`
            # would try to advance and re-acquire an already-drained slot
            # (deadlock per DEBUG_2CTA 2CTA-2). Bare acquire on the NEXT
            # phase visits the same index with the flipped phase and
            # blocks only if epilogue hasn't drained. For the kernel
            # last iteration this is a no-op.
            if is_leader_cta:
                pipeline_acc.producer_acquire(
                    make_pipeline_state_from_index_phase(
                        self.num_acc_stage, Int32(0), acc_producer_phase,
                    )
                )

        # =========================================================
        # EPILOGUE warps — tmem → smem → gmem (TMA store), per tile.
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

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, 0)]
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

            acc_consumer_phase = Int32(0)

            tile_idx = initial_tile_idx
            grid_stride = self._grid_stride()
            while tile_idx < total_tiles_cluster:
                l_tile = tile_idx // tiles_per_l
                mn_rem = tile_idx % tiles_per_l
                m_tile = mn_rem // num_n
                n_tile = mn_rem % num_n

                bSG_gC = bSG_gC_partitioned[(None, None, None, m_tile, n_tile, l_tile)]
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                # Wait for MMA to commit acc for this tile.
                pipeline_acc.consumer_wait(
                    make_pipeline_state_from_index_phase(
                        self.num_acc_stage, Int32(0), acc_consumer_phase,
                    )
                )

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

                # Release acc — MMA can overwrite for next tile.
                pipeline_acc.consumer_release(
                    make_pipeline_state_from_index_phase(
                        self.num_acc_stage, Int32(0), acc_consumer_phase,
                    )
                )
                acc_consumer_phase ^= 1

                tile_idx += grid_stride

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            c_pipeline.producer_tail()

    # -------- helpers (cta_group-aware grid stride; cps ported from kernel.py) --

    def _grid_stride(self) -> Int32:
        """Persistent advance stride. Cluster-aware: under cluster_m>1,
        advance by `cluster_dim`, else by `grid_dim`."""
        if cutlass.const_expr(self.cluster_shape_mn[0] == 1):
            return Int32(cute.arch.grid_dim()[0])
        return Int32(cute.arch.cluster_dim()[0])

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
        self, tidx: Int32, tAcc: cute.Tensor, gC_mnl: cute.Tensor,
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
        tidx: Int32, sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def _epilog_gmem_copy(
        self, tidx: Int32,
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


def _producer_tail_simple(pipe, state: PipelineStateSimple) -> None:
    """Stock `producer_tail` calls `state.advance()` num_stages-1 times
    then `producer_acquire(state)`. Matches the FA4-simple state under
    the monotonic `_phase_index += 1` rule. Kept as a free function so
    the simple state doesn't need to pretend to be a stock PipelineState
    for MLIR marshalling."""
    for _ in range(pipe.num_stages - 1):
        state.advance()
    pipe.producer_acquire(state)
