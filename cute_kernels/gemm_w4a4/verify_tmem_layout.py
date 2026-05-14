"""Verify that NVFP4 block-scaled MMA and fp16/bf16 dense MMA produce
the same accumulator fragment layout, which is the precondition for
sharing one tmem region across the two MMA segments of gemm_w4a4.

If acc TV layouts match → (α) shared-tmem design is viable.
If not → fall back to separate tmem regions + epilogue sum.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100

OperandMajorMode = tcgen05.OperandMajorMode
CtaGroup = tcgen05.CtaGroup

# Our fused op uses K-major for both A and B on both inner MMAs:
#   main : act[M, K/2]   · wgt[N, K/2].T   — K is the reduction, contig in K
#   lora : lora_act_in[M, R] · lora_up[N, R].T   — R is the reduction, contig in R
A_MODE = OperandMajorMode.K
B_MODE = OperandMajorMode.K


class LayoutProbe:
    """Mirror the @cute.jit entry pattern used by the CUTLASS examples."""

    def __init__(self, mma_tiler_mn, cta_group):
        self.mma_tiler_mn = mma_tiler_mn
        self.cta_group = cta_group

    @cute.jit
    def __call__(self):
        main_mma = sm100.make_blockscaled_trivial_tiled_mma(
            ab_dtype=cutlass.Float4E2M1FN,
            a_leading_mode=A_MODE,
            b_leading_mode=B_MODE,
            sf_dtype=cutlass.Float8E4M3FN,
            sf_vec_size=16,
            cta_group=self.cta_group,
            mma_tiler_mn=self.mma_tiler_mn,
        )
        lora_f16 = sm100.make_trivial_tiled_mma(
            ab_dtype=cutlass.Float16,
            a_leading_mode=A_MODE,
            b_leading_mode=B_MODE,
            acc_dtype=cutlass.Float32,
            cta_group=self.cta_group,
            mma_tiler_mn=self.mma_tiler_mn,
        )
        lora_bf16 = sm100.make_trivial_tiled_mma(
            ab_dtype=cutlass.BFloat16,
            a_leading_mode=A_MODE,
            b_leading_mode=B_MODE,
            acc_dtype=cutlass.Float32,
            cta_group=self.cta_group,
            mma_tiler_mn=self.mma_tiler_mn,
        )
        for name, mma in [("main NVFP4", main_mma),
                          ("lora fp16", lora_f16),
                          ("lora bf16", lora_bf16)]:
            print(f"[{name}] shape_mnk          = {mma.shape_mnk}")
            print(f"[{name}] thr_id.shape       = {mma.thr_id.shape}")
            print(f"[{name}] thr_layout_vmnk    = {mma.thr_layout_vmnk}")
            print(f"[{name}] tv_layout_C        = {mma.tv_layout_C}")
            print(f"[{name}] tv_layout_C_tiled  = {mma.tv_layout_C_tiled}")
            print(f"[{name}] partition_shape_C  = "
                  f"{mma.partition_shape_C((self.mma_tiler_mn[0], self.mma_tiler_mn[1]))}")
            print("---")

        def same(a, b):
            return str(a) == str(b)

        for probe, other in [("lora fp16", lora_f16), ("lora bf16", lora_bf16)]:
            tv_same    = same(main_mma.tv_layout_C_tiled, other.tv_layout_C_tiled)
            thr_same   = same(main_mma.thr_layout_vmnk, other.thr_layout_vmnk)
            shape_same = same(main_mma.shape_mnk, other.shape_mnk)
            print(f">>> {probe}: shape_mnk match={shape_same}  "
                  f"thr_layout_vmnk match={thr_same}  "
                  f"tv_layout_C_tiled match={tv_same}")


def main():
    for tiler_mn, cta_group, label in [
        ((128, 128), CtaGroup.ONE, "1SM 128x128"),
        ((128, 256), CtaGroup.ONE, "1SM 128x256"),
        ((256, 128), CtaGroup.TWO, "2SM 256x128 (v0_fa4 default)"),
        ((256, 256), CtaGroup.TWO, "2SM 256x256"),
    ]:
        print("=" * 72)
        print(f"config: {label}")
        print("=" * 72)
        LayoutProbe(tiler_mn, cta_group)()
        print()


if __name__ == "__main__":
    main()
