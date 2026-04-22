"""Minimal FA4-style pipeline state for v0 FA4 skeleton.

Vendored from `tmp/flash-attention/flash_attn/cute/pipeline.py` (Tri Dao,
BSD). Only `PipelineStateSimple` + `make_pipeline_state_simple`. The
monotonic `_phase_index` counter is a drop-in for the stock
`cutlass.pipeline.PipelineState` — `producer_acquire`/`consumer_wait`
only read `.index` / `.phase`, so stock pipeline classes accept this.

Why: stock `PipelineState.advance()` branches (`if index == stages:
index=0; phase^=1`). In a persistent kernel that branch drifts state
across tile boundaries; a prior 2-CTA persistent port hung 500× once
per-CTA tile count rose (see kernel.py README, `61905df`). FA4's
monotonic counter removes the branch — `index`/`phase` are pure divmod.
"""
from __future__ import annotations

from cutlass import Int32, const_expr
from cutlass.pipeline import PipelineState, PipelineUserType


class PipelineStateSimple:
    """Single-Int32 state, `index = _phase_index % stages`,
    `phase = _phase_index // stages`. Stages==1 shortcut: phase is
    stored directly (XOR toggle), index is always 0."""

    def __init__(self, stages: int, phase_index: Int32):
        self._stages = stages
        self._phase_index = phase_index

    def clone(self) -> "PipelineStateSimple":
        return PipelineStateSimple(self._stages, self._phase_index)

    @property
    def stages(self) -> int:
        return self._stages

    @property
    def index(self) -> Int32:
        if const_expr(self._stages == 1):
            return Int32(0)
        return self._phase_index % self._stages

    @property
    def phase(self) -> Int32:
        if const_expr(self._stages == 1):
            return self._phase_index
        return self._phase_index // self._stages

    def advance(self) -> None:
        if const_expr(self._stages == 1):
            self._phase_index ^= 1
        else:
            self._phase_index += 1

    def __extract_mlir_values__(self):
        return [self._phase_index.ir_value()]

    def __new_from_mlir_values__(self, values):
        return PipelineStateSimple(self._stages, Int32(values[0]))


def make_pipeline_state_simple(kind: PipelineUserType, stages: int) -> PipelineStateSimple:
    """Producer starts with phase_index = stages (empty buffer, phase=1).
    Consumer starts at 0."""
    if kind is PipelineUserType.Producer:
        return PipelineStateSimple(stages, Int32(stages))
    if kind is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
    raise AssertionError(f"invalid PipelineUserType: {kind}")


def make_pipeline_state_from_index_phase(
    stages: int, index: Int32, phase: Int32,
) -> PipelineState:
    """Build a stock `PipelineState` from raw (index, phase). Count is
    unused by `producer_acquire` / `consumer_wait` / `producer_commit` /
    `consumer_release`, set to 0. Used for the FA4
    `producer_acquire_w_index_phase(0, phase)` pattern applied to
    single-stage pipelines — bypasses `producer_tail`'s state.advance
    loop (which deadlocks under 2CTA when the next state is already
    drained)."""
    return PipelineState(stages, Int32(0), index, phase)
