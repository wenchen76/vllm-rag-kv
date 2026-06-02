# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load + delta-RoPE pipeline for personal-context KV reuse.

Given the ``PlanLookup`` from the connector and the runtime positions
the scheduler has assigned to each chunk, this module produces
``LoadedBlock`` tensors (K rotated by ``delta = new_pos - old_pos``,
V cloned) ready to be scattered into the paged KV cache.

Miss blocks pass through as ``None`` so callers can selectively fall
back to oracle-path prefill at block granularity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.personal_context.chunk import Chunk, validate_chunk_alignment
from vllm.v1.personal_context.connector import PlanLookup
from vllm.v1.personal_context.rope import apply_delta_rope_batched

# TEMP profiling accumulator for load_plan phase timing (VLLM_PC_TIME_ROPE).
# Reset per load_plan call. Remove with the rest of the instrumentation.
_PROF_ACC: dict = {"stack": 0.0, "k_move": 0.0, "rotate": 0.0, "v_move": 0.0}


@dataclass(frozen=True)
class LoadedBlock:
    """One block of K (delta-rotated) and V (cloned), placed at ``new_pos_start``."""

    # length == num_layers; each tensor shape [block_size, num_kv_heads, head_dim]
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    new_pos_start: int


@dataclass(frozen=True)
class LoadedChunk:
    """Per-chunk load result. ``blocks`` is in chunk order; ``None`` == miss."""

    chunk: Chunk
    new_pos_start: int
    blocks: tuple[LoadedBlock | None, ...]


@dataclass(frozen=True)
class LoadedPlan:
    """Plan-wide load result; one ``LoadedChunk`` per chunk in the plan."""

    chunks: tuple[LoadedChunk, ...]


def load_plan(
    plan_lookup: PlanLookup,
    new_pos_starts: tuple[int, ...] | list[int],
    rope_theta: float = 10000.0,
    device: torch.device | str | None = None,
) -> LoadedPlan:
    """Materialise loaded blocks for every hit in ``plan_lookup``.

    Args:
        plan_lookup: Output of ``PersonalContextConnector.lookup``.
        new_pos_starts: Runtime start position assigned to each chunk
            by the scheduler. Must have the same length as
            ``plan_lookup.chunks`` and each entry must be a non-negative
            multiple of ``plan_lookup.block_size``.
        rope_theta: RoPE base. Must match the model the K was encoded
            under.
        device: If given, K (and V) are moved to this device before the
            delta-RoPE rotation. Stores hand back CPU tensors (the encoder
            ``.cpu()``s blocks before storing, regardless of backend), and
            rotating them on CPU dominated load latency — profiled at
            ~15-65 ms per chunk (~470 ms total for a 14-chunk 2k context)
            vs ~0.7 ms per chunk on GPU, a ~25x gap, with the host→device
            copy itself only ~1 ms. Passing the paged cache's device
            (cuda) runs the rotation on GPU and makes the downstream
            scatter a GPU→GPU copy. ``None`` keeps tensors on whatever
            device the store returned (CPU), preserving prior behaviour.

    Returns:
        ``LoadedPlan`` with one ``LoadedChunk`` per chunk and one entry
        per block (``None`` for misses).

    Raises:
        ValueError: ``new_pos_starts`` length mismatch.
        AlignmentError: a ``new_pos_start`` is not block-aligned.
    """
    if len(new_pos_starts) != len(plan_lookup.chunks):
        raise ValueError(
            f"new_pos_starts has {len(new_pos_starts)} entries but plan "
            f"has {len(plan_lookup.chunks)} chunks"
        )

    # TEMP profiling accumulator (VLLM_PC_TIME_ROPE=1): per-phase totals
    # across all chunks of this plan, to attribute load_plan's residual
    # latency. Reset each call. Remove with the rest of the instrumentation.
    global _PROF_ACC
    _PROF_ACC = {"stack": 0.0, "k_move": 0.0, "rotate": 0.0, "v_move": 0.0}

    block_size = plan_lookup.block_size
    loaded_chunks: list[LoadedChunk] = []
    for chunk_lookup, new_pos in zip(plan_lookup.chunks, new_pos_starts):
        chunk = chunk_lookup.chunk
        # Re-uses the same alignment contract as the store side: this
        # also checks that the chunk's length is still block-aligned.
        validate_chunk_alignment(new_pos, len(chunk.token_ids), block_size)
        delta = new_pos - chunk.old_pos_start

        # All (block, layer) K tensors of this chunk shift by the same
        # ``delta``, so rotate them in ONE batched call instead of one
        # apply_delta_rope per (block, layer) — the latter recomputes the
        # identical cos/sin and launches a kernel ~num_blocks*num_layers
        # times, which dominated load latency for long contexts. Gather
        # every hit block's K stack, rotate once, then scatter back.
        hit_indices = [
            i for i, bl in enumerate(chunk_lookup.blocks)
            if bl.block is not None
        ]
        rotated_per_block: dict[int, list] = {}
        values_per_block: dict[int, list] = {}
        if hit_indices:
            num_layers = len(chunk_lookup.blocks[hit_indices[0]].block.keys)
            # ---- TEMP instrumentation: VLLM_PC_TIME_ROPE=1 splits the
            # batched path into stack / move-to-GPU / rotate to find where
            # load_plan's ~470ms actually goes (CPU rotate vs stack vs
            # transfer) and whether GPU rotate is the fix. Remove after.
            import os as _os  # noqa: PLC0415
            import time as _time  # noqa: PLC0415
            _prof = bool(_os.environ.get("VLLM_PC_TIME_ROPE"))

            # K and V together: ONE stack of [2*M, block, nh, hd] (the M
            # K tensors then the M V tensors) and ONE host→device copy,
            # instead of stacking + moving K and V separately. This halves
            # both the CPU stack-copies and the H2D transfers, and the
            # single larger contiguous transfer uses PCIe bandwidth better
            # than two smaller ones. After the move, the K half is sliced
            # off and delta-RoPE'd; the V half (no rotation) is used as-is.
            m = len(hit_indices) * num_layers
            if _prof:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                _ts = _time.perf_counter()
            kv_stacked = torch.stack(
                [
                    chunk_lookup.blocks[i].block.keys[layer]
                    for i in hit_indices
                    for layer in range(num_layers)
                ]
                + [
                    chunk_lookup.blocks[i].block.values[layer]
                    for i in hit_indices
                    for layer in range(num_layers)
                ],
                dim=0,
            )
            if _prof:
                _t_stack = _time.perf_counter() - _ts
                # (a) rotate on the tensor's native device (CPU for mmap).
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                _ts = _time.perf_counter()
                _ = apply_delta_rope_batched(
                    kv_stacked[:m], delta, rope_theta=rope_theta
                )
                _t_cpu = _time.perf_counter() - _ts
                # (b) move to GPU then rotate on GPU.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    _ts = _time.perf_counter()
                    _g = kv_stacked[:m].to("cuda")
                    torch.cuda.synchronize()
                    _t_move = _time.perf_counter() - _ts
                    _ts = _time.perf_counter()
                    _ = apply_delta_rope_batched(
                        _g, delta, rope_theta=rope_theta
                    )
                    torch.cuda.synchronize()
                    _t_gpu = _time.perf_counter() - _ts
                else:
                    _t_move = _t_gpu = float("nan")
                import logging as _lg  # noqa: PLC0415
                _lg.getLogger(__name__).info(
                    "PC rope-prof (M=%d): stack=%.1fms cpu_rotate=%.1fms "
                    "move_gpu=%.1fms gpu_rotate=%.1fms",
                    m,
                    _t_stack * 1000, _t_cpu * 1000,
                    _t_move * 1000, _t_gpu * 1000,
                )
            # Single combined H2D copy for K+V. Stores hand back CPU
            # tensors; rotating K on CPU was profiled ~25x slower than GPU
            # and dominated load latency, so when a ``device`` is given we
            # move the whole K+V stack there once (the H2D is ~1ms) and the
            # downstream scatter becomes a GPU→GPU copy.
            # ---- PROD-PATH profiling (VLLM_PC_TIME_ROPE=1): k_move now
            # times the combined K+V H2D; v_move is folded in (stays ~0).
            if _prof and device is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _ts2 = _time.perf_counter()
            if device is not None:
                kv_stacked = kv_stacked.to(device)
            else:
                kv_stacked = kv_stacked.clone()
            if _prof and device is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _PROF_ACC["k_move"] += _time.perf_counter() - _ts2
                _PROF_ACC["stack"] += _t_stack
                _ts2 = _time.perf_counter()
            # K half: rotate; V half: leave as-is.
            rotated = apply_delta_rope_batched(
                kv_stacked[:m], delta, rope_theta=rope_theta
            )
            v_moved = kv_stacked[m:]
            if _prof and device is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _PROF_ACC["rotate"] += _time.perf_counter() - _ts2
            for slot, i in enumerate(hit_indices):
                base = slot * num_layers
                rotated_per_block[i] = [
                    rotated[base + layer] for layer in range(num_layers)
                ]
                values_per_block[i] = [
                    v_moved[base + layer] for layer in range(num_layers)
                ]

        loaded_blocks: list[LoadedBlock | None] = []
        for i, bl in enumerate(chunk_lookup.blocks):
            if bl.block is None:
                loaded_blocks.append(None)
                continue
            block_new_pos = new_pos + i * block_size
            cloned_values = values_per_block[i]
            loaded_blocks.append(
                LoadedBlock(
                    keys=rotated_per_block[i],
                    values=cloned_values,
                    new_pos_start=block_new_pos,
                )
            )
        loaded_chunks.append(
            LoadedChunk(
                chunk=chunk,
                new_pos_start=new_pos,
                blocks=tuple(loaded_blocks),
            )
        )

    import os as _os3  # noqa: PLC0415
    if _os3.environ.get("VLLM_PC_TIME_ROPE") and device is not None:
        import logging as _lg3  # noqa: PLC0415
        _lg3.getLogger(__name__).info(
            "PC load_plan prod-path (%d chunks): stack=%.1fms "
            "k_move_h2d=%.1fms rotate=%.1fms v_move_h2d=%.1fms",
            len(plan_lookup.chunks),
            _PROF_ACC["stack"] * 1000, _PROF_ACC["k_move"] * 1000,
            _PROF_ACC["rotate"] * 1000, _PROF_ACC["v_move"] * 1000,
        )

    return LoadedPlan(chunks=tuple(loaded_chunks))
