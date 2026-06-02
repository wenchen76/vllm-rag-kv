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
        if hit_indices:
            num_layers = len(chunk_lookup.blocks[hit_indices[0]].block.keys)
            # ---- TEMP instrumentation: VLLM_PC_TIME_ROPE=1 splits the
            # batched path into stack / move-to-GPU / rotate to find where
            # load_plan's ~470ms actually goes (CPU rotate vs stack vs
            # transfer) and whether GPU rotate is the fix. Remove after.
            import os as _os  # noqa: PLC0415
            import time as _time  # noqa: PLC0415
            _prof = bool(_os.environ.get("VLLM_PC_TIME_ROPE"))

            if _prof:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                _ts = _time.perf_counter()
            # Stack: [num_hit_blocks * num_layers, block_size, nh, hd],
            # block-major then layer (so we can slice it back per block).
            stacked = torch.stack(
                [
                    chunk_lookup.blocks[i].block.keys[layer]
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
                    stacked, delta, rope_theta=rope_theta
                )
                _t_cpu = _time.perf_counter() - _ts
                # (b) move to GPU then rotate on GPU.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    _ts = _time.perf_counter()
                    _g = stacked.to("cuda")
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
                    stacked.shape[0],
                    _t_stack * 1000, _t_cpu * 1000,
                    _t_move * 1000, _t_gpu * 1000,
                )
            rotated = apply_delta_rope_batched(
                stacked, delta, rope_theta=rope_theta
            )
            for slot, i in enumerate(hit_indices):
                base = slot * num_layers
                rotated_per_block[i] = [
                    rotated[base + layer] for layer in range(num_layers)
                ]

        loaded_blocks: list[LoadedBlock | None] = []
        for i, bl in enumerate(chunk_lookup.blocks):
            if bl.block is None:
                loaded_blocks.append(None)
                continue
            block_new_pos = new_pos + i * block_size
            cloned_values = [v.clone() for v in bl.block.values]
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

    return LoadedPlan(chunks=tuple(loaded_chunks))
