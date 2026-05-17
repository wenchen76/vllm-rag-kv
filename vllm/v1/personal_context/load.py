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
from vllm.v1.personal_context.rope import apply_delta_rope


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

        loaded_blocks: list[LoadedBlock | None] = []
        for i, bl in enumerate(chunk_lookup.blocks):
            if bl.block is None:
                loaded_blocks.append(None)
                continue
            block_new_pos = new_pos + i * block_size
            rotated_keys = [
                apply_delta_rope(k, delta, rope_theta=rope_theta)
                for k in bl.block.keys
            ]
            cloned_values = [v.clone() for v in bl.block.values]
            loaded_blocks.append(
                LoadedBlock(
                    keys=rotated_keys,
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
