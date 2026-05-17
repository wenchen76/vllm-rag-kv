# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scatter a ``LoadedPlan`` into the paged KV cache.

``load_plan`` (Phase 5) produces per-chunk, per-block K/V tensors with
delta-RoPE applied at the runtime positions. Those tensors live as
standalone allocations and must be copied into the paged KV cache at
scheduler-assigned physical block ids before any attention layer can
consume them.

This module exposes two primitives:

    - ``scatter_loaded_block(loaded_block, kv_caches, block_id)`` —
      the per-block primitive; writes one ``LoadedBlock`` across all
      layers' caches at one physical block id.
    - ``scatter_loaded_plan(loaded_plan, kv_caches, block_assignments)``
      — the plan-wide convenience wrapper; walks chunks × blocks and
      delegates to the per-block primitive.

Layout: NHD only. Each per-layer cache must be a 5-D tensor of shape
``[num_blocks, 2, page_size, num_kv_heads, head_dim]`` — index 0 along
dim 1 is K, index 1 is V. This matches the layout assumption used by
``PrefillSetupSpec`` (Phase 6) and by FlashInfer's NHD path. HND can be
added later if a backend requires it.

The scatter is pure in-place ``copy_``: no FlashInfer / CUDA kernels.
It runs unchanged on CPU tensors, which keeps the primitive
unit-testable without a GPU. ``reshape_and_cache_flash`` (vLLM's
per-token slot-mapping path) is *not* used because chunk-loaded data is
already block-aligned by construction (see
``validate_chunk_alignment``) and a direct slice copy is both simpler
and exposes shape mismatches as clear ``ValueError``s instead of opaque
kernel failures.

Validation is two-pass: every block's shape, dtype, and assigned block
id is checked before any copy happens, so a downstream failure cannot
leave the paged cache in a half-written state.
"""

from __future__ import annotations

from typing import Sequence

import torch

from vllm.v1.personal_context.load import LoadedBlock, LoadedPlan


def scatter_loaded_block(
    loaded_block: LoadedBlock,
    kv_caches: Sequence[torch.Tensor],
    physical_block_id: int,
) -> None:
    """Write one ``LoadedBlock`` into every layer's paged cache.

    For every layer ``L``, copies
        ``loaded_block.keys[L]``   → ``kv_caches[L][block, 0]``
        ``loaded_block.values[L]`` → ``kv_caches[L][block, 1]``

    where ``block == physical_block_id``.

    Args:
        loaded_block: Per-layer K (delta-rotated) and V (cloned) from
            ``load_plan``; each tensor has shape
            ``[block_size, num_kv_heads, head_dim]``.
        kv_caches: Per-layer paged KV cache in NHD layout, shape
            ``[num_blocks, 2, page_size, num_kv_heads, head_dim]``.
            Length must equal ``len(loaded_block.keys)``.
        physical_block_id: Destination block id, used for every layer.
            Must satisfy ``0 <= id < num_blocks``.

    Raises:
        ValueError: layer count mismatch, wrong cache rank/shape,
            block id out of range, per-layer K/V shape or dtype
            mismatch.
    """
    _validate_block(loaded_block, kv_caches, physical_block_id)
    for k, v, cache in zip(
        loaded_block.keys, loaded_block.values, kv_caches
    ):
        cache[physical_block_id, 0].copy_(k)
        cache[physical_block_id, 1].copy_(v)


def scatter_loaded_plan(
    loaded_plan: LoadedPlan,
    kv_caches: Sequence[torch.Tensor],
    block_assignments: Sequence[Sequence[int | None]],
) -> None:
    """Write every hit block in ``loaded_plan`` into ``kv_caches``.

    For each chunk ``i`` and block ``j`` in ``loaded_plan``:

        - If ``loaded_plan.chunks[i].blocks[j] is None`` (miss): skip.
        - Else if ``block_assignments[i][j] is None``: skip (caller
          chose not to scatter — e.g. selective recompute opted out
          of this block).
        - Else: scatter that block at ``block_assignments[i][j]``.

    All shape / dtype / range validation runs upfront across every
    (chunk, block) pair that *would* be written. If any check fails,
    a ``ValueError`` is raised and no copy is performed.

    Args:
        loaded_plan: Plan-wide result from ``load_plan``.
        kv_caches: Per-layer paged KV cache (see
            ``scatter_loaded_block`` for layout).
        block_assignments: ``block_assignments[i][j]`` is the physical
            block id for the ``j``-th block of the ``i``-th chunk in
            the plan, or ``None`` to skip. The outer length must equal
            ``len(loaded_plan.chunks)``; each inner length must equal
            ``len(loaded_plan.chunks[i].blocks)``.

    Raises:
        ValueError: structural mismatch between ``loaded_plan`` and
            ``block_assignments``, or any per-block validation failure.
    """
    if len(block_assignments) != len(loaded_plan.chunks):
        raise ValueError(
            f"block_assignments has {len(block_assignments)} entries, "
            f"loaded_plan has {len(loaded_plan.chunks)} chunks"
        )
    for chunk_idx, (loaded_chunk, chunk_assignments) in enumerate(
        zip(loaded_plan.chunks, block_assignments)
    ):
        if len(chunk_assignments) != len(loaded_chunk.blocks):
            raise ValueError(
                f"chunk {chunk_idx}: block_assignments has "
                f"{len(chunk_assignments)} entries but the loaded chunk "
                f"has {len(loaded_chunk.blocks)} blocks"
            )
        for loaded_block, assignment in zip(
            loaded_chunk.blocks, chunk_assignments
        ):
            if loaded_block is None or assignment is None:
                continue
            _validate_block(loaded_block, kv_caches, assignment)

    for loaded_chunk, chunk_assignments in zip(
        loaded_plan.chunks, block_assignments
    ):
        for loaded_block, assignment in zip(
            loaded_chunk.blocks, chunk_assignments
        ):
            if loaded_block is None or assignment is None:
                continue
            for k, v, cache in zip(
                loaded_block.keys, loaded_block.values, kv_caches
            ):
                cache[assignment, 0].copy_(k)
                cache[assignment, 1].copy_(v)


def _validate_block(
    loaded_block: LoadedBlock,
    kv_caches: Sequence[torch.Tensor],
    physical_block_id: int,
) -> None:
    num_layers = len(kv_caches)
    if len(loaded_block.keys) != num_layers:
        raise ValueError(
            f"loaded_block has {len(loaded_block.keys)} K layers, "
            f"kv_caches has {num_layers}"
        )
    if len(loaded_block.values) != num_layers:
        raise ValueError(
            f"loaded_block has {len(loaded_block.values)} V layers, "
            f"kv_caches has {num_layers}"
        )
    for layer_idx, (k, v, cache) in enumerate(
        zip(loaded_block.keys, loaded_block.values, kv_caches)
    ):
        if cache.dim() != 5:
            raise ValueError(
                f"kv_caches[{layer_idx}] must be 5-D NHD "
                f"[num_blocks, 2, page_size, num_kv_heads, head_dim]; "
                f"got shape {tuple(cache.shape)}"
            )
        if cache.shape[1] != 2:
            raise ValueError(
                f"kv_caches[{layer_idx}] dim 1 must be 2 (K, V); got "
                f"{cache.shape[1]}"
            )
        if not 0 <= physical_block_id < cache.shape[0]:
            raise ValueError(
                f"physical_block_id {physical_block_id} out of range "
                f"for kv_caches[{layer_idx}] with {cache.shape[0]} blocks"
            )
        expected = tuple(cache.shape[2:])
        if tuple(k.shape) != expected:
            raise ValueError(
                f"loaded_block.keys[{layer_idx}] has shape "
                f"{tuple(k.shape)}, expected {expected}"
            )
        if tuple(v.shape) != expected:
            raise ValueError(
                f"loaded_block.values[{layer_idx}] has shape "
                f"{tuple(v.shape)}, expected {expected}"
            )
        if k.dtype != cache.dtype:
            raise ValueError(
                f"loaded_block.keys[{layer_idx}] dtype {k.dtype} != "
                f"kv_caches[{layer_idx}] dtype {cache.dtype}"
            )
        if v.dtype != cache.dtype:
            raise ValueError(
                f"loaded_block.values[{layer_idx}] dtype {v.dtype} != "
                f"kv_caches[{layer_idx}] dtype {cache.dtype}"
            )
