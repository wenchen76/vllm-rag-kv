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

Layout: two paged-KV layouts are auto-detected by shape:

    - ``block_first`` (FlashInfer NHD convention, also used by Phase 6
      ``PrefillSetupSpec``): ``[num_blocks, 2, page_size, num_kv_heads,
      head_dim]`` — dim 1 is the K/V split.
    - ``kv_first`` (vLLM FlashAttention GPU backend, see
      ``example_connector.py``'s default branch): ``[2, num_blocks,
      page_size, num_kv_heads, head_dim]`` — dim 0 is the K/V split.

Detection is by shape: whichever of ``shape[0]`` and ``shape[1]`` equals
2 is the K/V dim. The degenerate case ``num_blocks == 2`` (only seen in
contrived unit-test fixtures) is ambiguous and defaults to
``block_first`` for backward compatibility.

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

import os
import time
from typing import Sequence

import torch

from vllm.v1.personal_context.load import LoadedBlock, LoadedPlan

# TEMP profiling accumulator for scatter phase timing
# (VLLM_PC_TIME_SCATTER=1): validate / detect-layout / copy totals across
# one scatter_loaded_plan call. Reset per call. Remove after.
_SCATTER_PROF: dict = {"validate": 0.0, "copy": 0.0}


def _detect_kv_layout(cache: torch.Tensor) -> str:
    """Identify whether a paged KV cache tensor is ``block_first`` (NHD,
    dim 1 is K/V) or ``kv_first`` (vLLM FA backend, dim 0 is K/V).

    Falls back to ``block_first`` in the degenerate ``num_blocks == 2``
    case so existing CPU test fixtures with tiny ``num_blocks`` keep
    their original semantics.
    """
    if cache.dim() != 5:
        raise ValueError(
            f"kv_cache must be 5-D, got shape {tuple(cache.shape)}"
        )
    if cache.shape[1] == 2:
        return "block_first"
    if cache.shape[0] == 2:
        return "kv_first"
    raise ValueError(
        f"kv_cache shape {tuple(cache.shape)} has neither dim 0 nor "
        f"dim 1 == 2; cannot identify the K/V split."
    )


def _num_blocks(cache: torch.Tensor, layout: str) -> int:
    return cache.shape[1] if layout == "kv_first" else cache.shape[0]


def _write_kv_to_cache(
    cache: torch.Tensor,
    layout: str,
    block_id: int,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if layout == "block_first":
        cache[block_id, 0].copy_(k)
        cache[block_id, 1].copy_(v)
    else:  # "kv_first"
        cache[0, block_id].copy_(k)
        cache[1, block_id].copy_(v)


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
        _write_kv_to_cache(
            cache, _detect_kv_layout(cache), physical_block_id, k, v
        )


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
    # TEMP profiling (VLLM_PC_TIME_SCATTER=1): split validate vs copy to
    # attribute scatter latency. GPU copies are async so the copy phase is
    # synced before timing. Remove with the instrumentation.
    _prof = bool(os.environ.get("VLLM_PC_TIME_SCATTER"))
    if _prof:
        _SCATTER_PROF["validate"] = 0.0
        _SCATTER_PROF["copy"] = 0.0
        _t0 = time.perf_counter()
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
    if _prof:
        _SCATTER_PROF["validate"] = time.perf_counter() - _t0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter()

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
                _write_kv_to_cache(
                    cache, _detect_kv_layout(cache), assignment, k, v
                )
    if _prof:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _SCATTER_PROF["copy"] = time.perf_counter() - _t0
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).info(
            "PC scatter prof: validate=%.1fms copy=%.1fms",
            _SCATTER_PROF["validate"] * 1000,
            _SCATTER_PROF["copy"] * 1000,
        )


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
                f"kv_caches[{layer_idx}] must be 5-D; got shape "
                f"{tuple(cache.shape)}"
            )
        try:
            layout = _detect_kv_layout(cache)
        except ValueError as e:
            raise ValueError(
                f"kv_caches[{layer_idx}]: {e}"
            ) from None
        num_blocks = _num_blocks(cache, layout)
        if not 0 <= physical_block_id < num_blocks:
            raise ValueError(
                f"physical_block_id {physical_block_id} out of range "
                f"for kv_caches[{layer_idx}] with {num_blocks} blocks "
                f"(layout={layout})"
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
