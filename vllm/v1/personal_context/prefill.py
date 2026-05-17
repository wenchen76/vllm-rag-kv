# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer prefill path for personal-context KV reuse.

When personal-context KV reuse is active, a single prefill request has
K/V from two sources living side-by-side in the paged cache:

    - **Loaded**: K/V at positions assigned by the scheduler whose
      contents came from the personal-context store (delta-RoPE already
      applied by ``load_plan``; tokens NOT forwarded through the model
      this step).
    - **Computed**: K/V at positions where the model forwarded this
      step (system prefix, query, selectively-recomputed chunk tokens).

The Q rows in this step exist only for **computed** positions but must
attend (causally) over the **full** position range. FlashInfer's
``BatchPrefillWithPagedKVCacheWrapper`` supports exactly this shape via
``qo_indptr`` and ``paged_kv_indptr`` ranges that need not match, plus
a ``custom_mask`` that selects which (Q, KV) pairs are valid.

This module provides:

    1. ``build_chunk_aware_mask`` — the flat bool mask FlashInfer wants.
    2. ``build_paged_kv_metadata`` — single-request paged-KV indptr /
       indices / last-page-len tensors.
    3. ``PrefillSetupSpec`` + ``setup_chunk_aware_prefill`` — one-time
       FlashInfer plan-stage call (CPU-side prep, mask packing, tile
       schedule). Returns a planned wrapper.
    4. ``run_chunk_aware_prefill`` — per-layer GPU run call.

FlashInfer is imported lazily — CPU machines can still exercise the
mask + metadata builders, but the actual kernel run requires CUDA +
FlashInfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def build_chunk_aware_mask(
    q_positions: torch.Tensor, kv_positions: torch.Tensor
) -> torch.Tensor:
    """Flat bool mask for FlashInfer's ``custom_mask`` argument.

    A Q row at absolute position ``p`` attends to every KV slot at
    position ``<= p``, regardless of whether the slot is loaded or
    computed. The result is row-major flattened:

        out[i * KV + j] = (q_positions[i] >= kv_positions[j])

    Args:
        q_positions: ``[Q]`` integer tensor of Q absolute positions.
        kv_positions: ``[KV]`` integer tensor of KV absolute positions.

    Returns:
        ``[Q * KV]`` bool tensor, contiguous, ready to pass as
        ``custom_mask`` to ``BatchPrefillWithPagedKVCacheWrapper.plan``.
    """
    if q_positions.dim() != 1 or kv_positions.dim() != 1:
        raise ValueError("position tensors must be 1-D")
    mask_2d = q_positions[:, None] >= kv_positions[None, :]
    return mask_2d.flatten().contiguous()


def build_paged_kv_metadata(
    block_ids: Sequence[int],
    total_kv_tokens: int,
    page_size: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-request paged-KV metadata for FlashInfer.

    Args:
        block_ids: Ordered physical block indices that back this
            request's KV cache (loaded blocks first, then computed
            blocks; order matches absolute position order).
        total_kv_tokens: Number of valid KV tokens; must satisfy
            ``(len(block_ids) - 1) * page_size < total_kv_tokens
            <= len(block_ids) * page_size``.
        page_size: Tokens per page (== store ``block_size``).
        device: Optional torch device for the returned tensors.

    Returns:
        Tuple ``(paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len)``,
        all int32. ``paged_kv_indptr`` has shape ``[2]`` (single
        request: ``[0, num_pages]``); ``paged_kv_indices`` has shape
        ``[num_pages]``; ``paged_kv_last_page_len`` has shape ``[1]``.
    """
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if not block_ids:
        raise ValueError("block_ids must be non-empty")
    num_pages = len(block_ids)
    upper = num_pages * page_size
    lower = (num_pages - 1) * page_size + 1
    if not (lower <= total_kv_tokens <= upper):
        raise ValueError(
            f"total_kv_tokens {total_kv_tokens} must lie in [{lower}, {upper}] "
            f"for {num_pages} pages of size {page_size}"
        )
    last_page_len = total_kv_tokens - (num_pages - 1) * page_size

    indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    indices = torch.tensor(list(block_ids), dtype=torch.int32, device=device)
    last = torch.tensor([last_page_len], dtype=torch.int32, device=device)
    return indptr, indices, last


@dataclass(frozen=True)
class PrefillSetupSpec:
    """Inputs to ``setup_chunk_aware_prefill`` — shared across all layers.

    These are the per-request shape / metadata tensors FlashInfer wants
    at its ``wrapper.plan(...)`` call. ``setup_chunk_aware_prefill``
    consumes a spec once per request; the per-layer Q and KV cache are
    passed to ``run_chunk_aware_prefill`` instead.
    """

    qo_indptr: torch.Tensor
    """``[2]`` int32 — ``[0, num_q_tokens]`` for single-request."""

    paged_kv_indptr: torch.Tensor
    """``[2]`` int32 — ``[0, num_pages]``."""

    paged_kv_indices: torch.Tensor
    """``[num_pages]`` int32 — physical block ids in position order."""

    paged_kv_last_page_len: torch.Tensor
    """``[1]`` int32 — tokens occupying the last page (1..page_size)."""

    custom_mask: torch.Tensor
    """``[num_q_tokens * total_kv_tokens]`` bool — chunk-aware mask."""

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    q_dtype: torch.dtype
    """Q tensor dtype; FlashInfer needs this at plan time."""

    sm_scale: float | None = None
    kv_layout: str = "NHD"


def setup_chunk_aware_prefill(
    workspace_buffer: torch.Tensor,
    spec: PrefillSetupSpec,
):
    """One-time CPU-side FlashInfer setup for this request.

    Constructs a ``BatchPrefillWithPagedKVCacheWrapper``, calls its
    ``plan()`` with the chunk-aware ``custom_mask`` and the sparse-Q
    / full-KV indptrs, and returns the planned wrapper. The returned
    object is reusable across all transformer layers in this forward
    pass via ``run_chunk_aware_prefill``.

    Args:
        workspace_buffer: Scratch buffer FlashInfer needs (typically a
            ~128 MiB ``uint8`` GPU tensor). Reuse across requests.
        spec: Per-request plan-stage inputs.

    Returns:
        A planned ``BatchPrefillWithPagedKVCacheWrapper`` ready for
        per-layer ``run`` calls. Its exact type is FlashInfer's; we
        intentionally do not annotate it so this file can import on
        CPU-only machines.

    Raises:
        ImportError: FlashInfer is not installed.
    """
    try:
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    except ImportError as e:
        raise ImportError(
            "FlashInfer is required for chunk-aware prefill. "
            "Install via `uv pip install flashinfer-python` on a CUDA "
            "platform."
        ) from e

    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, spec.kv_layout
    )
    wrapper.plan(
        qo_indptr=spec.qo_indptr,
        paged_kv_indptr=spec.paged_kv_indptr,
        paged_kv_indices=spec.paged_kv_indices,
        paged_kv_last_page_len=spec.paged_kv_last_page_len,
        num_qo_heads=spec.num_q_heads,
        num_kv_heads=spec.num_kv_heads,
        head_dim_qk=spec.head_dim,
        page_size=spec.page_size,
        causal=False,  # custom_mask supersedes the built-in causal mask
        sm_scale=spec.sm_scale,
        q_data_type=spec.q_dtype,
        custom_mask=spec.custom_mask,
    )
    return wrapper


def run_chunk_aware_prefill(
    wrapper,
    q: torch.Tensor,
    paged_kv_cache: torch.Tensor,
) -> torch.Tensor:
    """Per-layer GPU attention run.

    Args:
        wrapper: Planned wrapper from ``setup_chunk_aware_prefill``.
        q: ``[num_q_tokens, num_q_heads, head_dim]`` — Q for the
            computed positions in this layer.
        paged_kv_cache: KV cache buffer in ``spec.kv_layout`` order.
            Layout convention follows FlashInfer:
                NHD: ``[num_blocks, 2, page_size, num_kv_heads, head_dim]``.
                HND: ``[num_blocks, 2, num_kv_heads, page_size, head_dim]``.
            Index 0 along dim-1 is K, index 1 is V.

    Returns:
        ``[num_q_tokens, num_q_heads, head_dim]`` attention output,
        same dtype as ``q``.
    """
    return wrapper.run(q, paged_kv_cache)
