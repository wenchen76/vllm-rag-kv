# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import importlib
import importlib.util

import pytest
import torch

from vllm.v1.personal_context import (
    PrefillSetupSpec,
    build_chunk_aware_mask,
    build_paged_kv_metadata,
    run_chunk_aware_prefill,
    setup_chunk_aware_prefill,
)


# ----------------------------- mask builder -----------------------------


def test_mask_is_lower_triangular_when_q_equals_kv():
    positions = torch.arange(4, dtype=torch.int64)
    mask = build_chunk_aware_mask(positions, positions)
    assert mask.dtype == torch.bool
    assert mask.shape == (16,)
    # 4x4 lower triangular when reshaped.
    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool)).flatten()
    assert torch.equal(mask, expected)


def test_mask_sparse_q_only_late_positions():
    """Q rows are a strict subset of KV positions (typical sparse-Q case)."""
    q_positions = torch.tensor([5, 6, 7], dtype=torch.int64)
    kv_positions = torch.arange(8, dtype=torch.int64)
    mask = build_chunk_aware_mask(q_positions, kv_positions)
    assert mask.shape == (3 * 8,)
    mask_2d = mask.view(3, 8)
    # q=5 attends to kv=0..5; q=6 attends to 0..6; q=7 attends to 0..7.
    expected = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask_2d, expected)


def test_mask_non_contiguous_q_positions():
    """Q positions need not be contiguous (selected + query mix)."""
    q_positions = torch.tensor([2, 5, 7], dtype=torch.int64)
    kv_positions = torch.arange(8, dtype=torch.int64)
    mask = build_chunk_aware_mask(q_positions, kv_positions).view(3, 8)
    assert mask[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert mask[1].tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
    assert mask[2].tolist() == [1, 1, 1, 1, 1, 1, 1, 1]


def test_mask_handles_non_monotonic_kv_positions():
    """KV positions are only compared by value, not assumed sorted."""
    q_positions = torch.tensor([10], dtype=torch.int64)
    kv_positions = torch.tensor([3, 11, 5], dtype=torch.int64)
    mask = build_chunk_aware_mask(q_positions, kv_positions)
    assert mask.tolist() == [True, False, True]


def test_mask_q_higher_than_all_kv():
    """A Q row higher than every KV position attends to everything."""
    q_positions = torch.tensor([100], dtype=torch.int64)
    kv_positions = torch.arange(4, dtype=torch.int64)
    mask = build_chunk_aware_mask(q_positions, kv_positions)
    assert mask.tolist() == [True, True, True, True]


def test_mask_rejects_non_1d():
    bad = torch.zeros(2, 3, dtype=torch.int64)
    good = torch.arange(3, dtype=torch.int64)
    with pytest.raises(ValueError, match="1-D"):
        build_chunk_aware_mask(bad, good)
    with pytest.raises(ValueError, match="1-D"):
        build_chunk_aware_mask(good, bad)


def test_mask_is_contiguous():
    q = torch.arange(3, dtype=torch.int64)
    kv = torch.arange(5, dtype=torch.int64)
    mask = build_chunk_aware_mask(q, kv)
    assert mask.is_contiguous()


# --------------------------- paged-KV metadata ---------------------------


def test_paged_metadata_full_pages():
    indptr, indices, last = build_paged_kv_metadata(
        block_ids=[7, 3, 9], total_kv_tokens=12, page_size=4
    )
    assert indptr.tolist() == [0, 3]
    assert indices.tolist() == [7, 3, 9]
    assert last.tolist() == [4]
    assert indptr.dtype == torch.int32
    assert indices.dtype == torch.int32
    assert last.dtype == torch.int32


def test_paged_metadata_partial_last_page():
    indptr, indices, last = build_paged_kv_metadata(
        block_ids=[1, 2, 3], total_kv_tokens=10, page_size=4
    )
    assert indptr.tolist() == [0, 3]
    assert indices.tolist() == [1, 2, 3]
    assert last.tolist() == [2]


def test_paged_metadata_single_page():
    indptr, indices, last = build_paged_kv_metadata(
        block_ids=[42], total_kv_tokens=3, page_size=4
    )
    assert indptr.tolist() == [0, 1]
    assert indices.tolist() == [42]
    assert last.tolist() == [3]


def test_paged_metadata_rejects_zero_pages():
    with pytest.raises(ValueError, match="non-empty"):
        build_paged_kv_metadata(block_ids=[], total_kv_tokens=0, page_size=4)


def test_paged_metadata_rejects_bad_page_size():
    with pytest.raises(ValueError, match="page_size"):
        build_paged_kv_metadata(block_ids=[1], total_kv_tokens=1, page_size=0)


def test_paged_metadata_rejects_under_filled_last_page():
    """total_kv_tokens cannot fall on a page boundary below the last page."""
    with pytest.raises(ValueError):
        # 2 pages of 4 ⇒ total must be in [5, 8]; 4 means a full page-1 with
        # zero tokens in page-2, which is invalid for a 2-page request.
        build_paged_kv_metadata(
            block_ids=[1, 2], total_kv_tokens=4, page_size=4
        )


def test_paged_metadata_rejects_overflow():
    with pytest.raises(ValueError):
        build_paged_kv_metadata(
            block_ids=[1, 2], total_kv_tokens=9, page_size=4
        )


# --------------------------- PrefillSetupSpec ---------------------------


def _example_spec(num_q: int = 3, num_kv: int = 8, page_size: int = 4):
    q_pos = torch.arange(num_kv - num_q, num_kv, dtype=torch.int64)
    kv_pos = torch.arange(num_kv, dtype=torch.int64)
    mask = build_chunk_aware_mask(q_pos, kv_pos)
    num_pages = (num_kv + page_size - 1) // page_size
    indptr, indices, last = build_paged_kv_metadata(
        block_ids=list(range(num_pages)),
        total_kv_tokens=num_kv,
        page_size=page_size,
    )
    return PrefillSetupSpec(
        qo_indptr=torch.tensor([0, num_q], dtype=torch.int32),
        paged_kv_indptr=indptr,
        paged_kv_indices=indices,
        paged_kv_last_page_len=last,
        custom_mask=mask,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        page_size=page_size,
        q_dtype=torch.float16,
    )


def test_spec_is_frozen():
    spec = _example_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.num_q_heads = 8  # type: ignore[misc]


def test_spec_defaults():
    spec = _example_spec()
    assert spec.sm_scale is None
    assert spec.kv_layout == "NHD"


# ---------------- FlashInfer call sites: lazy import contract ----------------

_HAS_FLASHINFER = importlib.util.find_spec("flashinfer") is not None


@pytest.mark.skipif(
    _HAS_FLASHINFER, reason="FlashInfer present — error path not applicable"
)
def test_setup_raises_helpful_importerror_without_flashinfer():
    spec = _example_spec()
    workspace = torch.empty(1, dtype=torch.uint8)
    with pytest.raises(ImportError, match="FlashInfer"):
        setup_chunk_aware_prefill(workspace, spec)


@pytest.mark.skipif(
    not (_HAS_FLASHINFER and torch.cuda.is_available()),
    reason="requires CUDA + FlashInfer",
)
def test_setup_and_run_roundtrip_on_cuda():
    """End-to-end smoke: tiny paged cache, sparse Q, run returns right shape."""
    device = torch.device("cuda")
    page_size = 4
    num_kv = 8
    num_q = 3
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    dtype = torch.float16

    q_pos = torch.arange(num_kv - num_q, num_kv, device=device, dtype=torch.int64)
    kv_pos = torch.arange(num_kv, device=device, dtype=torch.int64)
    mask = build_chunk_aware_mask(q_pos, kv_pos)
    num_pages = (num_kv + page_size - 1) // page_size
    indptr, indices, last = build_paged_kv_metadata(
        block_ids=list(range(num_pages)),
        total_kv_tokens=num_kv,
        page_size=page_size,
        device=device,
    )
    spec = PrefillSetupSpec(
        qo_indptr=torch.tensor([0, num_q], dtype=torch.int32, device=device),
        paged_kv_indptr=indptr,
        paged_kv_indices=indices,
        paged_kv_last_page_len=last,
        custom_mask=mask,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_dtype=dtype,
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = setup_chunk_aware_prefill(workspace, spec)

    q = torch.randn(num_q, num_q_heads, head_dim, dtype=dtype, device=device)
    # NHD layout: [num_blocks, 2, page_size, num_kv_heads, head_dim]
    kv_cache = torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    out = run_chunk_aware_prefill(wrapper, q, kv_cache)
    assert out.shape == (num_q, num_q_heads, head_dim)
    assert out.dtype == dtype
