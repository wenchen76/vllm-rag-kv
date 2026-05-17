# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.personal_context import (
    LoadedBlock,
    LoadedChunk,
    LoadedPlan,
    scatter_loaded_block,
    scatter_loaded_plan,
)
from vllm.v1.personal_context.chunk import Chunk


BLOCK_SIZE = 4
NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8
NUM_PAGES = 6


def _make_kv_caches(
    num_layers: int = NUM_LAYERS,
    num_pages: int = NUM_PAGES,
    page_size: int = BLOCK_SIZE,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Per-layer NHD paged caches, initialised to zeros for clean diffs."""
    shape = (num_pages, 2, page_size, num_kv_heads, head_dim)
    return [torch.zeros(shape, dtype=dtype) for _ in range(num_layers)]


def _make_loaded_block(
    seed: int,
    new_pos_start: int = 0,
    block_size: int = BLOCK_SIZE,
    num_layers: int = NUM_LAYERS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    dtype: torch.dtype = torch.float32,
) -> LoadedBlock:
    g = torch.Generator().manual_seed(seed)
    shape = (block_size, num_kv_heads, head_dim)
    keys = [torch.randn(shape, generator=g, dtype=dtype) for _ in range(num_layers)]
    values = [torch.randn(shape, generator=g, dtype=dtype) for _ in range(num_layers)]
    return LoadedBlock(keys=keys, values=values, new_pos_start=new_pos_start)


def _make_chunk(token_ids: tuple[int, ...] = (0, 1, 2, 3)) -> Chunk:
    return Chunk(token_ids=token_ids, old_pos_start=0)


# ----------------------- scatter_loaded_block -----------------------


def test_scatter_block_writes_k_and_v_to_correct_slot():
    caches = _make_kv_caches()
    block = _make_loaded_block(seed=1)
    scatter_loaded_block(block, caches, physical_block_id=2)

    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(
            caches[layer][2, 0], block.keys[layer], rtol=0, atol=0
        )
        torch.testing.assert_close(
            caches[layer][2, 1], block.values[layer], rtol=0, atol=0
        )


def test_scatter_block_leaves_other_slots_untouched():
    caches = _make_kv_caches()
    block = _make_loaded_block(seed=2)
    scatter_loaded_block(block, caches, physical_block_id=3)

    for layer in range(NUM_LAYERS):
        for page in range(NUM_PAGES):
            if page == 3:
                continue
            assert torch.all(caches[layer][page] == 0)


def test_scatter_block_does_not_alias_loaded_tensors():
    """The cache must contain an independent copy, not a view."""
    caches = _make_kv_caches()
    block = _make_loaded_block(seed=3)
    scatter_loaded_block(block, caches, physical_block_id=0)

    block.keys[0].zero_()
    block.values[0].zero_()
    # Cache slot should still hold the pre-mutation data.
    assert not torch.all(caches[0][0, 0] == 0)
    assert not torch.all(caches[0][0, 1] == 0)


def test_scatter_block_rejects_layer_count_mismatch():
    caches = _make_kv_caches(num_layers=NUM_LAYERS + 1)
    block = _make_loaded_block(seed=4)
    with pytest.raises(ValueError, match="K layers"):
        scatter_loaded_block(block, caches, physical_block_id=0)


def test_scatter_block_rejects_non_5d_cache():
    block = _make_loaded_block(seed=5)
    bad = [torch.zeros(NUM_PAGES, 2, BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM)
           for _ in range(NUM_LAYERS)]
    with pytest.raises(ValueError, match="5-D"):
        scatter_loaded_block(block, bad, physical_block_id=0)


def test_scatter_block_rejects_wrong_kv_pair_dim():
    block = _make_loaded_block(seed=6)
    bad = [torch.zeros(NUM_PAGES, 3, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
           for _ in range(NUM_LAYERS)]
    with pytest.raises(ValueError, match="dim 1 must be 2"):
        scatter_loaded_block(block, bad, physical_block_id=0)


def test_scatter_block_rejects_block_id_out_of_range():
    caches = _make_kv_caches()
    block = _make_loaded_block(seed=7)
    with pytest.raises(ValueError, match="out of range"):
        scatter_loaded_block(block, caches, physical_block_id=NUM_PAGES)
    with pytest.raises(ValueError, match="out of range"):
        scatter_loaded_block(block, caches, physical_block_id=-1)


def test_scatter_block_rejects_shape_mismatch():
    caches = _make_kv_caches()
    block = _make_loaded_block(seed=8, head_dim=HEAD_DIM + 1)
    with pytest.raises(ValueError, match="keys"):
        scatter_loaded_block(block, caches, physical_block_id=0)


def test_scatter_block_rejects_dtype_mismatch():
    caches = _make_kv_caches(dtype=torch.float32)
    block = _make_loaded_block(seed=9, dtype=torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        scatter_loaded_block(block, caches, physical_block_id=0)


# ----------------------- scatter_loaded_plan -----------------------


def _wrap_plan(
    blocks_per_chunk: list[list[LoadedBlock | None]],
) -> LoadedPlan:
    """Wrap raw block lists into a LoadedPlan with placeholder chunks."""
    loaded_chunks = []
    for blocks in blocks_per_chunk:
        loaded_chunks.append(
            LoadedChunk(
                chunk=_make_chunk(),
                new_pos_start=0,
                blocks=tuple(blocks),
            )
        )
    return LoadedPlan(chunks=tuple(loaded_chunks))


def test_scatter_plan_empty_is_noop():
    caches = _make_kv_caches()
    scatter_loaded_plan(_wrap_plan([]), caches, block_assignments=[])
    for layer in caches:
        assert torch.all(layer == 0)


def test_scatter_plan_writes_every_hit_to_assigned_block():
    b0 = _make_loaded_block(seed=10)
    b1 = _make_loaded_block(seed=11)
    plan = _wrap_plan([[b0, b1]])
    caches = _make_kv_caches()

    scatter_loaded_plan(plan, caches, block_assignments=[[1, 4]])

    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(caches[layer][1, 0], b0.keys[layer])
        torch.testing.assert_close(caches[layer][1, 1], b0.values[layer])
        torch.testing.assert_close(caches[layer][4, 0], b1.keys[layer])
        torch.testing.assert_close(caches[layer][4, 1], b1.values[layer])
        for page in (0, 2, 3, 5):
            assert torch.all(caches[layer][page] == 0)


def test_scatter_plan_skips_miss_blocks():
    b0 = _make_loaded_block(seed=12)
    plan = _wrap_plan([[b0, None]])
    caches = _make_kv_caches()

    scatter_loaded_plan(plan, caches, block_assignments=[[1, 4]])

    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(caches[layer][1, 0], b0.keys[layer])
        # Page 4 was "assigned" but the block was a miss → skipped.
        assert torch.all(caches[layer][4] == 0)


def test_scatter_plan_skips_none_assignment():
    """Caller passing assignment=None must not write that block."""
    b0 = _make_loaded_block(seed=13)
    b1 = _make_loaded_block(seed=14)
    plan = _wrap_plan([[b0, b1]])
    caches = _make_kv_caches()

    scatter_loaded_plan(plan, caches, block_assignments=[[2, None]])

    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(caches[layer][2, 0], b0.keys[layer])
        for page in (0, 1, 3, 4, 5):
            assert torch.all(caches[layer][page] == 0)


def test_scatter_plan_multi_chunk():
    a0 = _make_loaded_block(seed=20)
    c0 = _make_loaded_block(seed=21)
    c1 = _make_loaded_block(seed=22)
    plan = _wrap_plan([[a0], [c0, c1]])
    caches = _make_kv_caches()

    scatter_loaded_plan(plan, caches, block_assignments=[[0], [3, 5]])

    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(caches[layer][0, 0], a0.keys[layer])
        torch.testing.assert_close(caches[layer][3, 0], c0.keys[layer])
        torch.testing.assert_close(caches[layer][5, 0], c1.keys[layer])


def test_scatter_plan_rejects_outer_length_mismatch():
    plan = _wrap_plan([[_make_loaded_block(seed=30)]])
    caches = _make_kv_caches()
    with pytest.raises(ValueError, match="block_assignments has"):
        scatter_loaded_plan(plan, caches, block_assignments=[[0], [1]])


def test_scatter_plan_rejects_inner_length_mismatch():
    plan = _wrap_plan([[_make_loaded_block(seed=31), None]])
    caches = _make_kv_caches()
    with pytest.raises(ValueError, match="chunk 0"):
        scatter_loaded_plan(plan, caches, block_assignments=[[0]])


def test_scatter_plan_validates_all_blocks_before_writing():
    """A bad assignment in a later block must abort before any write."""
    good = _make_loaded_block(seed=40)
    bad = _make_loaded_block(seed=41, head_dim=HEAD_DIM + 1)
    plan = _wrap_plan([[good, bad]])
    caches = _make_kv_caches()
    before = [c.clone() for c in caches]

    with pytest.raises(ValueError):
        scatter_loaded_plan(plan, caches, block_assignments=[[0, 1]])

    # No partial state — both layers/pages must be unchanged.
    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(caches[layer], before[layer])


def test_scatter_plan_miss_with_none_assignment_does_not_validate():
    """If a block is a miss AND its assignment is None, validation skips it."""
    bad_but_skipped = None  # miss
    plan = _wrap_plan([[bad_but_skipped]])
    caches = _make_kv_caches()
    scatter_loaded_plan(plan, caches, block_assignments=[[None]])
    for layer in caches:
        assert torch.all(layer == 0)
