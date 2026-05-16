# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses

import pytest
import torch

from vllm.v1.personal_context import (
    AlignmentError,
    Chunk,
    InMemoryStorage,
    KVBlock,
    PersonalContextConnector,
    ReusePlan,
    StoreConfig,
)


BLOCK_SIZE = 4


@pytest.fixture
def config() -> StoreConfig:
    return StoreConfig(
        model_id="test-model",
        dtype=torch.float16,
        layout="NHD",
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=BLOCK_SIZE,
    )


def _make_block(config: StoreConfig, old_pos: int) -> KVBlock:
    shape = (config.block_size, config.num_kv_heads, config.head_dim)
    return KVBlock(
        keys=[torch.zeros(shape, dtype=config.dtype)
              for _ in range(config.num_layers)],
        values=[torch.zeros(shape, dtype=config.dtype)
                for _ in range(config.num_layers)],
        old_pos_start=old_pos,
    )


def _store_chunk(
    config: StoreConfig,
    storage: InMemoryStorage,
    chunk: Chunk,
    which: list[bool] | None = None,
) -> None:
    """Put blocks for ``chunk`` into ``storage``; ``which[i]=False`` skips i."""
    hashes = chunk.block_hashes(config.block_size)
    for i, key in enumerate(hashes):
        if which is None or which[i]:
            block = _make_block(
                config, old_pos=chunk.old_pos_start + i * config.block_size
            )
            storage.put(key, block)


def test_empty_plan_returns_zero_free_tokens(config):
    connector = PersonalContextConnector(InMemoryStorage(config))
    result = connector.lookup(ReusePlan())
    assert result.chunks == ()
    assert result.total_hit_blocks == 0
    assert result.free_tokens == 0
    assert result.block_size == config.block_size


def test_full_miss_zero_free_tokens(config):
    connector = PersonalContextConnector(InMemoryStorage(config))
    chunk = Chunk(token_ids=tuple(range(8)), old_pos_start=0)
    result = connector.lookup(ReusePlan(chunks=(chunk,)))
    assert len(result.chunks) == 1
    cl = result.chunks[0]
    assert cl.chunk is chunk
    assert len(cl.blocks) == 2
    assert all(not b.hit for b in cl.blocks)
    assert cl.hit_count == 0
    assert cl.all_miss is True
    assert cl.all_hit is False
    assert result.total_hit_blocks == 0
    assert result.free_tokens == 0


def test_full_hit_frees_all_tokens(config):
    storage = InMemoryStorage(config)
    chunk = Chunk(token_ids=tuple(range(8)), old_pos_start=0)
    _store_chunk(config, storage, chunk)
    result = PersonalContextConnector(storage).lookup(ReusePlan(chunks=(chunk,)))
    cl = result.chunks[0]
    assert cl.hit_count == 2
    assert cl.all_hit is True
    assert cl.all_miss is False
    assert all(b.hit for b in cl.blocks)
    assert result.total_hit_blocks == 2
    assert result.free_tokens == 2 * BLOCK_SIZE


def test_partial_hit_within_chunk(config):
    storage = InMemoryStorage(config)
    chunk = Chunk(token_ids=tuple(range(12)), old_pos_start=0)
    _store_chunk(config, storage, chunk, which=[True, False, True])
    result = PersonalContextConnector(storage).lookup(ReusePlan(chunks=(chunk,)))
    cl = result.chunks[0]
    assert [b.hit for b in cl.blocks] == [True, False, True]
    assert cl.hit_count == 2
    assert cl.all_hit is False
    assert cl.all_miss is False
    assert result.free_tokens == 2 * BLOCK_SIZE


def test_multi_chunk_mixed_hits(config):
    storage = InMemoryStorage(config)
    c0 = Chunk(token_ids=tuple(range(8)), old_pos_start=0)
    c1 = Chunk(token_ids=tuple(range(100, 108)), old_pos_start=64)
    c2 = Chunk(token_ids=tuple(range(200, 204)), old_pos_start=128)
    _store_chunk(config, storage, c0)
    _store_chunk(config, storage, c2)
    result = PersonalContextConnector(storage).lookup(
        ReusePlan(chunks=(c0, c1, c2))
    )
    assert len(result.chunks) == 3
    assert result.chunks[0].all_hit
    assert result.chunks[1].all_miss
    assert result.chunks[2].all_hit
    assert result.total_hit_blocks == 3
    assert result.free_tokens == 3 * BLOCK_SIZE


def test_misaligned_length_propagates_alignment_error(config):
    connector = PersonalContextConnector(InMemoryStorage(config))
    bad = Chunk(token_ids=(1, 2, 3), old_pos_start=0)
    with pytest.raises(AlignmentError):
        connector.lookup(ReusePlan(chunks=(bad,)))


def test_misaligned_start_propagates_alignment_error(config):
    connector = PersonalContextConnector(InMemoryStorage(config))
    bad = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=1)
    with pytest.raises(AlignmentError):
        connector.lookup(ReusePlan(chunks=(bad,)))


def test_block_lookup_carries_key_and_block(config):
    storage = InMemoryStorage(config)
    chunk = Chunk(token_ids=tuple(range(8)), old_pos_start=0)
    hashes = chunk.block_hashes(config.block_size)
    _store_chunk(config, storage, chunk, which=[True, False])
    result = PersonalContextConnector(storage).lookup(ReusePlan(chunks=(chunk,)))
    cl = result.chunks[0]
    assert cl.blocks[0].key == hashes[0]
    assert cl.blocks[0].block is not None
    assert cl.blocks[0].hit is True
    assert cl.blocks[1].key == hashes[1]
    assert cl.blocks[1].block is None
    assert cl.blocks[1].hit is False


def test_connector_exposes_block_size(config):
    connector = PersonalContextConnector(InMemoryStorage(config))
    assert connector.block_size == config.block_size


def test_plan_lookup_is_frozen(config):
    result = PersonalContextConnector(InMemoryStorage(config)).lookup(ReusePlan())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.block_size = 999  # type: ignore[misc]


def test_chunk_lookup_is_frozen(config):
    storage = InMemoryStorage(config)
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    result = PersonalContextConnector(storage).lookup(ReusePlan(chunks=(chunk,)))
    cl = result.chunks[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cl.blocks = ()  # type: ignore[misc]


def test_block_lookup_hit_property(config):
    storage = InMemoryStorage(config)
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(config, storage, chunk)
    result = PersonalContextConnector(storage).lookup(ReusePlan(chunks=(chunk,)))
    b = result.chunks[0].blocks[0]
    assert b.hit is True
    assert b.block is not None
