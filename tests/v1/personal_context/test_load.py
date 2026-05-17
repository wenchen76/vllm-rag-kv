# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.personal_context import (
    AlignmentError,
    Chunk,
    InMemoryStorage,
    KVBlock,
    LoadedBlock,
    PersonalContextConnector,
    ReusePlan,
    StoreConfig,
    apply_rope_at_positions,
    load_plan,
)


BLOCK_SIZE = 4
NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8


@pytest.fixture
def config() -> StoreConfig:
    # fp32 for exact composition-law verification.
    return StoreConfig(
        model_id="test-model",
        dtype=torch.float32,
        layout="NHD",
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
    )


def _random_block_kv(seed: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return per-layer pre-RoPE K and per-layer V for one block."""
    g = torch.Generator().manual_seed(seed)
    shape = (BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    keys = [torch.randn(shape, generator=g) for _ in range(NUM_LAYERS)]
    values = [torch.randn(shape, generator=g) for _ in range(NUM_LAYERS)]
    return keys, values


def _store_block_at(
    storage: InMemoryStorage,
    key: bytes,
    old_pos: int,
    pre_rope_keys: list[torch.Tensor],
    values: list[torch.Tensor],
) -> None:
    """Apply RoPE at ``old_pos`` to each K layer and put as a KVBlock."""
    positions = torch.arange(
        old_pos, old_pos + BLOCK_SIZE, dtype=torch.float32
    )
    rotated = [apply_rope_at_positions(k, positions) for k in pre_rope_keys]
    storage.put(
        key,
        KVBlock(
            keys=rotated,
            values=[v.clone() for v in values],
            old_pos_start=old_pos,
        ),
    )


def test_empty_plan_returns_empty_loaded_plan(config):
    connector = PersonalContextConnector(InMemoryStorage(config))
    result = load_plan(connector.lookup(ReusePlan()), new_pos_starts=())
    assert result.chunks == ()


def test_full_miss_yields_all_none(config):
    connector = PersonalContextConnector(InMemoryStorage(config))
    chunk = Chunk(token_ids=tuple(range(8)), old_pos_start=0)
    plan_lookup = connector.lookup(ReusePlan(chunks=(chunk,)))
    loaded = load_plan(plan_lookup, new_pos_starts=(64,))
    assert len(loaded.chunks) == 1
    lc = loaded.chunks[0]
    assert lc.chunk is chunk
    assert lc.new_pos_start == 64
    assert len(lc.blocks) == 2
    assert all(b is None for b in lc.blocks)


def test_full_hit_roundtrip_matches_oracle(config):
    """Loaded K at new_pos == RoPE(unrotated K, new positions), per layer."""
    storage = InMemoryStorage(config)
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    keys_unrot, values = _random_block_kv(seed=1)
    hash_ = chunk.block_hashes(BLOCK_SIZE)[0]
    _store_block_at(storage, hash_, old_pos=0, pre_rope_keys=keys_unrot,
                    values=values)

    new_pos = 128
    plan_lookup = PersonalContextConnector(storage).lookup(
        ReusePlan(chunks=(chunk,))
    )
    loaded = load_plan(plan_lookup, new_pos_starts=(new_pos,))

    lb = loaded.chunks[0].blocks[0]
    assert isinstance(lb, LoadedBlock)
    assert lb.new_pos_start == new_pos
    new_positions = torch.arange(
        new_pos, new_pos + BLOCK_SIZE, dtype=torch.float32
    )
    for layer in range(NUM_LAYERS):
        expected = apply_rope_at_positions(keys_unrot[layer], new_positions)
        torch.testing.assert_close(
            lb.keys[layer], expected, rtol=1e-5, atol=1e-5
        )


def test_values_are_cloned_not_aliased(config):
    storage = InMemoryStorage(config)
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    keys_unrot, values = _random_block_kv(seed=2)
    hash_ = chunk.block_hashes(BLOCK_SIZE)[0]
    _store_block_at(storage, hash_, old_pos=0, pre_rope_keys=keys_unrot,
                    values=values)

    plan_lookup = PersonalContextConnector(storage).lookup(
        ReusePlan(chunks=(chunk,))
    )
    loaded = load_plan(plan_lookup, new_pos_starts=(64,))
    lb = loaded.chunks[0].blocks[0]
    assert lb is not None

    stored = storage.get(hash_)
    for layer in range(NUM_LAYERS):
        # Same values...
        torch.testing.assert_close(
            lb.values[layer], stored.values[layer], rtol=0, atol=0
        )
        # ...but a different tensor object (clone, not alias).
        assert lb.values[layer].data_ptr() != stored.values[layer].data_ptr()


def test_partial_hit_mixes_none_and_loaded_block(config):
    storage = InMemoryStorage(config)
    chunk = Chunk(token_ids=tuple(range(12)), old_pos_start=0)
    hashes = chunk.block_hashes(BLOCK_SIZE)
    keys_unrot, values = _random_block_kv(seed=3)
    # Store only blocks 0 and 2.
    _store_block_at(storage, hashes[0], old_pos=0,
                    pre_rope_keys=keys_unrot, values=values)
    _store_block_at(storage, hashes[2], old_pos=2 * BLOCK_SIZE,
                    pre_rope_keys=keys_unrot, values=values)

    plan_lookup = PersonalContextConnector(storage).lookup(
        ReusePlan(chunks=(chunk,))
    )
    loaded = load_plan(plan_lookup, new_pos_starts=(64,))
    blocks = loaded.chunks[0].blocks
    assert isinstance(blocks[0], LoadedBlock)
    assert blocks[1] is None
    assert isinstance(blocks[2], LoadedBlock)


def test_per_block_new_pos_start_arithmetic(config):
    storage = InMemoryStorage(config)
    chunk = Chunk(token_ids=tuple(range(12)), old_pos_start=0)
    hashes = chunk.block_hashes(BLOCK_SIZE)
    keys_unrot, values = _random_block_kv(seed=4)
    for i, h in enumerate(hashes):
        _store_block_at(storage, h, old_pos=i * BLOCK_SIZE,
                        pre_rope_keys=keys_unrot, values=values)

    new_pos = 64
    plan_lookup = PersonalContextConnector(storage).lookup(
        ReusePlan(chunks=(chunk,))
    )
    loaded = load_plan(plan_lookup, new_pos_starts=(new_pos,))
    starts = [b.new_pos_start for b in loaded.chunks[0].blocks]
    assert starts == [new_pos, new_pos + BLOCK_SIZE, new_pos + 2 * BLOCK_SIZE]


def test_multi_chunk(config):
    storage = InMemoryStorage(config)
    c0 = Chunk(token_ids=tuple(range(8)), old_pos_start=0)
    c1 = Chunk(token_ids=tuple(range(100, 104)), old_pos_start=64)
    keys_unrot, values = _random_block_kv(seed=5)
    for i, h in enumerate(c0.block_hashes(BLOCK_SIZE)):
        _store_block_at(storage, h, old_pos=i * BLOCK_SIZE,
                        pre_rope_keys=keys_unrot, values=values)
    # c1 intentionally NOT stored — should be all-miss.

    plan_lookup = PersonalContextConnector(storage).lookup(
        ReusePlan(chunks=(c0, c1))
    )
    loaded = load_plan(plan_lookup, new_pos_starts=(128, 256))
    assert len(loaded.chunks) == 2
    assert loaded.chunks[0].new_pos_start == 128
    assert all(isinstance(b, LoadedBlock) for b in loaded.chunks[0].blocks)
    assert loaded.chunks[1].new_pos_start == 256
    assert all(b is None for b in loaded.chunks[1].blocks)


def test_new_pos_starts_length_mismatch_raises(config):
    connector = PersonalContextConnector(InMemoryStorage(config))
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    plan_lookup = connector.lookup(ReusePlan(chunks=(chunk,)))
    with pytest.raises(ValueError, match="new_pos_starts"):
        load_plan(plan_lookup, new_pos_starts=(0, BLOCK_SIZE))


def test_misaligned_new_pos_raises_alignment_error(config):
    connector = PersonalContextConnector(InMemoryStorage(config))
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    plan_lookup = connector.lookup(ReusePlan(chunks=(chunk,)))
    with pytest.raises(AlignmentError):
        load_plan(plan_lookup, new_pos_starts=(1,))


def test_negative_new_pos_raises_alignment_error(config):
    connector = PersonalContextConnector(InMemoryStorage(config))
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    plan_lookup = connector.lookup(ReusePlan(chunks=(chunk,)))
    with pytest.raises(AlignmentError):
        load_plan(plan_lookup, new_pos_starts=(-BLOCK_SIZE,))


def test_rope_theta_threads_through_load(config):
    """A custom rope_theta must change the loaded K for non-zero delta."""
    storage = InMemoryStorage(config)
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    keys_unrot, values = _random_block_kv(seed=6)
    hash_ = chunk.block_hashes(BLOCK_SIZE)[0]
    _store_block_at(storage, hash_, old_pos=0, pre_rope_keys=keys_unrot,
                    values=values)

    plan_lookup = PersonalContextConnector(storage).lookup(
        ReusePlan(chunks=(chunk,))
    )
    a = load_plan(plan_lookup, new_pos_starts=(64,), rope_theta=10000.0)
    b = load_plan(plan_lookup, new_pos_starts=(64,), rope_theta=500000.0)
    assert not torch.allclose(
        a.chunks[0].blocks[0].keys[0], b.chunks[0].blocks[0].keys[0]
    )
