# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.personal_context import (
    AlignmentError,
    InMemoryStorage,
    KVBlock,
    StoreConfig,
    hash_block,
)


@pytest.fixture
def config() -> StoreConfig:
    return StoreConfig(
        model_id="test-model",
        dtype=torch.float16,
        layout="NHD",
        num_layers=4,
        num_kv_heads=8,
        head_dim=64,
        block_size=16,
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


def test_put_get_roundtrip(config):
    store = InMemoryStorage(config)
    block = _make_block(config, old_pos=0)
    key = hash_block(list(range(config.block_size)))
    store.put(key, block)
    assert key in store
    assert store.get(key) is block
    assert len(store) == 1


def test_missing_key_returns_none(config):
    store = InMemoryStorage(config)
    assert store.get(b"\x00" * 32) is None
    assert b"\x00" * 32 not in store
    assert len(store) == 0


def test_lookup_batched_preserves_order(config):
    store = InMemoryStorage(config)
    keys = []
    for i in range(3):
        block = _make_block(config, old_pos=i * config.block_size)
        key = hash_block(list(range(i, i + config.block_size)))
        store.put(key, block)
        keys.append(key)
    miss = b"\xff" * 32
    results = store.lookup([keys[0], miss, keys[2], keys[1]])
    assert results[0] is not None and results[0].old_pos_start == 0
    assert results[1] is None
    assert results[2].old_pos_start == 2 * config.block_size
    assert results[3].old_pos_start == config.block_size


def test_hash_deterministic():
    tokens = [1, 2, 3, 42, 99]
    assert hash_block(tokens) == hash_block(tokens)


def test_hash_differs_by_content():
    assert hash_block([1, 2, 3]) != hash_block([1, 2, 4])
    assert hash_block([1, 2, 3]) != hash_block([1, 2])
    assert hash_block([1, 2, 3]) != hash_block([3, 2, 1])


def test_hash_differs_by_salt():
    tokens = [1, 2, 3]
    assert hash_block(tokens, salt=b"a") != hash_block(tokens, salt=b"b")
    assert hash_block(tokens, salt=b"a") != hash_block(tokens)


def test_hash_is_position_independent():
    # Independent calls with identical tokens must agree. This is the
    # property that makes the schema usable for RAG chunk reuse - distinct
    # from vLLM's prefix-rolling block hash, which depends on prior context.
    assert hash_block([10, 20, 30, 40]) == hash_block([10, 20, 30, 40])


def test_store_config_is_hashable(config):
    assert hash(config) == hash(config)


def _good_block(config: StoreConfig) -> KVBlock:
    return _make_block(config, old_pos=0)


def test_put_rejects_wrong_num_layers(config):
    store = InMemoryStorage(config)
    block = _good_block(config)
    block.keys.pop()
    with pytest.raises(ValueError, match="key tensors"):
        store.put(b"k", block)


def test_put_rejects_wrong_num_value_layers(config):
    store = InMemoryStorage(config)
    block = _good_block(config)
    block.values.pop()
    with pytest.raises(ValueError, match="value tensors"):
        store.put(b"k", block)


def test_put_rejects_wrong_block_size(config):
    store = InMemoryStorage(config)
    bad_shape = (config.block_size + 1, config.num_kv_heads, config.head_dim)
    block = KVBlock(
        keys=[torch.zeros(bad_shape, dtype=config.dtype)
              for _ in range(config.num_layers)],
        values=[torch.zeros(bad_shape, dtype=config.dtype)
                for _ in range(config.num_layers)],
        old_pos_start=0,
    )
    with pytest.raises(ValueError, match="shape"):
        store.put(b"k", block)


def test_put_rejects_wrong_num_kv_heads(config):
    store = InMemoryStorage(config)
    bad_shape = (config.block_size, config.num_kv_heads + 1, config.head_dim)
    block = KVBlock(
        keys=[torch.zeros(bad_shape, dtype=config.dtype)
              for _ in range(config.num_layers)],
        values=[torch.zeros(bad_shape, dtype=config.dtype)
                for _ in range(config.num_layers)],
        old_pos_start=0,
    )
    with pytest.raises(ValueError, match="shape"):
        store.put(b"k", block)


def test_put_rejects_wrong_head_dim(config):
    store = InMemoryStorage(config)
    bad_shape = (config.block_size, config.num_kv_heads, config.head_dim + 1)
    block = KVBlock(
        keys=[torch.zeros(bad_shape, dtype=config.dtype)
              for _ in range(config.num_layers)],
        values=[torch.zeros(bad_shape, dtype=config.dtype)
                for _ in range(config.num_layers)],
        old_pos_start=0,
    )
    with pytest.raises(ValueError, match="shape"):
        store.put(b"k", block)


def test_put_rejects_wrong_dtype(config):
    store = InMemoryStorage(config)
    shape = (config.block_size, config.num_kv_heads, config.head_dim)
    block = KVBlock(
        keys=[torch.zeros(shape, dtype=torch.float32)
              for _ in range(config.num_layers)],
        values=[torch.zeros(shape, dtype=torch.float32)
                for _ in range(config.num_layers)],
        old_pos_start=0,
    )
    with pytest.raises(ValueError, match="dtype"):
        store.put(b"k", block)


def test_put_rejects_misaligned_old_pos_start(config):
    store = InMemoryStorage(config)
    block = _make_block(config, old_pos=config.block_size + 1)
    with pytest.raises(AlignmentError, match="old_pos_start"):
        store.put(b"k", block)


def test_put_rejects_negative_old_pos_start(config):
    store = InMemoryStorage(config)
    block = _make_block(config, old_pos=-config.block_size)
    with pytest.raises(AlignmentError, match="old_pos_start"):
        store.put(b"k", block)
