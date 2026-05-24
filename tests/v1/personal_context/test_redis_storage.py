# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``RedisKVStorage``.

Use ``fakeredis`` as an in-process drop-in for ``redis.Redis`` so the
test suite stays hermetic — no real Redis server required. The
RedisKVStorage class accepts an injectable ``client`` arg for exactly
this purpose; production callers pass ``url`` instead.
"""

from __future__ import annotations

import pytest
import torch

fakeredis = pytest.importorskip("fakeredis")

from vllm.v1.personal_context.entry import KVBlock, StoreConfig  # noqa: E402
from vllm.v1.personal_context.redis_storage import (  # noqa: E402
    RedisKVStorage,
)


BLOCK_SIZE = 4
NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8


def _store_config(**overrides) -> StoreConfig:
    base = dict(
        model_id="test",
        dtype=torch.float32,
        layout="NHD",
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
    )
    base.update(overrides)
    return StoreConfig(**base)


def _make_block(old_pos_start: int = 0, seed: int = 0) -> KVBlock:
    g = torch.Generator().manual_seed(seed)
    shape = (BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    keys = [torch.randn(shape, generator=g) for _ in range(NUM_LAYERS)]
    values = [torch.randn(shape, generator=g) for _ in range(NUM_LAYERS)]
    return KVBlock(keys=keys, values=values, old_pos_start=old_pos_start)


def _new_store(**cfg_overrides) -> RedisKVStorage:
    """Fresh fakeredis-backed store with a fresh in-memory server."""
    client = fakeredis.FakeRedis()
    return RedisKVStorage(_store_config(**cfg_overrides), client=client)


# --------------------------- construction ---------------------------


def test_requires_url_or_client():
    with pytest.raises(ValueError, match="url.*client"):
        RedisKVStorage(_store_config())


def test_first_init_writes_config():
    client = fakeredis.FakeRedis()
    store = RedisKVStorage(_store_config(), client=client)
    assert client.get(b"pc:config") is not None
    # Config should round-trip.
    assert store.config == _store_config()


def test_second_init_against_matching_config_ok():
    client = fakeredis.FakeRedis()
    cfg = _store_config()
    RedisKVStorage(cfg, client=client)  # writes config
    # Second instance against same backing store + same config: OK.
    s2 = RedisKVStorage(cfg, client=client)
    assert s2.config == cfg


def test_second_init_against_mismatched_config_raises():
    client = fakeredis.FakeRedis()
    RedisKVStorage(_store_config(num_layers=2), client=client)
    with pytest.raises(ValueError, match="schema mismatch"):
        RedisKVStorage(_store_config(num_layers=4), client=client)


# --------------------------- put / get ---------------------------


def test_put_then_get_roundtrip():
    store = _new_store()
    block = _make_block(seed=42)
    store.put(b"hash-A", block)

    got = store.get(b"hash-A")
    assert got is not None
    for k_orig, k_back in zip(block.keys, got.keys):
        torch.testing.assert_close(k_orig, k_back, rtol=0, atol=0)
    for v_orig, v_back in zip(block.values, got.values):
        torch.testing.assert_close(v_orig, v_back, rtol=0, atol=0)
    assert got.old_pos_start == block.old_pos_start


def test_get_miss_returns_none():
    store = _new_store()
    assert store.get(b"nonexistent") is None


def test_put_overwrites():
    store = _new_store()
    a = _make_block(seed=1)
    b = _make_block(seed=2)
    store.put(b"key", a)
    store.put(b"key", b)
    got = store.get(b"key")
    assert got is not None
    # Second write wins — keys[0] should match b not a.
    torch.testing.assert_close(got.keys[0], b.keys[0], rtol=0, atol=0)
    assert not torch.allclose(got.keys[0], a.keys[0])


# --------------------------- batch lookup ---------------------------


def test_lookup_empty_returns_empty():
    store = _new_store()
    assert store.lookup([]) == []


def test_lookup_preserves_order_and_holes():
    store = _new_store()
    store.put(b"present-1", _make_block(seed=1))
    store.put(b"present-2", _make_block(seed=2))

    results = store.lookup([b"present-1", b"absent", b"present-2", b"also-absent"])
    assert len(results) == 4
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None
    assert results[3] is None


def test_lookup_round_trips_tensors():
    store = _new_store()
    block_a = _make_block(seed=10)
    block_b = _make_block(seed=20, old_pos_start=BLOCK_SIZE)
    store.put(b"a", block_a)
    store.put(b"b", block_b)

    got_a, got_b = store.lookup([b"a", b"b"])
    assert got_a is not None and got_b is not None
    torch.testing.assert_close(got_a.keys[0], block_a.keys[0], rtol=0, atol=0)
    torch.testing.assert_close(got_b.values[1], block_b.values[1], rtol=0, atol=0)
    assert got_b.old_pos_start == BLOCK_SIZE


# --------------------------- contains / len ---------------------------


def test_contains():
    store = _new_store()
    assert b"x" not in store
    store.put(b"x", _make_block())
    assert b"x" in store
    assert b"y" not in store


def test_len_counts_only_kv_blocks():
    """``__len__`` must exclude the ``pc:config`` sentinel key."""
    store = _new_store()
    # Even though pc:config was written on init, len is 0 until first put.
    assert len(store) == 0
    store.put(b"k1", _make_block(seed=1))
    store.put(b"k2", _make_block(seed=2))
    store.put(b"k3", _make_block(seed=3))
    assert len(store) == 3


# --------------------------- validation ---------------------------


def test_put_validates_layer_count():
    store = _new_store()
    block = _make_block()
    # Drop a layer to break the contract.
    bad = KVBlock(
        keys=block.keys[:1],
        values=block.values[:1],
        old_pos_start=block.old_pos_start,
    )
    with pytest.raises(ValueError, match="key tensors"):
        store.put(b"k", bad)


def test_put_validates_shape():
    store = _new_store()
    bad_keys = [torch.randn(8, NUM_KV_HEADS, HEAD_DIM) for _ in range(NUM_LAYERS)]
    bad = KVBlock(
        keys=bad_keys,
        values=_make_block().values,
        old_pos_start=0,
    )
    with pytest.raises(ValueError, match="K shape"):
        store.put(b"k", bad)


def test_put_validates_old_pos_alignment():
    store = _new_store()
    bad = _make_block(old_pos_start=3)  # not a multiple of block_size=4
    with pytest.raises(Exception, match="old_pos_start"):
        store.put(b"k", bad)


# --------------------------- interface parity with InMemoryStorage ---------


def test_interface_parity_with_in_memory():
    """RedisKVStorage and InMemoryStorage share enough surface that a
    caller can swap one for the other without code changes.
    """
    from vllm.v1.personal_context.storage import InMemoryStorage

    in_mem = InMemoryStorage(_store_config())
    redis = _new_store()
    block = _make_block(seed=99)

    # Same put → same get
    in_mem.put(b"k", block)
    redis.put(b"k", block)
    a = in_mem.get(b"k")
    b = redis.get(b"k")
    assert a is not None and b is not None
    torch.testing.assert_close(a.keys[0], b.keys[0], rtol=0, atol=0)

    # Same len / contains semantics
    assert len(in_mem) == len(redis) == 1
    assert (b"k" in in_mem) and (b"k" in redis)
    assert (b"missing" not in in_mem) and (b"missing" not in redis)

    # Same lookup behaviour
    a_lookup = in_mem.lookup([b"k", b"missing"])
    b_lookup = redis.lookup([b"k", b"missing"])
    assert len(a_lookup) == len(b_lookup) == 2
    assert a_lookup[1] is None and b_lookup[1] is None
    assert a_lookup[0] is not None and b_lookup[0] is not None
