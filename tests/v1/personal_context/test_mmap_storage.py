# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``MmapKVStorage``.

Mirrors ``test_redis_storage.py``'s matrix (round-trip, miss, overwrite,
ordered lookup, validation, interface parity) plus mmap-specific cases:
byte-exact tensor round-trip, persistence across reopen, and capacity
growth past the initial slot count.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.personal_context.entry import KVBlock, StoreConfig
from vllm.v1.personal_context.mmap_storage import MmapKVStorage

BLOCK_SIZE = 4
NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8


def _store_config(**overrides) -> StoreConfig:
    base = dict(
        model_id="test",
        dtype=torch.float16,
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
    keys = [
        torch.randn(shape, generator=g).to(torch.float16)
        for _ in range(NUM_LAYERS)
    ]
    values = [
        torch.randn(shape, generator=g).to(torch.float16)
        for _ in range(NUM_LAYERS)
    ]
    return KVBlock(keys=keys, values=values, old_pos_start=old_pos_start)


def _new_store(tmp_path, **cfg_overrides) -> MmapKVStorage:
    return MmapKVStorage(_store_config(**cfg_overrides), root_dir=str(tmp_path))


# --------------------------- construction ---------------------------


def test_first_init_creates_files(tmp_path):
    store = MmapKVStorage(_store_config(), root_dir=str(tmp_path))
    assert (tmp_path / "index.json").exists()
    assert (tmp_path / "blocks.dat").exists()
    assert store.config == _store_config()


def test_reopen_against_matching_config_ok(tmp_path):
    cfg = _store_config()
    MmapKVStorage(cfg, root_dir=str(tmp_path))
    s2 = MmapKVStorage(cfg, root_dir=str(tmp_path))
    assert s2.config == cfg


def test_reopen_against_mismatched_config_raises(tmp_path):
    MmapKVStorage(_store_config(num_layers=2), root_dir=str(tmp_path))
    with pytest.raises(ValueError, match="schema mismatch"):
        MmapKVStorage(_store_config(num_layers=4), root_dir=str(tmp_path))


# --------------------------- put / get ---------------------------


def test_put_then_get_roundtrip(tmp_path):
    store = _new_store(tmp_path)
    block = _make_block(seed=42)
    store.put(b"hash-A", block)

    got = store.get(b"hash-A")
    assert got is not None
    for k_orig, k_back in zip(block.keys, got.keys):
        torch.testing.assert_close(k_orig, k_back, rtol=0, atol=0)
    for v_orig, v_back in zip(block.values, got.values):
        torch.testing.assert_close(v_orig, v_back, rtol=0, atol=0)
    assert got.old_pos_start == block.old_pos_start


def test_get_miss_returns_none(tmp_path):
    store = _new_store(tmp_path)
    assert store.get(b"nonexistent") is None


def test_put_overwrites(tmp_path):
    store = _new_store(tmp_path)
    a = _make_block(seed=1)
    b = _make_block(seed=2)
    store.put(b"key", a)
    store.put(b"key", b)
    got = store.get(b"key")
    assert got is not None
    torch.testing.assert_close(got.keys[0], b.keys[0], rtol=0, atol=0)
    assert not torch.allclose(got.keys[0], a.keys[0])
    # Overwrite must reuse the slot, not grow the store.
    assert len(store) == 1


def test_returned_tensor_is_writable_and_independent(tmp_path):
    """frombuffer over the mapping is read-only; the store must clone so
    the caller gets a writable tensor that doesn't alias the file."""
    store = _new_store(tmp_path)
    store.put(b"k", _make_block(seed=5))
    got = store.get(b"k")
    # Writable (would raise if it still aliased the read-only mmap view).
    got.keys[0].add_(1.0)
    # And the stored copy is unchanged by that mutation.
    again = store.get(b"k")
    assert not torch.allclose(got.keys[0], again.keys[0])


# --------------------------- batch lookup ---------------------------


def test_lookup_empty_returns_empty(tmp_path):
    store = _new_store(tmp_path)
    assert store.lookup([]) == []


def test_lookup_preserves_order_and_holes(tmp_path):
    store = _new_store(tmp_path)
    store.put(b"present-1", _make_block(seed=1))
    store.put(b"present-2", _make_block(seed=2))

    results = store.lookup(
        [b"present-1", b"absent", b"present-2", b"also-absent"]
    )
    assert len(results) == 4
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None
    assert results[3] is None


def test_lookup_round_trips_tensors(tmp_path):
    store = _new_store(tmp_path)
    block_a = _make_block(seed=10)
    block_b = _make_block(seed=20, old_pos_start=BLOCK_SIZE)
    store.put(b"a", block_a)
    store.put(b"b", block_b)

    got_a, got_b = store.lookup([b"a", b"b"])
    assert got_a is not None and got_b is not None
    torch.testing.assert_close(got_a.keys[0], block_a.keys[0], rtol=0, atol=0)
    torch.testing.assert_close(
        got_b.values[1], block_b.values[1], rtol=0, atol=0
    )
    assert got_b.old_pos_start == BLOCK_SIZE


# --------------------------- contains / len ---------------------------


def test_contains(tmp_path):
    store = _new_store(tmp_path)
    assert b"x" not in store
    store.put(b"x", _make_block())
    assert b"x" in store
    assert b"y" not in store


def test_len_counts_blocks(tmp_path):
    store = _new_store(tmp_path)
    assert len(store) == 0
    store.put(b"k1", _make_block(seed=1))
    store.put(b"k2", _make_block(seed=2))
    store.put(b"k3", _make_block(seed=3))
    assert len(store) == 3


# --------------------------- persistence ---------------------------


def test_persists_across_reopen(tmp_path):
    """Blocks written by one instance are visible (byte-exact) to a fresh
    instance mapping the same directory — the persistence guarantee."""
    block = _make_block(seed=77, old_pos_start=BLOCK_SIZE)
    s1 = MmapKVStorage(_store_config(), root_dir=str(tmp_path))
    s1.put(b"persist-me", block)
    s1.close()

    s2 = MmapKVStorage(_store_config(), root_dir=str(tmp_path))
    got = s2.get(b"persist-me")
    assert got is not None
    assert b"persist-me" in s2
    assert len(s2) == 1
    torch.testing.assert_close(got.keys[0], block.keys[0], rtol=0, atol=0)
    torch.testing.assert_close(got.values[1], block.values[1], rtol=0, atol=0)
    assert got.old_pos_start == block.old_pos_start


# --------------------------- capacity growth ---------------------------


def test_grows_past_initial_capacity(tmp_path):
    """Writing more blocks than the initial slot count must trigger a
    remap and keep every prior block byte-exact."""
    from vllm.v1.personal_context.mmap_storage import _INITIAL_SLOTS

    store = _new_store(tmp_path)
    n = _INITIAL_SLOTS + 5  # force at least one doubling
    originals = {}
    for i in range(n):
        blk = _make_block(seed=1000 + i)
        store.put(f"k{i}".encode(), blk)
        originals[i] = blk

    assert len(store) == n
    # Every block — including ones written before the grow — round-trips.
    for i in range(n):
        got = store.get(f"k{i}".encode())
        assert got is not None
        torch.testing.assert_close(
            got.keys[0], originals[i].keys[0], rtol=0, atol=0
        )


# --------------------------- validation ---------------------------


def test_put_validates_layer_count(tmp_path):
    store = _new_store(tmp_path)
    block = _make_block()
    bad = KVBlock(
        keys=block.keys[:1],
        values=block.values[:1],
        old_pos_start=block.old_pos_start,
    )
    with pytest.raises(ValueError, match="key tensors"):
        store.put(b"k", bad)


def test_put_validates_shape(tmp_path):
    store = _new_store(tmp_path)
    bad_keys = [
        torch.randn(8, NUM_KV_HEADS, HEAD_DIM).to(torch.float16)
        for _ in range(NUM_LAYERS)
    ]
    bad = KVBlock(
        keys=bad_keys,
        values=_make_block().values,
        old_pos_start=0,
    )
    with pytest.raises(ValueError, match="K shape"):
        store.put(b"k", bad)


def test_put_validates_old_pos_alignment(tmp_path):
    store = _new_store(tmp_path)
    bad = _make_block(old_pos_start=3)  # not a multiple of block_size=4
    with pytest.raises(Exception, match="old_pos_start"):
        store.put(b"k", bad)


# --------------------------- interface parity ---------------------------


def test_interface_parity_with_in_memory(tmp_path):
    """MmapKVStorage and InMemoryStorage share enough surface that a
    caller can swap one for the other without code changes."""
    from vllm.v1.personal_context.storage import InMemoryStorage

    in_mem = InMemoryStorage(_store_config())
    mmap_store = _new_store(tmp_path)
    block = _make_block(seed=99)

    in_mem.put(b"k", block)
    mmap_store.put(b"k", block)
    a = in_mem.get(b"k")
    b = mmap_store.get(b"k")
    assert a is not None and b is not None
    torch.testing.assert_close(a.keys[0], b.keys[0], rtol=0, atol=0)

    assert len(in_mem) == len(mmap_store) == 1
    assert (b"k" in in_mem) and (b"k" in mmap_store)
    assert (b"missing" not in in_mem) and (b"missing" not in mmap_store)

    a_lookup = in_mem.lookup([b"k", b"missing"])
    b_lookup = mmap_store.lookup([b"k", b"missing"])
    assert len(a_lookup) == len(b_lookup) == 2
    assert a_lookup[1] is None and b_lookup[1] is None
    assert a_lookup[0] is not None and b_lookup[0] is not None
