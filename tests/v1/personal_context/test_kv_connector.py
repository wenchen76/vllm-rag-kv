# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for PersonalContextKVConnector (Phase 7 Step 1-2)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.personal_context_connector import (
    PersonalContextConnectorMetadata,
    PersonalContextKVConnector,
)
from vllm.v1.personal_context import (
    Chunk,
    InMemoryStorage,
    KVBlock,
    StoreConfig,
    apply_rope_at_positions,
)


BLOCK_SIZE = 4
NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8


# ----------------------- fixtures / helpers -----------------------


def _fake_vllm_config(block_size: int = BLOCK_SIZE):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="PersonalContextKVConnector",
            extra_config={},
        ),
        cache_config=SimpleNamespace(block_size=block_size),
    )


def _fake_kv_cache_config():
    return SimpleNamespace()


def _store_config(block_size: int = BLOCK_SIZE) -> StoreConfig:
    return StoreConfig(
        model_id="test",
        dtype=torch.float32,
        layout="NHD",
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=block_size,
    )


def _connector(bind: bool = True) -> PersonalContextKVConnector:
    c = PersonalContextKVConnector(
        _fake_vllm_config(),
        KVConnectorRole.SCHEDULER,
        _fake_kv_cache_config(),
    )
    if bind:
        c.bind_storage(InMemoryStorage(_store_config()))
    return c


def _request(kv_transfer_params=None, request_id: str = "req-1"):
    return SimpleNamespace(
        request_id=request_id,
        kv_transfer_params=kv_transfer_params,
    )


def _plan_dict(chunks):
    """Serialise a list of Chunks into the kv_transfer_params format."""
    return {
        "reuse_plan": {
            "chunks": [
                {
                    "token_ids": list(c.token_ids),
                    "old_pos_start": c.old_pos_start,
                    "salt_hex": c.salt.hex(),
                }
                for c in chunks
            ]
        }
    }


def _store_chunk(storage: InMemoryStorage, chunk: Chunk, seed: int = 0) -> None:
    """Populate storage with every block of ``chunk`` as a hit."""
    hashes = chunk.block_hashes(storage.config.block_size)
    g = torch.Generator().manual_seed(seed)
    for i, h in enumerate(hashes):
        old_pos = chunk.old_pos_start + i * BLOCK_SIZE
        positions = torch.arange(
            old_pos, old_pos + BLOCK_SIZE, dtype=torch.float32
        )
        keys_raw = [
            torch.randn(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, generator=g)
            for _ in range(NUM_LAYERS)
        ]
        keys_rot = [apply_rope_at_positions(k, positions) for k in keys_raw]
        values = [
            torch.randn(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, generator=g)
            for _ in range(NUM_LAYERS)
        ]
        storage.put(
            h, KVBlock(keys=keys_rot, values=values, old_pos_start=old_pos)
        )


# ----------------------- get_num_new_matched_tokens -----------------------


def test_unbound_storage_returns_zero():
    c = _connector(bind=False)
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    req = _request(_plan_dict([chunk]))
    assert c.get_num_new_matched_tokens(req, 0) == (0, False)


def test_none_kv_transfer_params_returns_zero():
    c = _connector()
    req = _request(kv_transfer_params=None)
    assert c.get_num_new_matched_tokens(req, 0) == (0, False)


def test_empty_kv_transfer_params_returns_zero():
    c = _connector()
    req = _request(kv_transfer_params={})
    assert c.get_num_new_matched_tokens(req, 0) == (0, False)


def test_missing_reuse_plan_field_returns_zero():
    c = _connector()
    req = _request(kv_transfer_params={"other_field": "x"})
    assert c.get_num_new_matched_tokens(req, 0) == (0, False)


def test_empty_chunks_returns_zero():
    c = _connector()
    req = _request(_plan_dict([]))
    assert c.get_num_new_matched_tokens(req, 0) == (0, False)


def test_full_hit_returns_total_chunk_tokens():
    """Strategy B: report sum of chunk lengths when every block hits."""
    c = _connector()
    chunk_a = Chunk(token_ids=tuple(range(BLOCK_SIZE * 2)), old_pos_start=0)
    chunk_b = Chunk(
        token_ids=tuple(range(100, 100 + BLOCK_SIZE * 3)),
        old_pos_start=BLOCK_SIZE * 2,
    )
    _store_chunk(c._storage, chunk_a, seed=1)
    _store_chunk(c._storage, chunk_b, seed=2)
    req = _request(_plan_dict([chunk_a, chunk_b]))
    n, async_load = c.get_num_new_matched_tokens(req, 0)
    assert n == len(chunk_a.token_ids) + len(chunk_b.token_ids)
    assert async_load is False


def test_partial_miss_falls_back_to_zero():
    """One chunk hits, one chunk misses — fail closed (Phase 9 will recompute)."""
    c = _connector()
    chunk_hit = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    chunk_miss = Chunk(
        token_ids=tuple(range(200, 200 + BLOCK_SIZE)),
        old_pos_start=BLOCK_SIZE,
    )
    _store_chunk(c._storage, chunk_hit, seed=3)
    # chunk_miss intentionally not stored.
    req = _request(_plan_dict([chunk_hit, chunk_miss]))
    assert c.get_num_new_matched_tokens(req, 0) == (0, False)


def test_intra_chunk_partial_miss_falls_back_to_zero():
    """Some blocks of a chunk hit, others miss → whole plan fails closed."""
    c = _connector()
    chunk = Chunk(
        token_ids=tuple(range(BLOCK_SIZE * 3)), old_pos_start=0
    )
    # Manually store only the FIRST block's hash; leave the other two out.
    hashes = chunk.block_hashes(BLOCK_SIZE)
    g = torch.Generator().manual_seed(4)
    positions = torch.arange(0, BLOCK_SIZE, dtype=torch.float32)
    keys_raw = [
        torch.randn(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, generator=g)
        for _ in range(NUM_LAYERS)
    ]
    keys_rot = [apply_rope_at_positions(k, positions) for k in keys_raw]
    values = [
        torch.randn(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, generator=g)
        for _ in range(NUM_LAYERS)
    ]
    c._storage.put(
        hashes[0], KVBlock(keys=keys_rot, values=values, old_pos_start=0)
    )
    req = _request(_plan_dict([chunk]))
    assert c.get_num_new_matched_tokens(req, 0) == (0, False)


def test_all_miss_returns_zero():
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    req = _request(_plan_dict([chunk]))
    # Store is empty.
    assert c.get_num_new_matched_tokens(req, 0) == (0, False)


def test_misaligned_chunk_returns_zero():
    """Alignment violation surfaces from block_hashes() as 0."""
    c = _connector()
    # length 3 is not a multiple of block_size 4 → AlignmentError.
    chunk = Chunk(token_ids=tuple(range(3)), old_pos_start=0)
    req = _request(_plan_dict([chunk]))
    assert c.get_num_new_matched_tokens(req, 0) == (0, False)


def test_misaligned_start_pos_returns_zero():
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=1)
    req = _request(_plan_dict([chunk]))
    assert c.get_num_new_matched_tokens(req, 0) == (0, False)


@pytest.mark.parametrize(
    "params",
    [
        {"reuse_plan": "not-a-dict"},
        {"reuse_plan": []},
        {"reuse_plan": {}},  # missing "chunks"
        {"reuse_plan": {"chunks": "not-a-list"}},
        {"reuse_plan": {"chunks": [{"old_pos_start": 0}]}},  # missing token_ids
        {"reuse_plan": {"chunks": [{"token_ids": [1]}]}},  # missing old_pos_start
        {"reuse_plan": {"chunks": [{"token_ids": "abc", "old_pos_start": 0}]}},
        {
            "reuse_plan": {
                "chunks": [
                    {
                        "token_ids": [1, 2, 3, 4],
                        "old_pos_start": 0,
                        "salt_hex": "zzz",
                    }
                ]
            }
        },
    ],
)
def test_malformed_plan_returns_zero(params):
    c = _connector()
    req = _request(params)
    assert c.get_num_new_matched_tokens(req, 0) == (0, False)


def test_idempotent_calls():
    """Repeated calls with the same request must return the same value."""
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE * 2)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=5)
    req = _request(_plan_dict([chunk]))
    a = c.get_num_new_matched_tokens(req, 0)
    b = c.get_num_new_matched_tokens(req, 0)
    c2 = c.get_num_new_matched_tokens(req, 0)
    assert a == b == c2 == (len(chunk.token_ids), False)


def test_num_computed_tokens_is_ignored_by_skeleton():
    """Step 1-2 returns plan-defined total; later phases may refine."""
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=6)
    req = _request(_plan_dict([chunk]))
    a, _ = c.get_num_new_matched_tokens(req, 0)
    b, _ = c.get_num_new_matched_tokens(req, 9999)
    assert a == b == len(chunk.token_ids)


# ----------------------- bind_storage -----------------------


def test_bind_storage_block_size_mismatch_raises():
    c = PersonalContextKVConnector(
        _fake_vllm_config(block_size=BLOCK_SIZE),
        KVConnectorRole.SCHEDULER,
        _fake_kv_cache_config(),
    )
    bad = InMemoryStorage(_store_config(block_size=BLOCK_SIZE * 2))
    with pytest.raises(ValueError, match="block_size"):
        c.bind_storage(bad)


def test_bind_storage_succeeds_with_matching_block_size():
    c = PersonalContextKVConnector(
        _fake_vllm_config(),
        KVConnectorRole.SCHEDULER,
        _fake_kv_cache_config(),
    )
    storage = InMemoryStorage(_store_config())
    c.bind_storage(storage)
    assert c._storage is storage
    assert c._lookup is not None


# ----------------------- Factory registration -----------------------


def test_factory_registration_returns_the_connector_class():
    cls = KVConnectorFactory.get_connector_class_by_name(
        "PersonalContextKVConnector"
    )
    assert cls is PersonalContextKVConnector


# ----------------------- Stub methods -----------------------


def test_stubs_are_callable_without_error():
    c = _connector()
    meta = c.build_connector_meta(scheduler_output=None)
    assert isinstance(meta, PersonalContextConnectorMetadata)
    c.start_load_kv(forward_context=None)
    c.wait_for_layer_load("layer.0")
    c.save_kv_layer(
        layer_name="layer.0",
        kv_layer=torch.zeros(1),
        attn_metadata=None,
    )
    c.wait_for_save()


# ----------------------- update_state_after_alloc (Step 3) -----------------------


def test_pending_loads_starts_empty():
    c = _connector()
    assert c._pending_loads == {}


def test_update_state_zero_external_tokens_is_noop():
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=10)
    req = _request(_plan_dict([chunk]), request_id="req-zero")
    c.update_state_after_alloc(req, blocks=None, num_external_tokens=0)
    assert c._pending_loads == {}


def test_update_state_negative_external_tokens_is_noop():
    """Defensive: scheduler should never pass negative, but no-op anyway."""
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=11)
    req = _request(_plan_dict([chunk]), request_id="req-neg")
    c.update_state_after_alloc(req, blocks=None, num_external_tokens=-1)
    assert c._pending_loads == {}


def test_update_state_no_plan_skips():
    """num_external_tokens>0 but request carries no plan → ignore.

    Behaviour-only check: vllm's root logger has ``propagate=False`` so
    pytest's caplog cannot capture the WARNING. The connector still
    emits one to stdout (visible in captured output); a maintainer
    silencing it without fixing the underlying drop would break the
    `_pending_loads == {}` assertion below first.
    """
    c = _connector()
    req = _request(kv_transfer_params=None, request_id="req-noplan")
    c.update_state_after_alloc(req, blocks=None, num_external_tokens=BLOCK_SIZE)
    assert c._pending_loads == {}


def test_update_state_empty_chunks_skips():
    c = _connector()
    req = _request(_plan_dict([]), request_id="req-empty-chunks")
    c.update_state_after_alloc(req, blocks=None, num_external_tokens=BLOCK_SIZE)
    assert c._pending_loads == {}


def test_update_state_mismatched_total_skips():
    """num_external_tokens != plan total → state divergence; ignore."""
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=12)
    req = _request(_plan_dict([chunk]), request_id="req-mismatch")
    c.update_state_after_alloc(
        req, blocks=None, num_external_tokens=BLOCK_SIZE * 2
    )
    assert c._pending_loads == {}


def test_update_state_records_valid_plan():
    c = _connector()
    chunk_a = Chunk(token_ids=tuple(range(BLOCK_SIZE * 2)), old_pos_start=0)
    chunk_b = Chunk(
        token_ids=tuple(range(50, 50 + BLOCK_SIZE)),
        old_pos_start=BLOCK_SIZE * 2,
    )
    _store_chunk(c._storage, chunk_a, seed=13)
    _store_chunk(c._storage, chunk_b, seed=14)
    total = len(chunk_a.token_ids) + len(chunk_b.token_ids)
    req = _request(_plan_dict([chunk_a, chunk_b]), request_id="req-ok")

    c.update_state_after_alloc(req, blocks=None, num_external_tokens=total)

    assert set(c._pending_loads.keys()) == {"req-ok"}
    entry = c._pending_loads["req-ok"]
    assert entry.num_external_tokens == total
    assert len(entry.plan.chunks) == 2
    assert entry.plan.chunks[0].token_ids == chunk_a.token_ids
    assert entry.plan.chunks[1].old_pos_start == chunk_b.old_pos_start


def test_update_state_multi_request_independent():
    c = _connector()
    chunk_a = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    chunk_b = Chunk(
        token_ids=tuple(range(60, 60 + BLOCK_SIZE * 2)), old_pos_start=0
    )
    _store_chunk(c._storage, chunk_a, seed=15)
    _store_chunk(c._storage, chunk_b, seed=16)

    c.update_state_after_alloc(
        _request(_plan_dict([chunk_a]), request_id="rA"),
        blocks=None,
        num_external_tokens=BLOCK_SIZE,
    )
    c.update_state_after_alloc(
        _request(_plan_dict([chunk_b]), request_id="rB"),
        blocks=None,
        num_external_tokens=BLOCK_SIZE * 2,
    )

    assert set(c._pending_loads.keys()) == {"rA", "rB"}
    assert c._pending_loads["rA"].num_external_tokens == BLOCK_SIZE
    assert c._pending_loads["rB"].num_external_tokens == BLOCK_SIZE * 2
    assert c._pending_loads["rA"].plan.chunks[0].token_ids == chunk_a.token_ids
    assert c._pending_loads["rB"].plan.chunks[0].token_ids == chunk_b.token_ids


def test_update_state_repeat_call_overwrites():
    """Idempotent overwrite — last write wins; defensive against double-call."""
    c = _connector()
    chunk_one = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    chunk_two = Chunk(
        token_ids=tuple(range(BLOCK_SIZE * 2)), old_pos_start=0
    )
    _store_chunk(c._storage, chunk_one, seed=17)
    _store_chunk(c._storage, chunk_two, seed=18)

    c.update_state_after_alloc(
        _request(_plan_dict([chunk_one]), request_id="rep"),
        blocks=None,
        num_external_tokens=BLOCK_SIZE,
    )
    c.update_state_after_alloc(
        _request(_plan_dict([chunk_two]), request_id="rep"),
        blocks=None,
        num_external_tokens=BLOCK_SIZE * 2,
    )

    entry = c._pending_loads["rep"]
    assert entry.num_external_tokens == BLOCK_SIZE * 2
    assert entry.plan.chunks[0].token_ids == chunk_two.token_ids


def test_update_state_does_not_validate_storage_hits():
    """Step 3 trusts get_num_new_matched_tokens; doesn't re-lookup the store.

    If the plan parses cleanly and num_external_tokens matches, we record.
    Eviction races between Step 1-2 and Step 3 are not re-checked here —
    that's Phase 11 / 9 territory.
    """
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    # Note: chunk intentionally NOT stored.
    req = _request(_plan_dict([chunk]), request_id="trust")
    c.update_state_after_alloc(req, blocks=None, num_external_tokens=BLOCK_SIZE)
    assert "trust" in c._pending_loads
