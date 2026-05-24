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
    PersonalContextReqMeta,
)
from vllm.v1.personal_context import (
    Chunk,
    InMemoryStorage,
    KVBlock,
    NoSelection,
    ReusePlan,
    SelectFirstR,
    StoreConfig,
    apply_rope_at_positions,
)


BLOCK_SIZE = 4
NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8


# ----------------------- fixtures / helpers -----------------------


def _fake_vllm_config(
    block_size: int = BLOCK_SIZE,
    kv_connector_extra_config: dict | None = None,
    with_model_config: bool = False,
):
    """Minimal fake of ``VllmConfig`` for the connector.

    ``with_model_config=True`` adds a ``model_config`` substruct whose
    ``hf_config`` agrees numerically with the test BLOCK/NUM_LAYERS/
    NUM_KV_HEADS/HEAD_DIM constants. Required when exercising
    ``_derive_store_config`` (Redis backend) — pure scheduler-side
    tests can leave it off.
    """
    cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="PersonalContextKVConnector",
            kv_connector_extra_config=kv_connector_extra_config or {},
        ),
        cache_config=SimpleNamespace(block_size=block_size),
    )
    if with_model_config:
        cfg.model_config = SimpleNamespace(
            model="test/model",
            dtype=torch.float32,
            hf_config=SimpleNamespace(
                num_hidden_layers=NUM_LAYERS,
                num_key_value_heads=NUM_KV_HEADS,
                num_attention_heads=NUM_KV_HEADS,
                hidden_size=NUM_KV_HEADS * HEAD_DIM,
                rope_theta=10000.0,
            ),
        )
    return cfg


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


def _new_req(
    req_id: str,
    block_ids: list[int],
    num_computed_tokens: int,
):
    """Minimal duck-typed NewRequestData stand-in.

    PersonalContextKVConnector only reads ``.req_id``,
    ``.block_ids``, and ``.num_computed_tokens``; a SimpleNamespace
    with those three fields suffices.
    """
    return SimpleNamespace(
        req_id=req_id,
        block_ids=(list(block_ids),),
        num_computed_tokens=num_computed_tokens,
    )


def _scheduler_output(new_reqs):
    return SimpleNamespace(scheduled_new_reqs=list(new_reqs))


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
    meta = c.build_connector_meta(_scheduler_output(new_reqs=[]))
    assert isinstance(meta, PersonalContextConnectorMetadata)
    assert meta.requests == ()
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


# ----------------------- build_connector_meta (Step 4) -----------------------


def _admit(connector: PersonalContextKVConnector, req_id: str, chunks: list[Chunk]):
    """Helper: simulate scheduler admission (Step 1-2 + Step 3 chain)."""
    total = sum(len(c.token_ids) for c in chunks)
    req = _request(_plan_dict(chunks), request_id=req_id)
    connector.update_state_after_alloc(
        req, blocks=None, num_external_tokens=total
    )
    return total


def test_personal_context_req_meta_is_frozen():
    plan = ReusePlan(
        chunks=(Chunk(token_ids=(1, 2, 3, 4), old_pos_start=0),)
    )
    meta = PersonalContextReqMeta(
        request_id="x",
        plan=plan,
        block_assignments=((7,),),
        new_pos_starts=(0,),
    )
    with pytest.raises(AttributeError):
        meta.request_id = "y"  # type: ignore[misc]


def test_build_meta_empty_when_no_pending():
    c = _connector()
    meta = c.build_connector_meta(_scheduler_output(new_reqs=[]))
    assert isinstance(meta, PersonalContextConnectorMetadata)
    assert meta.requests == ()


def test_build_meta_drains_pending_loads():
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=20)
    _admit(c, "drain-1", [chunk])
    assert "drain-1" in c._pending_loads

    c.build_connector_meta(
        _scheduler_output([_new_req("drain-1", [42], num_computed_tokens=BLOCK_SIZE)])
    )
    assert c._pending_loads == {}


def test_build_meta_drops_pending_without_scheduled_match():
    """Pending entry with no matching scheduled_new_reqs row → drain + no meta."""
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=21)
    _admit(c, "lost", [chunk])

    meta = c.build_connector_meta(_scheduler_output(new_reqs=[]))
    assert meta.requests == ()
    assert c._pending_loads == {}


def test_build_meta_ignores_unrelated_scheduled_reqs():
    c = _connector()
    out = _scheduler_output(
        [_new_req("not-ours", [0, 1, 2], num_computed_tokens=BLOCK_SIZE * 3)]
    )
    meta = c.build_connector_meta(out)
    assert meta.requests == ()


def test_build_meta_single_chunk_single_block():
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=22)
    total = _admit(c, "r1", [chunk])

    # 1 block from external + 1 block of remaining prefill = 2 blocks total.
    out = _scheduler_output(
        [_new_req("r1", [101, 102], num_computed_tokens=total)]
    )
    meta = c.build_connector_meta(out)

    assert len(meta.requests) == 1
    rm = meta.requests[0]
    assert rm.request_id == "r1"
    assert rm.plan.chunks[0].token_ids == chunk.token_ids
    assert rm.block_assignments == ((101,),)
    assert rm.new_pos_starts == (0,)


def test_build_meta_single_chunk_multi_block():
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE * 3)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=23)
    total = _admit(c, "r2", [chunk])

    # 3 chunk-blocks + 1 extra for query suffix
    out = _scheduler_output(
        [_new_req("r2", [10, 11, 12, 13], num_computed_tokens=total)]
    )
    meta = c.build_connector_meta(out)

    rm = meta.requests[0]
    assert rm.block_assignments == ((10, 11, 12),)
    assert rm.new_pos_starts == (0,)


def test_build_meta_multi_chunk():
    c = _connector()
    chunk_a = Chunk(
        token_ids=tuple(range(BLOCK_SIZE * 2)), old_pos_start=0
    )
    chunk_b = Chunk(
        token_ids=tuple(range(50, 50 + BLOCK_SIZE)),
        old_pos_start=BLOCK_SIZE * 2,
    )
    _store_chunk(c._storage, chunk_a, seed=24)
    _store_chunk(c._storage, chunk_b, seed=25)
    total = _admit(c, "r3", [chunk_a, chunk_b])

    # chunk_a uses 2 blocks, chunk_b 1 block. Plus 1 block for query.
    out = _scheduler_output(
        [_new_req("r3", [20, 21, 22, 23], num_computed_tokens=total)]
    )
    meta = c.build_connector_meta(out)

    rm = meta.requests[0]
    assert rm.block_assignments == ((20, 21), (22,))
    assert rm.new_pos_starts == (0, BLOCK_SIZE * 2)


def test_build_meta_with_local_prefix():
    """Local prefix cache already covered some leading blocks."""
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE * 2)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=26)
    external_total = _admit(c, "r4", [chunk])

    # Pretend local prefix cache covered 2 blocks (positions 0..2*BS).
    # Our chunk covers the next 2 blocks (positions 2*BS .. 4*BS).
    local_prefix = BLOCK_SIZE * 2
    num_computed = local_prefix + external_total
    out = _scheduler_output(
        [
            _new_req(
                "r4",
                [200, 201, 202, 203, 204],
                num_computed_tokens=num_computed,
            )
        ]
    )
    meta = c.build_connector_meta(out)

    rm = meta.requests[0]
    # External range starts at block index 2 → block ids [202, 203]
    assert rm.block_assignments == ((202, 203),)
    assert rm.new_pos_starts == (local_prefix,)


def test_build_meta_multi_request():
    c = _connector()
    chunk_a = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    chunk_b = Chunk(token_ids=tuple(range(50, 50 + BLOCK_SIZE * 2)), old_pos_start=0)
    _store_chunk(c._storage, chunk_a, seed=27)
    _store_chunk(c._storage, chunk_b, seed=28)
    total_a = _admit(c, "rA", [chunk_a])
    total_b = _admit(c, "rB", [chunk_b])

    out = _scheduler_output(
        [
            _new_req("rA", [1, 2], num_computed_tokens=total_a),
            _new_req("rB", [10, 11, 12], num_computed_tokens=total_b),
        ]
    )
    meta = c.build_connector_meta(out)

    assert len(meta.requests) == 2
    by_id = {r.request_id: r for r in meta.requests}
    assert by_id["rA"].block_assignments == ((1,),)
    assert by_id["rB"].block_assignments == ((10, 11),)


def test_build_meta_misaligned_local_prefix_drops():
    """num_computed - num_external must be block-aligned; otherwise drop."""
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=29)
    _admit(c, "bad-prefix", [chunk])

    # Inject misalignment: num_computed = external + 1 byte → local_prefix=1.
    out = _scheduler_output(
        [_new_req("bad-prefix", [33], num_computed_tokens=BLOCK_SIZE + 1)]
    )
    meta = c.build_connector_meta(out)
    assert meta.requests == ()


def test_build_meta_negative_local_prefix_drops():
    """num_external > num_computed is impossible from scheduler; defensive drop."""
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=30)
    _admit(c, "neg", [chunk])

    out = _scheduler_output(
        [_new_req("neg", [33], num_computed_tokens=0)]
    )
    meta = c.build_connector_meta(out)
    assert meta.requests == ()


def test_build_meta_short_block_ids_drops():
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE * 3)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=31)
    total = _admit(c, "short", [chunk])

    # Need 3 blocks for the chunk but only 2 supplied.
    out = _scheduler_output(
        [_new_req("short", [9, 8], num_computed_tokens=total)]
    )
    meta = c.build_connector_meta(out)
    assert meta.requests == ()


def test_build_meta_wrong_kv_cache_group_count_drops():
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=32)
    total = _admit(c, "multi-group", [chunk])

    # Force two KV cache groups — connector only supports one.
    new_req = SimpleNamespace(
        req_id="multi-group",
        block_ids=([5], [9]),
        num_computed_tokens=total,
    )
    meta = c.build_connector_meta(_scheduler_output([new_req]))
    assert meta.requests == ()


def test_build_meta_is_idempotent_when_empty():
    c = _connector()
    a = c.build_connector_meta(_scheduler_output(new_reqs=[]))
    b = c.build_connector_meta(_scheduler_output(new_reqs=[]))
    assert a.requests == () and b.requests == ()


# ----------------------- selection (Step 6.1 / 6.2) -----------------------


def _drive_build_meta(
    connector: PersonalContextKVConnector,
    chunks: list[Chunk],
) -> PersonalContextReqMeta:
    """Helper: push a pending load and call build_connector_meta, returning
    the single PersonalContextReqMeta produced. Used to exercise selector
    integration without re-deriving block_assignments by hand."""
    plan_dict = _plan_dict(chunks)
    req = _request(plan_dict, request_id="sel-test")
    total = sum(len(c.token_ids) for c in chunks)
    connector.update_state_after_alloc(
        req, blocks=None, num_external_tokens=total
    )
    num_blocks = sum(len(c.token_ids) // BLOCK_SIZE for c in chunks)
    new_req = _new_req(
        req_id="sel-test",
        block_ids=list(range(100, 100 + num_blocks)),
        num_computed_tokens=total,
    )
    out = connector.build_connector_meta(
        _scheduler_output(new_reqs=[new_req])
    )
    assert len(out.requests) == 1
    return out.requests[0]


def test_selected_positions_defaults_to_empty_without_selector():
    """Backwards compat: existing behaviour (strategy B) gives empty selection."""
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE * 2)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=60)
    rm = _drive_build_meta(c, [chunk])
    assert rm.selected_positions == ()


def test_noselection_keeps_selected_positions_empty():
    c = _connector()
    c.bind_selector(NoSelection())
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE * 2)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=61)
    rm = _drive_build_meta(c, [chunk])
    assert rm.selected_positions == ()


def test_select_first_r_populates_selected_positions():
    c = _connector()
    c.bind_selector(SelectFirstR(0.5))
    # L=8, ceil(0.5 * 8) = 4 positions; chunk at new_pos_start=0
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE * 2)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=62)
    rm = _drive_build_meta(c, [chunk])
    assert rm.selected_positions == (0, 1, 2, 3)


def test_select_first_r_multi_chunk_concatenates():
    c = _connector()
    c.bind_selector(SelectFirstR(0.5))
    chunk_a = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    chunk_b = Chunk(
        token_ids=tuple(range(50, 50 + BLOCK_SIZE * 2)),
        old_pos_start=BLOCK_SIZE,
    )
    _store_chunk(c._storage, chunk_a, seed=63)
    _store_chunk(c._storage, chunk_b, seed=64)
    rm = _drive_build_meta(c, [chunk_a, chunk_b])
    # Chunk A: L=4, ceil(2)=2 positions [0, 1]
    # Chunk B: L=8, ceil(4)=4 positions starting at new_pos_start=4 → [4, 5, 6, 7]
    assert rm.selected_positions == (0, 1, 4, 5, 6, 7)


def test_bind_selector_none_reverts_to_no_selection():
    c = _connector()
    c.bind_selector(SelectFirstR(0.5))
    c.bind_selector(None)
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE * 2)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=65)
    rm = _drive_build_meta(c, [chunk])
    assert rm.selected_positions == ()


def test_selector_output_is_sorted_and_deduped():
    """The connector defensively sorts + dedupes the selector's output."""

    class _UnsortedDupSelector:
        def select(self, plan, new_pos_starts):
            return (5, 1, 5, 3, 1, 2)

    c = _connector()
    c.bind_selector(_UnsortedDupSelector())
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE * 2)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=66)
    rm = _drive_build_meta(c, [chunk])
    assert rm.selected_positions == (1, 2, 3, 5)


def test_misbehaving_selector_falls_back_to_empty():
    """A selector raising should not break build_connector_meta."""

    class _BrokenSelector:
        def select(self, plan, new_pos_starts):
            raise RuntimeError("boom")

    c = _connector()
    c.bind_selector(_BrokenSelector())
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE * 2)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=67)
    rm = _drive_build_meta(c, [chunk])
    # Connector falls back to strategy B and ships the meta with empty
    # selection rather than dropping the load entirely.
    assert rm.selected_positions == ()


def test_selector_receives_plan_and_new_pos_starts():
    """Spy selector confirms connector passes plan + new_pos_starts correctly."""
    captured: dict = {}

    class _SpySelector:
        def select(self, plan, new_pos_starts):
            captured["plan_chunks"] = plan.chunks
            captured["new_pos_starts"] = tuple(new_pos_starts)
            return ()

    c = _connector()
    c.bind_selector(_SpySelector())
    chunk_a = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    chunk_b = Chunk(
        token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=BLOCK_SIZE,
    )
    _store_chunk(c._storage, chunk_a, seed=68)
    _store_chunk(c._storage, chunk_b, seed=69)
    _drive_build_meta(c, [chunk_a, chunk_b])

    assert captured["plan_chunks"] == (chunk_a, chunk_b)
    # new_pos_starts are derived from local_prefix (=0 here) + cumulative
    # chunk lengths.
    assert captured["new_pos_starts"] == (0, BLOCK_SIZE)


# ----------------------- start_load_kv (Step 5) -----------------------


def _make_kv_caches(num_blocks: int = 8):
    """Per-layer NHD paged caches, zeroed."""
    shape = (num_blocks, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    return [torch.zeros(shape, dtype=torch.float32) for _ in range(NUM_LAYERS)]


def _forward_context_with_kvs(kv_caches):
    """SimpleNamespace shaped like vLLM's ForwardContext.

    PersonalContextKVConnector reads ``forward_context.no_compile_layers``
    as a name→layer dict; each layer is duck-typed to expose ``.kv_cache``.
    """
    layers = {
        f"layer.{i}": SimpleNamespace(kv_cache=kv)
        for i, kv in enumerate(kv_caches)
    }
    return SimpleNamespace(no_compile_layers=layers)


def _bind_meta(
    connector: PersonalContextKVConnector,
    req_metas: list[PersonalContextReqMeta],
):
    meta = PersonalContextConnectorMetadata(requests=tuple(req_metas))
    connector.bind_connector_metadata(meta)


def _req_meta(
    request_id: str,
    chunks: list[Chunk],
    block_assignments: tuple[tuple[int, ...], ...],
    new_pos_starts: tuple[int, ...] | None = None,
) -> PersonalContextReqMeta:
    if new_pos_starts is None:
        new_pos_starts = tuple(c.old_pos_start for c in chunks)
    return PersonalContextReqMeta(
        request_id=request_id,
        plan=ReusePlan(chunks=tuple(chunks)),
        block_assignments=block_assignments,
        new_pos_starts=new_pos_starts,
    )


# ----- early-exit branches -----


def test_start_load_kv_noop_without_metadata():
    c = _connector()
    kv_caches = _make_kv_caches()
    fwd = _forward_context_with_kvs(kv_caches)
    # No bind_connector_metadata called → has_connector_metadata() == False
    c.start_load_kv(fwd)
    for kv in kv_caches:
        assert torch.all(kv == 0)


def test_start_load_kv_noop_with_wrong_metadata_type():
    c = _connector()
    kv_caches = _make_kv_caches()
    fwd = _forward_context_with_kvs(kv_caches)

    # A sibling KVConnectorMetadata subclass (or stand-in) → wrong type.
    class _OtherMeta(PersonalContextConnectorMetadata.__bases__[0]):
        pass

    c.bind_connector_metadata(_OtherMeta())
    c.start_load_kv(fwd)
    for kv in kv_caches:
        assert torch.all(kv == 0)


def test_start_load_kv_empty_requests_is_noop():
    c = _connector()
    kv_caches = _make_kv_caches()
    fwd = _forward_context_with_kvs(kv_caches)
    _bind_meta(c, [])
    c.start_load_kv(fwd)
    for kv in kv_caches:
        assert torch.all(kv == 0)


def test_start_load_kv_unbound_storage_skips():
    c = _connector(bind=False)  # No storage bound on worker side.
    kv_caches = _make_kv_caches()
    fwd = _forward_context_with_kvs(kv_caches)
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _bind_meta(c, [_req_meta("x", [chunk], ((1,),))])
    c.start_load_kv(fwd)
    for kv in kv_caches:
        assert torch.all(kv == 0)


def test_start_load_kv_missing_layers_skips():
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=40)
    fwd = SimpleNamespace(no_compile_layers={})
    _bind_meta(c, [_req_meta("x", [chunk], ((1,),))])
    # No assertion needed beyond "doesn't crash".
    c.start_load_kv(fwd)


# ----- happy path -----


def test_start_load_kv_single_chunk_writes_paged_cache():
    """delta=0 case: stored K/V == cache slot after scatter."""
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk, seed=41)
    kv_caches = _make_kv_caches()
    fwd = _forward_context_with_kvs(kv_caches)

    target_block = 3
    _bind_meta(
        c,
        [
            _req_meta(
                "r1",
                [chunk],
                block_assignments=((target_block,),),
                new_pos_starts=(0,),
            )
        ],
    )
    c.start_load_kv(fwd)

    # Stored block should now sit at kv_caches[layer][target_block].
    stored = c._storage.get(chunk.block_hashes(BLOCK_SIZE)[0])
    assert stored is not None
    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(
            kv_caches[layer][target_block, 0], stored.keys[layer]
        )
        torch.testing.assert_close(
            kv_caches[layer][target_block, 1], stored.values[layer]
        )


def test_start_load_kv_multi_chunk_per_request():
    c = _connector()
    chunk_a = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    chunk_b = Chunk(
        token_ids=tuple(range(50, 50 + BLOCK_SIZE)),
        old_pos_start=BLOCK_SIZE,
    )
    _store_chunk(c._storage, chunk_a, seed=42)
    _store_chunk(c._storage, chunk_b, seed=43)
    kv_caches = _make_kv_caches()
    fwd = _forward_context_with_kvs(kv_caches)

    _bind_meta(
        c,
        [
            _req_meta(
                "r2",
                [chunk_a, chunk_b],
                block_assignments=((2,), (5,)),
                new_pos_starts=(0, BLOCK_SIZE),
            )
        ],
    )
    c.start_load_kv(fwd)

    stored_a = c._storage.get(chunk_a.block_hashes(BLOCK_SIZE)[0])
    stored_b = c._storage.get(chunk_b.block_hashes(BLOCK_SIZE)[0])
    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(kv_caches[layer][2, 0], stored_a.keys[layer])
        torch.testing.assert_close(kv_caches[layer][5, 0], stored_b.keys[layer])
        # Untouched blocks stay zeroed.
        for unused_block in (0, 1, 3, 4, 6, 7):
            assert torch.all(kv_caches[layer][unused_block] == 0)


def test_start_load_kv_multi_request():
    c = _connector()
    chunk_a = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    chunk_b = Chunk(token_ids=tuple(range(60, 60 + BLOCK_SIZE)), old_pos_start=0)
    _store_chunk(c._storage, chunk_a, seed=44)
    _store_chunk(c._storage, chunk_b, seed=45)
    kv_caches = _make_kv_caches()
    fwd = _forward_context_with_kvs(kv_caches)

    _bind_meta(
        c,
        [
            _req_meta("rA", [chunk_a], ((1,),)),
            _req_meta("rB", [chunk_b], ((4,),)),
        ],
    )
    c.start_load_kv(fwd)

    stored_a = c._storage.get(chunk_a.block_hashes(BLOCK_SIZE)[0])
    stored_b = c._storage.get(chunk_b.block_hashes(BLOCK_SIZE)[0])
    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(kv_caches[layer][1, 0], stored_a.keys[layer])
        torch.testing.assert_close(kv_caches[layer][4, 0], stored_b.keys[layer])


# ----- per-request failure isolation -----


def test_start_load_kv_alignment_error_skips_only_bad_request():
    c = _connector()
    good = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    bad = Chunk(token_ids=tuple(range(3)), old_pos_start=0)  # length=3, not aligned
    _store_chunk(c._storage, good, seed=46)
    kv_caches = _make_kv_caches()
    fwd = _forward_context_with_kvs(kv_caches)

    _bind_meta(
        c,
        [
            _req_meta("good", [good], ((2,),)),
            _req_meta("bad", [bad], ((3,),)),
        ],
    )
    c.start_load_kv(fwd)

    stored_good = c._storage.get(good.block_hashes(BLOCK_SIZE)[0])
    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(kv_caches[layer][2, 0], stored_good.keys[layer])
        # block 3 never gets touched (bad request skipped).
        assert torch.all(kv_caches[layer][3] == 0)


def test_start_load_kv_store_miss_skips_request():
    c = _connector()
    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    # Intentionally do NOT store the chunk on worker side.
    kv_caches = _make_kv_caches()
    fwd = _forward_context_with_kvs(kv_caches)
    _bind_meta(c, [_req_meta("miss", [chunk], ((1,),))])
    c.start_load_kv(fwd)
    for kv in kv_caches:
        assert torch.all(kv == 0)


# ----- rope_theta plumbing -----


def test_rope_theta_default_is_10000():
    c = _connector()
    assert c._rope_theta == 10000.0


def test_rope_theta_picks_up_model_config_when_present():
    cfg = _fake_vllm_config()
    cfg.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(rope_theta=500000.0)
    )
    c = PersonalContextKVConnector(cfg, KVConnectorRole.WORKER, _fake_kv_cache_config())
    assert c._rope_theta == 500000.0


def test_rope_theta_threads_through_load_plan():
    """Loaded K under custom rope_theta differs for non-zero delta."""
    cfg = _fake_vllm_config()
    cfg.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(rope_theta=500000.0)
    )
    c_custom = PersonalContextKVConnector(
        cfg, KVConnectorRole.WORKER, _fake_kv_cache_config()
    )
    c_custom.bind_storage(InMemoryStorage(_store_config()))
    c_default = _connector()

    chunk = Chunk(token_ids=tuple(range(BLOCK_SIZE)), old_pos_start=0)
    # Same K/V on both sides (deterministic seed).
    _store_chunk(c_custom._storage, chunk, seed=50)
    _store_chunk(c_default._storage, chunk, seed=50)

    kvs_custom = _make_kv_caches()
    kvs_default = _make_kv_caches()

    # new_pos_start = BLOCK_SIZE → non-zero delta on K.
    _bind_meta(
        c_custom,
        [_req_meta("r", [chunk], ((1,),), new_pos_starts=(BLOCK_SIZE,))],
    )
    _bind_meta(
        c_default,
        [_req_meta("r", [chunk], ((1,),), new_pos_starts=(BLOCK_SIZE,))],
    )
    c_custom.start_load_kv(_forward_context_with_kvs(kvs_custom))
    c_default.start_load_kv(_forward_context_with_kvs(kvs_default))

    # K differs (different rope_theta);
    # V identical (position-independent).
    assert not torch.allclose(kvs_custom[0][1, 0], kvs_default[0][1, 0])
    torch.testing.assert_close(kvs_custom[0][1, 1], kvs_default[0][1, 1])


# ----------------------- selector resolved from config -----------------------


def _connector_with_extra(extra: dict) -> PersonalContextKVConnector:
    return PersonalContextKVConnector(
        _fake_vllm_config(kv_connector_extra_config=extra),
        KVConnectorRole.SCHEDULER,
        _fake_kv_cache_config(),
    )


def test_resolve_selector_default_none():
    c = _connector_with_extra({})
    assert c._selector is None


def test_resolve_selector_explicit_none_string():
    c = _connector_with_extra({"selector": "NoSelection"})
    assert c._selector is None


def test_resolve_selector_first_r_shorthand_defaults_to_one():
    c = _connector_with_extra({"selector": "SelectFirstR"})
    assert isinstance(c._selector, SelectFirstR)
    assert c._selector.r == 1.0


def test_resolve_selector_first_r_with_dict_params():
    c = _connector_with_extra({"selector": {"type": "SelectFirstR", "r": 0.5}})
    assert isinstance(c._selector, SelectFirstR)
    assert c._selector.r == 0.5


def test_resolve_selector_unknown_type_falls_back_to_none():
    c = _connector_with_extra({"selector": "DefinitelyNotAType"})
    assert c._selector is None


def test_resolve_selector_malformed_r_falls_back_to_none():
    c = _connector_with_extra(
        {"selector": {"type": "SelectFirstR", "r": "not-a-number"}}
    )
    assert c._selector is None


def test_resolve_selector_r_out_of_range_falls_back_to_none():
    c = _connector_with_extra({"selector": {"type": "SelectFirstR", "r": 2.0}})
    assert c._selector is None


def test_resolve_selector_non_string_non_dict_falls_back_to_none():
    c = _connector_with_extra({"selector": 42})
    assert c._selector is None


def test_test_bind_overrides_config_selector():
    """``VLLM_PERSONAL_CONTEXT_TEST_BIND`` takes priority over config.

    Allows GPU e2e tests to inject arbitrary selectors without
    touching the engine-level ``kv_connector_extra_config``.
    """
    import os
    import pickle
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".pkl", prefix="pc_test_")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            pickle.dump({"selector": NoSelection()}, f)
        os.environ["VLLM_PERSONAL_CONTEXT_TEST_BIND"] = path
        try:
            c = _connector_with_extra(
                {"selector": {"type": "SelectFirstR", "r": 0.25}}
            )
            # Config would have given SelectFirstR(r=0.25); test bind
            # overwrites with NoSelection (None per bind_selector
            # semantics for the NoSelection instance is preserved as-is).
            assert isinstance(c._selector, NoSelection)
        finally:
            os.environ.pop("VLLM_PERSONAL_CONTEXT_TEST_BIND", None)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ----------------------- storage backend resolved from config -----------


def _connector_with_storage_config(
    extra: dict, **fake_kwargs
) -> PersonalContextKVConnector:
    return PersonalContextKVConnector(
        _fake_vllm_config(
            kv_connector_extra_config=extra,
            with_model_config=True,
            **fake_kwargs,
        ),
        KVConnectorRole.SCHEDULER,
        _fake_kv_cache_config(),
    )


def test_storage_default_unbound():
    c = _connector_with_storage_config({})
    assert c._storage is None
    assert c._lookup is None


def test_storage_backend_memory_is_noop():
    """``store_backend='memory'`` is explicit-but-still-unbound — the
    caller is expected to ``bind_storage`` directly afterwards."""
    c = _connector_with_storage_config({"store_backend": "memory"})
    assert c._storage is None


def test_storage_backend_unknown_logs_and_unbound():
    c = _connector_with_storage_config({"store_backend": "totally-made-up"})
    assert c._storage is None


def test_storage_backend_redis_without_url_unbound():
    c = _connector_with_storage_config({"store_backend": "redis"})
    assert c._storage is None


def test_storage_backend_redis_with_url_binds(monkeypatch):
    """Config-driven Redis binding: connector init builds a
    ``RedisKVStorage`` against the configured URL and binds it. We
    monkeypatch ``redis.Redis.from_url`` so no real server is needed.
    """
    fakeredis = pytest.importorskip("fakeredis")
    import redis as _redis

    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(
        _redis.Redis, "from_url", classmethod(lambda cls, url, **kw: fake)
    )

    c = _connector_with_storage_config(
        {
            "store_backend": "redis",
            "store_url": "redis://does-not-matter:6379",
        }
    )
    from vllm.v1.personal_context.redis_storage import RedisKVStorage

    assert isinstance(c._storage, RedisKVStorage)
    assert c._lookup is not None
    # Schema check wrote ``pc:config`` once.
    assert fake.get(b"pc:config") is not None


def test_storage_backend_redis_schema_mismatch_unbound(monkeypatch):
    """Pre-seed fakeredis with a config that disagrees with the
    connector's derived StoreConfig → RedisKVStorage init raises →
    connector stays unbound rather than crashing engine startup."""
    fakeredis = pytest.importorskip("fakeredis")
    import pickle

    import redis as _redis

    fake = fakeredis.FakeRedis()
    # Pre-seed with a config that disagrees (different num_layers).
    bad_cfg = StoreConfig(
        model_id="test/model",
        dtype=torch.float32,
        layout="NHD",
        num_layers=NUM_LAYERS + 5,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
    )
    fake.set(b"pc:config", pickle.dumps(bad_cfg))

    monkeypatch.setattr(
        _redis.Redis, "from_url", classmethod(lambda cls, url, **kw: fake)
    )

    c = _connector_with_storage_config(
        {
            "store_backend": "redis",
            "store_url": "redis://does-not-matter:6379",
        }
    )
    # RedisKVStorage init raised; connector swallowed and stayed unbound.
    assert c._storage is None


def test_env_var_pickle_overrides_config_redis(monkeypatch):
    """``VLLM_PERSONAL_CONTEXT_TEST_BIND`` wins over a Redis-configured
    backend so GPU e2e tests can inject populated InMemoryStorage
    fixtures without rewriting the engine config.
    """
    fakeredis = pytest.importorskip("fakeredis")
    import pickle
    import tempfile

    import redis as _redis

    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(
        _redis.Redis, "from_url", classmethod(lambda cls, url, **kw: fake)
    )

    in_mem_store = InMemoryStorage(_store_config())

    fd, path = tempfile.mkstemp(suffix=".pkl", prefix="pc_test_")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            pickle.dump({"storage": in_mem_store}, f)
        monkeypatch.setenv("VLLM_PERSONAL_CONTEXT_TEST_BIND", path)

        c = _connector_with_storage_config(
            {
                "store_backend": "redis",
                "store_url": "redis://does-not-matter:6379",
            }
        )
        # Env-var bind ran AFTER config init, so InMemoryStorage wins.
        assert isinstance(c._storage, InMemoryStorage)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
