# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test for personal-context KV reuse (Phase 7 MVP).

Exercises Phase 7 Steps 1-5 against a REAL vLLM Scheduler running on
CPU, without invoking any model forward:

    Step 1-2  Scheduler.get_num_new_matched_tokens
    Step 3    update_state_after_alloc
    Step 4    build_connector_meta
    Step 5    KVConnectorModelRunnerMixin → start_load_kv → scatter

What this proves
----------------
- The connector wiring (factory registration, scheduler-side instance,
  worker-side instance) actually picks up our class.
- ``get_num_new_matched_tokens`` returning ``total_chunk_tokens`` makes
  the scheduler shrink ``num_scheduled_tokens`` to just the query
  suffix and inflate ``request.num_computed_tokens`` to
  ``local + external``.
- Scheduler-allocated block ids end up in
  ``PersonalContextReqMeta.block_assignments`` in chunk position order.
- The worker-side mixin (``_get_kv_connector_output``) drives
  ``bind_connector_metadata`` + ``start_load_kv`` correctly, and our
  ``start_load_kv`` scatters K/V into the paged cache slots advertised
  by ``block_assignments``.

What this does NOT prove
------------------------
- Attention math correctness (no model forward).
- Logits self-consistency vs a non-personal-context baseline (would
  need a GPU run).
- FlashInfer chunk-aware ``custom_mask`` path (Phase 7 Step 8).
- Selective recompute (Phase 8-9).

Dependencies
------------
- ``facebook/opt-125m`` HF config must be reachable (cached locally
  in ``~/.cache/huggingface/hub``; downloaded on first run).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.personal_context_connector import (
    PersonalContextConnectorMetadata,
    PersonalContextKVConnector,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
)
from vllm.forward_context import override_forward_context
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.personal_context import (
    Chunk,
    InMemoryStorage,
    KVBlock,
    StoreConfig,
    apply_rope_at_positions,
)
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
)

from tests.v1.kv_connector.unit.utils import (
    create_request,
    create_scheduler,
    create_vllm_config,
)


HF_TEST_MODEL = "facebook/opt-125m"
BLOCK_SIZE = 4
NUM_LAYERS = 1
NUM_KV_HEADS = 1
HEAD_DIM = 2  # must be even for apply_rope_at_positions
NUM_BLOCKS = 200


# --------------------------- helpers ---------------------------


def _make_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer.0"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=NUM_KV_HEADS,
                    head_size=HEAD_DIM,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def _store_config() -> StoreConfig:
    return StoreConfig(
        model_id=HF_TEST_MODEL,
        dtype=torch.float32,
        layout="NHD",
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
    )


def _populate_chunk(
    storage: InMemoryStorage, chunk: Chunk, seed: int = 0
) -> None:
    """Pre-RoPE store of a chunk's per-block K/V."""
    cfg = storage.config
    g = torch.Generator().manual_seed(seed)
    shape = (cfg.block_size, cfg.num_kv_heads, cfg.head_dim)
    hashes = chunk.block_hashes(cfg.block_size)
    for i, h in enumerate(hashes):
        old_pos = chunk.old_pos_start + i * cfg.block_size
        positions = torch.arange(
            old_pos, old_pos + cfg.block_size, dtype=torch.float32
        )
        keys_raw = [
            torch.randn(shape, generator=g, dtype=cfg.dtype)
            for _ in range(cfg.num_layers)
        ]
        keys_rot = [apply_rope_at_positions(k, positions) for k in keys_raw]
        values = [
            torch.randn(shape, generator=g, dtype=cfg.dtype)
            for _ in range(cfg.num_layers)
        ]
        storage.put(
            h, KVBlock(keys=keys_rot, values=values, old_pos_start=old_pos)
        )


def _plan_payload(chunks: list[Chunk]) -> dict:
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


def _attach_plan(request, chunks: list[Chunk], query_tokens: list[int]) -> None:
    """Rewrite the request to look like ``[chunk_tokens..., query_tokens...]``
    and attach the matching reuse plan.

    The retriever-stub flow is deferred to Phase 12; this helper plays
    that role for the e2e test (see project memo, the kv_transfer_params
    fallback path).
    """
    prompt = []
    for c in chunks:
        prompt.extend(c.token_ids)
    prompt.extend(query_tokens)
    request.prompt_token_ids = prompt
    # ``_all_token_ids`` is the mutable backing store that
    # ``all_token_ids`` (a ConstantList view) wraps. Mutate in place
    # so the view stays consistent.
    request._all_token_ids.clear()
    request._all_token_ids.extend(prompt)
    request.num_prompt_tokens = len(prompt)
    request.kv_transfer_params = _plan_payload(chunks)


def _forward_context_with_layer(kv_cache_layer: torch.Tensor):
    return SimpleNamespace(
        no_compile_layers={
            "layer.0": SimpleNamespace(kv_cache=kv_cache_layer),
        }
    )


# --------------------------- fixture ---------------------------


@pytest.fixture
def e2e_env():
    """Real Scheduler + worker-side connector, both bound to one storage."""
    kv_cache_config = _make_kv_cache_config()

    vllm_config = create_vllm_config(
        kv_connector="PersonalContextKVConnector",
        kv_role="kv_both",
        block_size=BLOCK_SIZE,
        model=HF_TEST_MODEL,
        disable_hybrid_kv_cache_manager=True,
    )
    # Personal-context plans MUST NOT pollute the prefix cache (see
    # policy.py). The assertion guard isn't wired into the scheduler
    # yet, so disable prefix caching outright to keep the test isolated.
    vllm_config.cache_config.enable_prefix_caching = False

    # ensure_kv_transfer_initialized broadcasts engine_id across TP via
    # parallel_state.get_tp_group(); fake it for a single-process test.
    mock_tp_group = MagicMock()
    mock_tp_group.broadcast_object.side_effect = lambda value, src=0: value

    with patch(
        "vllm.distributed.parallel_state.get_tp_group",
        return_value=mock_tp_group,
    ):
        ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
        scheduler = create_scheduler(
            vllm_config, kv_cache_config=kv_cache_config
        )

    storage = InMemoryStorage(_store_config())
    # Scheduler-side and worker-side are independent connector
    # instances; in single-process tests we bind the SAME storage to
    # both. Production with separate scheduler/worker processes keeps
    # them in sync via the storage backend.
    assert isinstance(scheduler.connector, PersonalContextKVConnector)
    scheduler.connector.bind_storage(storage)
    worker = get_kv_transfer_group()
    assert isinstance(worker, PersonalContextKVConnector)
    worker.bind_storage(storage)

    try:
        yield SimpleNamespace(
            scheduler=scheduler,
            storage=storage,
            scheduler_connector=scheduler.connector,
            worker_connector=worker,
        )
    finally:
        ensure_kv_transfer_shutdown()


# --------------------------- tests ---------------------------


def test_e2e_scheduler_admits_personal_context_request(e2e_env):
    """Step 1-4 end-to-end: scheduler accepts the plan and emits meta."""
    env = e2e_env

    chunk = Chunk(token_ids=(10, 11, 12, 13), old_pos_start=0)
    _populate_chunk(env.storage, chunk, seed=1)

    request = create_request(num_tokens=BLOCK_SIZE * 2, block_size=BLOCK_SIZE)
    _attach_plan(request, [chunk], query_tokens=[100, 101, 102, 103])

    env.scheduler.add_request(request)
    output = env.scheduler.schedule()

    # Scheduler should have shrunk new tokens to just the query suffix
    # (4 tokens). ``num_computed_tokens`` at this point reflects
    # local(0) + external(4) + just-scheduled(4) = 8, i.e. the full
    # prompt — the chunk part is treated as already cached, the query
    # part is being prefilled this step.
    assert output.total_num_scheduled_tokens == BLOCK_SIZE
    assert request.num_computed_tokens == BLOCK_SIZE * 2  # full prompt

    # Connector meta carries the load directive.
    meta = output.kv_connector_metadata
    assert isinstance(meta, PersonalContextConnectorMetadata)
    assert len(meta.requests) == 1
    rm = meta.requests[0]
    assert rm.request_id == request.request_id
    assert rm.new_pos_starts == (0,)
    # One chunk of 4 tokens with block_size 4 → 1 block assignment.
    assert len(rm.block_assignments) == 1
    assert len(rm.block_assignments[0]) == 1
    # The block id must be one the scheduler actually allocated for this
    # request, in position order. Position 0 → block index 0.
    scheduled_block_ids = next(
        nr.block_ids[0]
        for nr in output.scheduled_new_reqs
        if nr.req_id == request.request_id
    )
    assert rm.block_assignments[0][0] == scheduled_block_ids[0]


def test_e2e_worker_mixin_scatters_kv(e2e_env):
    """Step 5 end-to-end: mixin drives bind + start_load_kv → scatter."""
    env = e2e_env

    chunk = Chunk(token_ids=(20, 21, 22, 23), old_pos_start=0)
    _populate_chunk(env.storage, chunk, seed=2)

    request = create_request(num_tokens=BLOCK_SIZE * 2, block_size=BLOCK_SIZE)
    _attach_plan(request, [chunk], query_tokens=[200, 201, 202, 203])

    env.scheduler.add_request(request)
    output = env.scheduler.schedule()

    # Build a CPU NHD paged-cache tensor for the single layer; large
    # enough to cover any block id the scheduler could pick.
    kv_cache_layer = torch.zeros(
        NUM_BLOCKS, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32
    )
    fwd = _forward_context_with_layer(kv_cache_layer)

    # Drive the worker-side mixin lifecycle exactly as the real model
    # runner would: bind_connector_metadata → start_load_kv → forward
    # → wait_for_save → clear_connector_metadata. ``mixin`` calls
    # ``get_forward_context()`` to fetch the active context; the
    # canonical way to set it from a test is via
    # ``override_forward_context`` (a contextmanager that toggles the
    # module-global).
    with override_forward_context(fwd):
        ctx = KVConnectorModelRunnerMixin._get_kv_connector_output(output)
        with ctx:
            # No model forward; just exercise the wrappers.
            pass

    rm = output.kv_connector_metadata.requests[0]
    target_block = rm.block_assignments[0][0]
    stored = env.storage.get(chunk.block_hashes(BLOCK_SIZE)[0])
    assert stored is not None
    torch.testing.assert_close(kv_cache_layer[target_block, 0], stored.keys[0])
    torch.testing.assert_close(kv_cache_layer[target_block, 1], stored.values[0])
    # Worker side cleared metadata after the context closed.
    assert env.worker_connector._connector_metadata is None


def test_e2e_request_without_plan_is_unaffected(e2e_env):
    """No reuse_plan → connector reports 0; scheduler runs vanilla path."""
    env = e2e_env

    request = create_request(num_tokens=BLOCK_SIZE * 2, block_size=BLOCK_SIZE)
    # No _attach_plan → no kv_transfer_params

    env.scheduler.add_request(request)
    output = env.scheduler.schedule()

    # Vanilla prefill: every prompt token is scheduled this step
    # (no external coverage claimed by the connector).
    assert output.total_num_scheduled_tokens == BLOCK_SIZE * 2
    meta = output.kv_connector_metadata
    assert isinstance(meta, PersonalContextConnectorMetadata)
    assert all(
        rm.request_id != request.request_id for rm in meta.requests
    )
