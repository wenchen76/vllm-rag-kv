# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU end-to-end tests for PersonalContext sparse-Q correctness.

Two GPU-only tests against ``facebook/opt-125m`` running a real forward
pass with the PC connector enabled:

    1. ``test_pc_no_selection_diverges_from_vanilla``
       Pre-populate the store with **random** K/V for the chunk, set
       ``NoSelection``. Step 5 scatters that garbage into the paged
       cache; nothing gets recomputed; attention reads garbage. Output
       should DIFFER from the vanilla run on the same prompt.

       (Weak signal: under KV-layout mismatch our scatter might write to
       wrong slots and attention would read zeros — output still
       differs, but for the wrong reason. So this test only proves the
       connector is *plumbed in*, not that the scatter layout is right.)

    2. ``test_pc_full_selection_matches_vanilla``
       Same random K/V in the store, but bind ``SelectFirstR(1.0)``.
       The sparse-Q override widens the Q batch to include every chunk
       position; the forward pass writes fresh K/V via the standard
       ``slot_mapping`` path (which always uses the correct backend
       layout). The garbage we scattered gets overwritten before any
       attention layer reads it, so output should MATCH the vanilla
       run byte-for-byte.

       (Strong signal: layout-independent, exercises ``_prepare_inputs``
       sparse-Q override end-to-end. Any divergence from vanilla
       indicates a real bug in the Step 7 implementation.)

Both tests share the storage between the test process and vLLM's
spawned worker process via a pickle file + ``VLLM_PERSONAL_CONTEXT_TEST_BIND``
env var. The connector reads this on ``__init__`` (both scheduler-side
and worker-side instances).

The tests skip when no CUDA is present, so this file is safe to
collect on a Mac/CPU box.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from contextlib import contextmanager

import pytest
import torch


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU + real model forward required",
)

# ----------------------- constants -----------------------

MODEL = "facebook/opt-125m"
BLOCK_SIZE = 16
# OPT-125m architecture (must match what vLLM allocates):
NUM_LAYERS = 12
NUM_KV_HEADS = 12
HEAD_DIM = 64
# Single chunk of 2 blocks at the head of the prompt, followed by a
# 1-block "query" suffix. Token IDs are arbitrary integers inside the
# OPT vocab (~50272 entries).
CHUNK_LEN = BLOCK_SIZE * 2
QUERY_LEN = BLOCK_SIZE
CHUNK_TOKEN_IDS = tuple(range(100, 100 + CHUNK_LEN))
QUERY_TOKEN_IDS = tuple(range(200, 200 + QUERY_LEN))
PROMPT_TOKEN_IDS = list(CHUNK_TOKEN_IDS) + list(QUERY_TOKEN_IDS)
RNG_SEED = 1234


# ----------------------- helpers -----------------------


def _make_storage_with_random_chunk_kv():
    """Build an InMemoryStorage populated with **random** K/V for the
    test chunk.

    K is rotated by RoPE at the chunk's offline positions (per the
    store contract); V is left as raw random. The values themselves
    don't matter for these tests — they're chosen to be obviously not
    what the model would compute, so test 1 can detect "attention
    actually consumed our scatter" and test 2 can detect "sparse-Q
    overwrote the garbage".
    """
    from vllm.v1.personal_context import (
        Chunk,
        InMemoryStorage,
        KVBlock,
        StoreConfig,
        apply_rope_at_positions,
    )

    cfg = StoreConfig(
        model_id=MODEL,
        dtype=torch.float16,
        layout="NHD",
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
    )
    storage = InMemoryStorage(cfg)
    chunk = Chunk(token_ids=CHUNK_TOKEN_IDS, old_pos_start=0)

    g = torch.Generator().manual_seed(RNG_SEED)
    shape = (BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    for i, key in enumerate(chunk.block_hashes(BLOCK_SIZE)):
        old_pos = chunk.old_pos_start + i * BLOCK_SIZE
        positions = torch.arange(
            old_pos, old_pos + BLOCK_SIZE, dtype=torch.float32
        )
        raw_keys = [
            torch.randn(shape, generator=g, dtype=torch.float16)
            for _ in range(NUM_LAYERS)
        ]
        rot_keys = [apply_rope_at_positions(k, positions) for k in raw_keys]
        values = [
            torch.randn(shape, generator=g, dtype=torch.float16)
            for _ in range(NUM_LAYERS)
        ]
        storage.put(
            key, KVBlock(keys=rot_keys, values=values, old_pos_start=old_pos)
        )
    return storage, chunk


@contextmanager
def _pc_test_bind(storage, selector):
    """Pickle ``storage`` + ``selector`` to a temp path and set
    ``VLLM_PERSONAL_CONTEXT_TEST_BIND`` so the connector instances —
    both scheduler-side in this process and worker-side in vLLM's
    spawned worker — auto-bind on ``__init__``."""
    fd, path = tempfile.mkstemp(suffix=".pkl", prefix="pc_test_")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            pickle.dump({"storage": storage, "selector": selector}, f)
        prev = os.environ.get("VLLM_PERSONAL_CONTEXT_TEST_BIND")
        os.environ["VLLM_PERSONAL_CONTEXT_TEST_BIND"] = path
        try:
            yield path
        finally:
            if prev is None:
                os.environ.pop("VLLM_PERSONAL_CONTEXT_TEST_BIND", None)
            else:
                os.environ["VLLM_PERSONAL_CONTEXT_TEST_BIND"] = prev
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _sampling_params_vanilla():
    from vllm import SamplingParams

    return SamplingParams(max_tokens=20, temperature=0.0)


def _sampling_params_with_plan(chunk):
    """SamplingParams whose ``extra_args`` carries a reuse_plan covering
    ``chunk``. Connector parses this in ``_extract_reuse_plan``."""
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=20,
        temperature=0.0,
        extra_args={
            "kv_transfer_params": {
                "reuse_plan": {
                    "chunks": [
                        {
                            "token_ids": list(chunk.token_ids),
                            "old_pos_start": chunk.old_pos_start,
                            "salt_hex": chunk.salt.hex(),
                        }
                    ]
                }
            }
        },
    )


def _tokens_prompt(prompt_ids):
    """Pack token IDs into the ``TokensPrompt`` schema expected by
    ``LLM.generate(prompts=...)`` (it's a TypedDict; a plain dict with
    the right key works too)."""
    from vllm.inputs import TokensPrompt

    return TokensPrompt(prompt_token_ids=list(prompt_ids))


def _run_vanilla(prompt_ids):
    """Vanilla generation: no PC connector, baseline output."""
    from vllm import LLM

    llm = LLM(
        model=MODEL,
        dtype="float16",
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        attention_config={"backend": "FLASHINFER"},
    )
    try:
        out = llm.generate(
            [_tokens_prompt(prompt_ids)],
            sampling_params=[_sampling_params_vanilla()],
        )
        return out[0].outputs[0].text
    finally:
        del llm
        torch.cuda.empty_cache()


def _run_pc(prompt_ids, chunk, storage, selector):
    """PC generation: bind storage + selector via env var, then run."""
    from vllm import LLM
    from vllm.config import KVTransferConfig

    kv_transfer_config = KVTransferConfig(
        kv_connector="PersonalContextKVConnector",
        kv_role="kv_both",
    )
    with _pc_test_bind(storage, selector):
        llm = LLM(
            model=MODEL,
            dtype="float16",
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            kv_transfer_config=kv_transfer_config,
            attention_config={"backend": "FLASHINFER"},
        )
        try:
            out = llm.generate(
                [_tokens_prompt(prompt_ids)],
                sampling_params=[_sampling_params_with_plan(chunk)],
            )
            return out[0].outputs[0].text
        finally:
            del llm
            torch.cuda.empty_cache()


# ----------------------- tests -----------------------


@requires_cuda
def test_pc_no_selection_diverges_from_vanilla():
    """PC + random K/V + NoSelection → output ≠ vanilla.

    Weak signal: proves the connector is plumbed into model forward,
    but a KV-layout mismatch would also cause divergence (attention
    reads zeros instead of our scatter). Use it as a sanity check, not
    a correctness proof.
    """
    from vllm.v1.personal_context import NoSelection

    out_vanilla = _run_vanilla(PROMPT_TOKEN_IDS)
    storage, chunk = _make_storage_with_random_chunk_kv()
    out_pc = _run_pc(PROMPT_TOKEN_IDS, chunk, storage, NoSelection())

    assert out_pc != out_vanilla, (
        f"Expected divergence from vanilla under random scattered K/V "
        f"with no recomputation, got identical output:\n  {out_pc!r}"
    )


@requires_cuda
def test_pc_full_selection_matches_vanilla():
    """PC + random K/V + SelectFirstR(1.0) → output = vanilla.

    Strong signal: ``SelectFirstR(1.0)`` selects every chunk position,
    so the sparse-Q override expands the Q batch to cover all of them.
    During forward, vLLM writes fresh K/V via ``slot_mapping``
    (backend-aware, always the correct paged layout), overwriting the
    garbage we scattered. By the time attention runs, the cache holds
    exactly what vanilla would have computed.

    Any divergence here indicates a real bug in the Step 7 sparse-Q
    implementation — wrong positions, wrong slot_mapping, wrong
    ``num_scheduled_tokens`` plumbing, or RoPE indexing on selected
    positions.
    """
    from vllm.v1.personal_context import SelectFirstR

    out_vanilla = _run_vanilla(PROMPT_TOKEN_IDS)
    storage, chunk = _make_storage_with_random_chunk_kv()
    out_pc = _run_pc(PROMPT_TOKEN_IDS, chunk, storage, SelectFirstR(1.0))

    assert out_pc == out_vanilla, (
        "PC + SelectFirstR(1.0) should reproduce vanilla output via "
        "full recompute. Divergence indicates a Step 7 bug.\n"
        f"  vanilla: {out_vanilla!r}\n"
        f"  pc:      {out_pc!r}"
    )
