# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV connector for cross-request RAG-chunk KV reuse (personal-context).

This is the vLLM-facing ``KVConnectorBase_V1`` implementation that lets
the scheduler discount a request's prefill budget by the number of
tokens already in the personal-context store. The store side (block
layout, content hashing, delta-RoPE, scatter primitives) lives under
``vllm.v1.personal_context``; this module is the glue.

Phase 7 Step 1-2 scope (this file):

    - ``get_num_new_matched_tokens`` is wired with the strategy-B
      reporting model: trust the retriever (chunks were sourced from
      this same store), so the matched-tokens count is the sum of
      every chunk's token length. Lookup is run defensively to catch
      eviction races and alignment violations — any miss falls back
      to oracle-path full prefill until selective recompute lands
      in Phase 9.

    - All other ``KVConnectorBase_V1`` abstract methods are stubs.
      Step 3-9 (block allocation tracking, scatter, sparse-Q prefill,
      runner integration) will fill them in later commits.

Naming: the Phase 4 lookup wrapper at
``vllm.v1.personal_context.PersonalContextConnector`` is a *storage*
helper. This class is the *vLLM* connector. They are kept as separate
types intentionally; this file imports the lookup helper privately
under an alias to keep the public class name unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.personal_context.chunk import AlignmentError, Chunk
from vllm.v1.personal_context.connector import (
    PersonalContextConnector as _StorageLookup,
)
from vllm.v1.personal_context.policy import ReusePlan
from vllm.v1.personal_context.storage import InMemoryStorage

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(frozen=True)
class _PendingLoad:
    """Per-request state stashed by Step 3 for Step 4-5 to consume.

    ``update_state_after_alloc`` records ``(plan, num_external_tokens)``
    keyed by ``request_id`` once the scheduler has accepted our coverage
    claim and allocated blocks. ``build_connector_meta`` (Step 4) drains
    this dict, pairs each entry with the scheduler-assigned block ids
    pulled from ``scheduler_output``, and packages a
    ``PersonalContextReqMeta`` for the worker.
    """

    plan: ReusePlan
    num_external_tokens: int


@dataclass
class PersonalContextConnectorMetadata(KVConnectorMetadata):
    """Scheduler → worker handoff payload.

    Step 4 (``build_connector_meta``) will populate per-request
    scatter / load directives here. For Step 1-3 it is an empty
    placeholder so the abstract base type contract is satisfied.
    """

    pass


class PersonalContextKVConnector(KVConnectorBase_V1):
    """vLLM v1 KV connector for personal-context RAG-chunk reuse.

    Step 1-2 wires only the scheduler-side ``get_num_new_matched_tokens``
    hook with strategy B (trust-the-retriever, defensive verify).
    Every other abstract method is stubbed and will be filled by
    later Phase-7 steps.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._storage: InMemoryStorage | None = None
        self._lookup: _StorageLookup | None = None
        # Step 3 state: request_id → (plan, num_external_tokens). Drained
        # by ``build_connector_meta`` (Step 4) once metadata is shipped.
        self._pending_loads: dict[str, _PendingLoad] = {}

    def bind_storage(self, storage: InMemoryStorage) -> None:
        """Attach a pre-built storage backend.

        The skeleton has no production retrieval pipeline; tests and
        Phase 12 wiring use this hook to install a populated store.
        When unbound, ``get_num_new_matched_tokens`` reports zero
        matched tokens (i.e. the connector is effectively disabled).
        """
        if self._block_size != storage.config.block_size:
            raise ValueError(
                f"vllm cache block_size {self._block_size} does not match "
                f"store block_size {storage.config.block_size}"
            )
        self._storage = storage
        self._lookup = _StorageLookup(storage)

    # ----------------- Scheduler-side: Step 1-2 wired -----------------

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Report retrieved-chunk tokens the scheduler can skip.

        Strategy B (trust-the-retriever): the retriever sources chunks
        from the same store this connector queries, so plan chunks are
        all-hit by construction. We report the full chunk-token length
        and let the worker-side load pipeline (Step 6) handle the
        actual fetch + delta-RoPE + scatter.

        Defensive checks run regardless:

            - ``AlignmentError`` from ``Chunk.block_hashes()`` → return 0
            - any store miss (eviction race / out-of-sync backend) →
              return 0. Selective recompute is Phase 9 work; until then
              a miss is a fail-closed signal that demotes the request
              back to full prefill.

        Returns ``(0, False)`` whenever the connector has no storage
        bound, the request carries no plan, or any defensive check
        fails. The second tuple element is always ``False`` —
        personal-context load is synchronous (Step 6 runs
        ``scatter_loaded_plan`` inside ``start_load_kv`` directly).

        Side-effect free per ``KVConnectorBase_V1`` contract: repeated
        calls with the same request return the same value and do not
        mutate connector state.
        """
        if self._lookup is None:
            return 0, False
        plan = self._extract_reuse_plan(request)
        if plan is None or not plan.chunks:
            return 0, False
        total = sum(len(c.token_ids) for c in plan.chunks)
        try:
            result = self._lookup.lookup(plan)
        except AlignmentError as e:
            logger.warning(
                "Request %s: discarding reuse plan due to alignment "
                "violation: %s",
                request.request_id,
                e,
            )
            return 0, False
        if not all(c.all_hit for c in result.chunks):
            logger.warning(
                "Request %s: reuse plan has store misses (eviction race "
                "or store/retriever divergence). Selective recompute is "
                "not yet wired (Phase 9); falling back to full prefill.",
                request.request_id,
            )
            return 0, False
        return total, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Step 3: stash the request's plan for Step 4-5 to consume.

        The scheduler invokes this hook after allocating blocks for a
        request whose external coverage we claimed in
        ``get_num_new_matched_tokens``. We record
        ``(plan, num_external_tokens)``; ``build_connector_meta``
        (Step 4) pairs this with scheduler-assigned block ids drawn
        from ``scheduler_output`` and ships a ``PersonalContextReqMeta``
        to the worker.

        The ``blocks`` argument is intentionally not consumed here.
        ``KVCacheBlocks`` returned by ``get_blocks(request_id)``
        enumerates *all* blocks for the request — including the
        prefix-cached prefix and the freshly-allocated remainder for
        prefill — without exposing the offset where our externally
        cached range starts. Step 4 instead reads block ids from
        ``scheduler_output.scheduled_new_reqs[*].block_ids``, which
        every other connector also uses, and which matches the chunk
        position layout this connector wrote into the plan.

        No-ops when ``num_external_tokens <= 0`` (the scheduler did
        not accept our claim, or we never made one). Also no-ops with
        a warning when the request's plan is missing or disagrees with
        ``num_external_tokens`` — these are state-divergence symptoms
        and the safe response is to drop the load and fall back to
        normal prefill.
        """
        if num_external_tokens <= 0:
            return
        plan = self._extract_reuse_plan(request)
        if plan is None or not plan.chunks:
            logger.warning(
                "Request %s: update_state_after_alloc got "
                "num_external_tokens=%d but no usable reuse plan; "
                "ignoring (state divergence — request will fall back to "
                "full prefill).",
                request.request_id,
                num_external_tokens,
            )
            return
        expected = sum(len(c.token_ids) for c in plan.chunks)
        if num_external_tokens != expected:
            logger.warning(
                "Request %s: num_external_tokens (%d) does not match "
                "plan total (%d); ignoring.",
                request.request_id,
                num_external_tokens,
                expected,
            )
            return
        self._pending_loads[request.request_id] = _PendingLoad(
            plan=plan,
            num_external_tokens=num_external_tokens,
        )

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        """Step 5 stub: empty metadata; Step 5 will populate."""
        return PersonalContextConnectorMetadata()

    # ----------------- Worker-side: Step 6-9 stubs -----------------

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        """Step 6 stub: load + delta-RoPE + scatter pipeline."""
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """No-op: this connector is read-only.

        Population of the personal-context store is the retriever /
        offline-indexer's job, not the inference path's.
        """
        return

    def wait_for_save(self) -> None:
        return

    # ----------------- Helpers -----------------

    @staticmethod
    def _extract_reuse_plan(request: "Request") -> ReusePlan | None:
        """Parse a ``ReusePlan`` from ``request.kv_transfer_params``.

        Expected JSON-compatible shape::

            {
                "reuse_plan": {
                    "chunks": [
                        {
                            "token_ids": [int, ...],
                            "old_pos_start": int,
                            "salt_hex": str (optional, default "")
                        },
                        ...
                    ]
                }
            }

        Returns ``None`` if the field is absent or malformed (missing
        keys, wrong types, bad hex). The scheduler contract says we
        report zero matched tokens on uncertainty, never raise.
        """
        params = request.kv_transfer_params
        if not params:
            return None
        plan_data = params.get("reuse_plan")
        if not isinstance(plan_data, dict):
            return None
        try:
            chunk_dicts = plan_data["chunks"]
            chunks = tuple(
                Chunk(
                    token_ids=tuple(int(t) for t in c["token_ids"]),
                    old_pos_start=int(c["old_pos_start"]),
                    salt=bytes.fromhex(c.get("salt_hex", "")),
                )
                for c in chunk_dicts
            )
            return ReusePlan(chunks=chunks)
        except (KeyError, TypeError, ValueError):
            return None
