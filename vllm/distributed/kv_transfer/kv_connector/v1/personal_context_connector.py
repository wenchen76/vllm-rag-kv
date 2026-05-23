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

import itertools
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
from vllm.v1.personal_context.load import load_plan
from vllm.v1.personal_context.policy import ReusePlan
from vllm.v1.personal_context.scatter import scatter_loaded_plan
from vllm.v1.personal_context.selection import SelectFirstR, Selector
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


def _derive_rope_theta(vllm_config: Any) -> float:
    """Best-effort lookup of the model's RoPE base.

    Production ``VllmConfig`` exposes
    ``vllm_config.model_config.hf_config.rope_theta``; minimal test
    fixtures may not. Falls back to ``10000.0`` (the apply_rope default)
    on any missing attribute or unparseable value.
    """
    try:
        return float(vllm_config.model_config.hf_config.rope_theta)
    except (AttributeError, TypeError, ValueError):
        return 10000.0


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


@dataclass(frozen=True)
class PersonalContextReqMeta:
    """Worker-side load directive for one personal-context request.

    Carries the minimum Step 5 (``start_load_kv``) needs to run

        loaded_plan = load_plan(lookup_result, new_pos_starts, rope_theta)
        scatter_loaded_plan(loaded_plan, kv_caches, block_assignments)

    against the worker's local store and paged cache.

    Fields
    ------
    request_id
        Identification only. Useful for logs and per-request cleanup.
    plan
        The same ``ReusePlan`` the scheduler-side connector parsed from
        ``request.kv_transfer_params``. The worker re-runs lookup on its
        own storage handle against this plan (same backend, so hits are
        identical modulo eviction races).
    block_assignments
        ``block_assignments[i][j]`` is the physical block id where the
        ``j``-th block of the ``i``-th chunk should land in the paged
        KV cache. Outer length == ``len(plan.chunks)``; inner length
        matches each chunk's block count.
    new_pos_starts
        ``new_pos_starts[i]`` is the absolute position in this request's
        prompt where the ``i``-th chunk begins. Used by ``load_plan``
        to compute the delta-RoPE shift (``delta = new - old``).
    """

    request_id: str
    plan: ReusePlan
    block_assignments: tuple[tuple[int, ...], ...]
    new_pos_starts: tuple[int, ...]
    selected_positions: tuple[int, ...] = ()
    """Absolute prompt positions whose K/V should be recomputed.

    Empty == strategy B (every retrieved position is fresh; no
    recomputation needed). Non-empty == CacheBlend-style stale-KV
    reuse — Step 7 consumes this to widen the sparse-Q batch from
    just the query suffix to ``selected_positions ∪ query_positions``.

    Populated by ``build_connector_meta`` via the bound ``Selector``;
    the default (no selector bound) keeps this empty and the worker-
    side path identical to the strategy-B MVP.
    """


@dataclass
class PersonalContextConnectorMetadata(KVConnectorMetadata):
    """Scheduler → worker handoff payload.

    ``requests`` enumerates every request whose load is to be performed
    in this step. The worker-side connector iterates this in
    ``start_load_kv`` (Step 5). Empty tuple is the legitimate "no load"
    case (no personal-context requests this step) — not an error.
    """

    requests: tuple[PersonalContextReqMeta, ...] = ()


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
        # Per-model RoPE base; used by ``load_plan`` to compute the
        # delta rotation. Pull from the model config if available so the
        # production path picks up the right value automatically; the
        # 10000.0 fallback matches ``apply_rope_at_positions``'s default
        # and keeps minimal test fixtures working.
        self._rope_theta: float = _derive_rope_theta(vllm_config)
        # Step 6.1 stale-KV selection. ``None`` keeps strategy-B
        # behaviour (no recomputation, ``selected_positions`` always
        # empty). Install via ``bind_selector``.
        #
        # Resolution order (later overrides earlier):
        #   1. ``None`` default (strategy-B).
        #   2. ``kv_connector_extra_config.selector`` from the engine
        #      config — production path. Frozen at engine startup.
        #   3. ``VLLM_PERSONAL_CONTEXT_TEST_BIND`` pickle — test path.
        #      Wins so GPU e2e tests can plug in arbitrary selectors
        #      without touching engine config.
        self._selector: Selector | None = self._resolve_selector_from_config()
        self._maybe_auto_bind_from_env()

    def _resolve_selector_from_config(self) -> Selector | None:
        """Build a ``Selector`` from ``kv_connector_extra_config``.

        Recognised ``selector`` shapes in ``kv_connector_extra_config``:

            { "selector": null }                       → ``None`` (strategy-B)
            { "selector": "NoSelection" }              → ``None``
            { "selector": "SelectFirstR" }             → ``SelectFirstR(r=1.0)``
            { "selector": {"type": "SelectFirstR",
                           "r": 0.5} }                 → ``SelectFirstR(r=0.5)``

        Unknown types or malformed entries log a warning and fall back
        to ``None`` so a typo in config does not crash the engine — the
        request just runs as strategy-B.

        Frozen at engine startup; per-request selection control is out
        of scope and deferred to a later phase if needed.
        """
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        spec = extra.get("selector")
        if spec is None:
            return None

        if isinstance(spec, str):
            name = spec
            params: dict[str, Any] = {}
        elif isinstance(spec, dict):
            name = spec.get("type", "")
            params = {k: v for k, v in spec.items() if k != "type"}
        else:
            logger.warning(
                "PersonalContextKVConnector: kv_connector_extra_config."
                "selector must be a string or dict, got %s; ignoring.",
                type(spec).__name__,
            )
            return None

        normalised = name.strip().lower()
        if normalised in ("", "none", "noselection"):
            return None
        if normalised in ("first_r", "selectfirstr"):
            try:
                r = float(params.get("r", 1.0))
            except (TypeError, ValueError):
                logger.warning(
                    "PersonalContextKVConnector: SelectFirstR.r must be a "
                    "float in [0.0, 1.0], got %r; ignoring selector config.",
                    params.get("r"),
                )
                return None
            try:
                return SelectFirstR(r=r)
            except ValueError as e:
                logger.warning(
                    "PersonalContextKVConnector: %s; ignoring selector config.",
                    e,
                )
                return None
        logger.warning(
            "PersonalContextKVConnector: unknown selector type %r; "
            "ignoring (request runs as strategy-B).",
            name,
        )
        return None

    def _maybe_auto_bind_from_env(self) -> None:
        """If ``VLLM_PERSONAL_CONTEXT_TEST_BIND`` is set, load the pickle
        at that path and apply ``bind_storage`` / ``bind_selector``.

        Silently no-ops when the env var is absent. Failures (bad path,
        unpicklable, mismatched block size) log and continue with the
        unbound default — the test then surfaces the mistake as an
        assertion failure rather than a confusing import-time crash.
        """
        import os

        path = os.environ.get("VLLM_PERSONAL_CONTEXT_TEST_BIND")
        if not path:
            return
        if not os.path.exists(path):
            logger.warning(
                "VLLM_PERSONAL_CONTEXT_TEST_BIND=%s does not exist; "
                "skipping auto-bind.",
                path,
            )
            return
        try:
            import pickle

            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            logger.exception(
                "Failed to load PC auto-bind pickle from %s", path
            )
            return
        storage = payload.get("storage")
        selector = payload.get("selector")
        if storage is not None:
            try:
                self.bind_storage(storage)
            except Exception:
                logger.exception(
                    "Auto bind_storage failed; connector stays unbound."
                )
        if selector is not None:
            self.bind_selector(selector)

    def bind_selector(self, selector: Selector | None) -> None:
        """Install (or clear) the stale-KV ``Selector``.

        ``None`` reverts to the strategy-B default — every retrieved
        position is treated as fresh and ``selected_positions`` stays
        empty. A bound selector is invoked once per request inside
        ``build_connector_meta`` (Step 4) with the request's plan and
        per-chunk new positions; its sorted output flows to the worker
        as ``PersonalContextReqMeta.selected_positions`` and is
        consumed by Step 7's sparse-Q override (not yet wired).
        """
        self._selector = selector

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

            - **Placement validation**: chunk token IDs must match the
              prompt slice at ``[num_computed_tokens, ... + total)``.
              If not (most commonly because an assumed prefix such as a
              system prompt is not actually prefix-cached, so the chunk
              would land at the wrong position), → return 0. Must
              happen here (before scheduler's ``allocate_slots`` commits
              ``num_computed_tokens``) because no later hook can undo
              the allocation.
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

        # Placement validation: PC's chunk scatter writes K/V to paged
        # cache slots at positions ``[num_computed_tokens, ... + total)``
        # under the assumption that the chunk token IDs match the prompt
        # at that offset. The scheduler reaches that assumption by
        # treating ``num_computed_tokens`` (local prefix-cache hits) as
        # the immediate prefix to PC's external coverage. When that
        # assumption fails — most commonly because the assumed prefix
        # (e.g., a system prompt) is not actually prefix-cached — the
        # scheduler would skip prefill for tokens whose K/V we never
        # write, producing silent garbage.
        #
        # Bail out early so this connector returns 0 matched tokens and
        # the request falls back to a vanilla full prefill. This must
        # happen here (before ``allocate_slots`` is called in the
        # scheduler) because later hooks cannot undo the allocation.
        prompt_token_ids = request.prompt_token_ids
        end = num_computed_tokens + total
        if prompt_token_ids is None or end > len(prompt_token_ids):
            logger.warning(
                "Request %s: PC reuse plan overflows prompt "
                "(local_prefix=%d + chunks=%d > prompt_len=%s); "
                "falling back to full prefill.",
                request.request_id,
                num_computed_tokens,
                total,
                len(prompt_token_ids) if prompt_token_ids is not None else None,
            )
            return 0, False
        expected_tokens = list(
            itertools.chain.from_iterable(c.token_ids for c in plan.chunks)
        )
        actual_tokens = list(prompt_token_ids[num_computed_tokens:end])
        if actual_tokens != expected_tokens:
            logger.warning(
                "Request %s: chunk tokens do not match prompt at offset "
                "%d (likely prefix-cache miss on the assumed prefix); "
                "falling back to full prefill.",
                request.request_id,
                num_computed_tokens,
            )
            return 0, False

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
        """Step 4: drain ``_pending_loads`` into a worker-bound payload.

        For every request stashed by ``update_state_after_alloc``, look
        it up in ``scheduler_output.scheduled_new_reqs`` to pick up the
        scheduler-assigned block ids and final ``num_computed_tokens``.
        Slice the block-id list to the range covering our externally-
        loaded tokens and split per chunk, then build a
        ``PersonalContextReqMeta`` per request.

        The local-prefix offset is recovered as
        ``num_computed_tokens - num_external_tokens``: by the time
        ``build_connector_meta`` runs, the scheduler has already set
        ``request.num_computed_tokens`` to ``local + external`` (see
        scheduler.py:760 / :783), and ``NewRequestData`` captured that
        same value (output.py:60). Both must be block-aligned by the
        scheduler's own invariants.

        ``_pending_loads`` is unconditionally cleared at the end —
        every pending entry is either packaged into the meta or
        considered dropped (defensive warn). A pending entry never
        carries over to a later step.
        """
        meta_requests: list[PersonalContextReqMeta] = []
        for new_req in scheduler_output.scheduled_new_reqs:
            pending = self._pending_loads.get(new_req.req_id)
            if pending is None:
                continue
            req_meta = self._build_req_meta(new_req, pending)
            if req_meta is not None:
                meta_requests.append(req_meta)
        # Any entries left in _pending_loads weren't matched to a
        # scheduled_new_reqs row this step — that's a state-divergence
        # symptom (request should have been scheduled the same step it
        # was admitted). Drop them silently here; the lack of meta will
        # make Step 5 a no-op for those requests, which is the safe
        # outcome.
        self._pending_loads.clear()
        return PersonalContextConnectorMetadata(
            requests=tuple(meta_requests)
        )

    def _build_req_meta(
        self,
        new_req: Any,
        pending: _PendingLoad,
    ) -> "PersonalContextReqMeta | None":
        """Pair one ``NewRequestData`` with its pending plan.

        Returns ``None`` (with a warning) on any geometry mismatch —
        block_ids too short, local-prefix not block-aligned, or
        per-chunk block count overrunning the slice. Caller treats
        ``None`` as "drop this load, request will fall back to full
        prefill".
        """
        # Single KV cache group expected for personal-context (Phase 7
        # MVP scope — multi-group hybrid models are deferred).
        if len(new_req.block_ids) != 1:
            logger.warning(
                "Request %s: expected 1 KV cache group, got %d; "
                "dropping load.",
                new_req.req_id,
                len(new_req.block_ids),
            )
            return None
        all_block_ids = new_req.block_ids[0]

        local_prefix = new_req.num_computed_tokens - pending.num_external_tokens
        if local_prefix < 0 or local_prefix % self._block_size != 0:
            logger.warning(
                "Request %s: unexpected local prefix %d (computed=%d, "
                "external=%d); dropping load.",
                new_req.req_id,
                local_prefix,
                new_req.num_computed_tokens,
                pending.num_external_tokens,
            )
            return None

        block_cursor = local_prefix // self._block_size
        block_assignments: list[tuple[int, ...]] = []
        new_pos_starts: list[int] = []
        pos_cursor = local_prefix
        for chunk in pending.plan.chunks:
            num_blocks = len(chunk.token_ids) // self._block_size
            end = block_cursor + num_blocks
            if end > len(all_block_ids):
                logger.warning(
                    "Request %s: block_ids range too short for chunk "
                    "(have %d, need %d); dropping load.",
                    new_req.req_id,
                    len(all_block_ids),
                    end,
                )
                return None
            block_assignments.append(tuple(all_block_ids[block_cursor:end]))
            new_pos_starts.append(pos_cursor)
            block_cursor = end
            pos_cursor += len(chunk.token_ids)

        new_pos_starts_tup = tuple(new_pos_starts)
        selected_positions = self._invoke_selector(
            pending.plan, new_pos_starts_tup
        )

        return PersonalContextReqMeta(
            request_id=new_req.req_id,
            plan=pending.plan,
            block_assignments=tuple(block_assignments),
            new_pos_starts=new_pos_starts_tup,
            selected_positions=selected_positions,
        )

    def _invoke_selector(
        self,
        plan: ReusePlan,
        new_pos_starts: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Run the bound selector; sort + dedupe + return ``()`` on error.

        A selector raising or returning a non-tuple/list is treated as
        "selector misbehaved" — we fall back to strategy-B (no
        recomputation) rather than letting the exception escape into
        the scheduler. The selector's contract says positions must be
        sorted and in-range; we sort/dedupe defensively here but do not
        clamp out-of-range (that surfaces as a downstream Q-shape
        mismatch, which is the right failure mode).
        """
        if self._selector is None:
            return ()
        try:
            raw = self._selector.select(plan, new_pos_starts)
        except Exception:  # noqa: BLE001 — selector is user-pluggable
            logger.exception(
                "Selector %s raised; falling back to no selection.",
                type(self._selector).__name__,
            )
            return ()
        return tuple(sorted(set(raw)))

    # ----------------- Worker-side: Step 6-9 stubs -----------------

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        """Step 5: run lookup + load_plan + scatter_loaded_plan per request.

        For every ``PersonalContextReqMeta`` in the connector metadata
        bound to this forward pass:

            1. ``_lookup.lookup(meta.plan)`` re-runs the storage probe
               on the worker-side store (same backend as scheduler).
            2. ``load_plan(lookup, meta.new_pos_starts, rope_theta)``
               materialises K (delta-RoPE applied) and V tensors per
               layer per chunk per block.
            3. ``scatter_loaded_plan(loaded, kv_caches, block_assignments)``
               copies them into the paged KV cache at the
               scheduler-assigned slots.

        Each request is wrapped in its own try block so an
        ``AlignmentError`` or other failure on one request does not
        wreck the rest of the batch — the bad request simply gets no
        K/V loaded and will read garbage on attention, which surfaces
        loudly downstream. Hard scatter shape / dtype mismatches do
        ``raise``, since those indicate model/store configuration
        divergence the caller has to fix.

        No-ops if there is no metadata bound, no requests in it, no
        worker-side storage bound, or no KV caches discoverable in
        ``forward_context``.
        """
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, PersonalContextConnectorMetadata):
            logger.warning(
                "start_load_kv: bound metadata is %s, not "
                "PersonalContextConnectorMetadata; skipping.",
                type(metadata).__name__,
            )
            return
        if not metadata.requests:
            return
        if self._lookup is None:
            logger.warning(
                "start_load_kv: worker-side storage is not bound but "
                "%d request(s) are queued for load; skipping.",
                len(metadata.requests),
            )
            return
        kv_caches = self._extract_kv_caches(forward_context)
        if not kv_caches:
            logger.warning(
                "start_load_kv: no KV cache layers discovered in "
                "forward_context; skipping load for %d request(s).",
                len(metadata.requests),
            )
            return

        for req_meta in metadata.requests:
            try:
                lookup_result = self._lookup.lookup(req_meta.plan)
            except AlignmentError as e:
                logger.warning(
                    "Request %s: alignment error during worker-side "
                    "lookup: %s. Skipping load (request will read "
                    "uninitialised KV — fail loudly downstream).",
                    req_meta.request_id,
                    e,
                )
                continue
            if not all(c.all_hit for c in lookup_result.chunks):
                logger.warning(
                    "Request %s: store miss on worker side (eviction "
                    "between scheduler and worker?). Skipping load.",
                    req_meta.request_id,
                )
                continue
            loaded = load_plan(
                lookup_result,
                req_meta.new_pos_starts,
                rope_theta=self._rope_theta,
            )
            scatter_loaded_plan(
                loaded, kv_caches, req_meta.block_assignments
            )

    def _extract_kv_caches(
        self, forward_context: "ForwardContext | None"
    ) -> list[torch.Tensor]:
        """Pull per-layer paged KV cache tensors from the forward context.

        Returns layers in registration order (``no_compile_layers`` is
        an insertion-ordered dict). The order must match the per-layer
        K/V order in ``KVBlock.keys`` so that
        ``scatter_loaded_plan(loaded, kv_caches, ...)``'s zip lines up.

        Empty list when the context is absent or exposes no layers
        with a ``kv_cache`` attribute — Step 5 treats that as a no-op
        skip rather than an error so dry-run / smoke tests can call
        ``start_load_kv`` without a fully wired model.
        """
        if forward_context is None:
            return []
        layers = getattr(forward_context, "no_compile_layers", None)
        if not layers:
            return []
        kv_caches: list[torch.Tensor] = []
        for layer in layers.values():
            cache = getattr(layer, "kv_cache", None)
            if cache is not None:
                kv_caches.append(cache)
        return kv_caches

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
