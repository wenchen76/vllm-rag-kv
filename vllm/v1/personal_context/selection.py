# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection logic for stale-KV personal-context reuse.

Phase 7 MVP (strategy B, "trust the retriever") treats every retrieved
chunk as fresh — its KV is computed under the same effective left-
context as the current request, so no recomputation is needed. The
selector returns ``()`` and the request runs as in our smoke-tested
control plane.

For stale-KV reuse (CacheBlend-style, when chunks are sourced from
different prefixes than the current request would imply), some retrieved
positions must be recomputed because their stale K is wrong by enough to
hurt attention output. Which positions to recompute is policy. This
module formalises that policy as a pluggable ``Selector``.

Two selectors ship here:

    - ``NoSelection``: strategy-B default (or any "all-fresh" case).
        Always returns ``()``.
    - ``SelectFirstR``: pick the first ``r``-fraction of positions in
        each chunk. This is the cheap baseline used by the original
        CacheBlend ablation: early tokens in a chunk often anchor the
        downstream attention pattern ("attention sink" effect), so
        recomputing just the prefix tends to recover most of the
        accuracy lost to stale K.

The intended dialectic is **``SelectFirstR`` vs ``HKVDSelector``**
(Hidden Key-Value Drift, layer-0-drift-based, deferred to Step 6.3):

    ``SelectFirstR``: O(1) decision per chunk, no model access.
        Bias: front-loaded — assumes early-token importance is a
        good proxy for K drift. Cheap to deploy; useful as the
        ablation baseline.

    ``HKVDSelector`` (TODO): runs layer-0 with all retrieved positions
        present, compares fresh vs stale K, recomputes the top-r
        positions by drift. Accurate to the model, costs one extra
        attention pass on layer 0. Hooks into the forward path.

The output of ``select()`` flows through ``PersonalContextReqMeta``
as ``selected_positions: tuple[int, ...]`` and is consumed by Step 7
(``_prepare_inputs`` sparse-Q override) — not yet wired.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence, runtime_checkable

from vllm.v1.personal_context.policy import ReusePlan


@runtime_checkable
class Selector(Protocol):
    """Decides which absolute prompt positions need fresh K/V.

    Implementations may be:

        - Heuristic + scheduler-side: cheap to evaluate, can run inside
          ``build_connector_meta``. ``SelectFirstR`` is the example.
        - Model-aware + worker-side: requires a forward-pass signal
          (e.g. layer-0 K drift). Deferred to Step 6.3.

    The contract for the returned value:

        - Sorted ascending.
        - Deduplicated.
        - Every element lies inside some chunk's ``[new_pos_start,
          new_pos_start + len(chunk.token_ids))``. Selecting outside
          chunks is meaningless (the standard prefill path already
          recomputes the query suffix) and is treated as a selector
          bug; the connector does not defensively clamp.

    Empty tuple is the legitimate "nothing to recompute" answer
    (strategy B). Implementations should return ``()`` rather than
    raising when there is no work.
    """

    def select(
        self,
        plan: ReusePlan,
        new_pos_starts: Sequence[int],
    ) -> tuple[int, ...]:
        """Return positions to recompute for one request's plan.

        Args:
            plan: The ``ReusePlan`` parsed from
                ``request.kv_transfer_params``.
            new_pos_starts: Per-chunk starting positions in this
                request's prompt, aligned 1-to-1 with ``plan.chunks``.
        """
        ...


class NoSelection:
    """Strategy-B default: no position is recomputed.

    Bind this — or leave the connector's selector unset, which is
    equivalent — when retrieved K/V is sourced from the same store
    as the current request. Under that invariant chunks are fresh by
    construction and recomputation is wasted work.
    """

    def select(
        self,
        plan: ReusePlan,
        new_pos_starts: Sequence[int],
    ) -> tuple[int, ...]:
        return ()


class SelectFirstR:
    """Recompute the first ``r``-fraction of positions in each chunk.

    Concretely, for a chunk of length ``L`` starting at
    ``new_pos_start = p``, this selector emits the contiguous range::

        [p, p + 1, ..., p + ceil(r * L) - 1]

    Per-chunk independent — multi-chunk plans concatenate ranges in
    chunk order.

    Why ``first``-r (not random, not stride): the empirical CacheBlend
    finding is that early positions in a chunk anchor the attention
    pattern (the "attention sink" effect). Recomputing just the
    front-loaded prefix tends to recover most of the accuracy lost to
    stale K, at a fraction of the cost of recomputing everything. This
    is the **baseline** the project will compare against
    drift-based selection (Step 6.3) on the same workload — apples-to-
    apples cost (recompute same number of positions, ``ceil(r * L)``)
    vs. apples-to-oranges quality (front-loaded vs. drift-targeted).

    Args:
        r: Fraction in ``[0.0, 1.0]``. ``r = 0.0`` is equivalent to
            ``NoSelection`` (returns ``()``); ``r = 1.0`` selects every
            position (full recompute, useful as the upper-bound
            accuracy reference).

    Raises:
        ValueError: ``r`` outside ``[0.0, 1.0]``.
    """

    def __init__(self, r: float):
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"r must be in [0.0, 1.0], got {r}")
        self._r = float(r)

    @property
    def r(self) -> float:
        return self._r

    def select(
        self,
        plan: ReusePlan,
        new_pos_starts: Sequence[int],
    ) -> tuple[int, ...]:
        if len(new_pos_starts) != len(plan.chunks):
            raise ValueError(
                f"new_pos_starts length {len(new_pos_starts)} does not "
                f"match plan.chunks length {len(plan.chunks)}"
            )
        out: list[int] = []
        for chunk, start in zip(plan.chunks, new_pos_starts):
            length = len(chunk.token_ids)
            k = math.ceil(self._r * length)
            if k <= 0:
                continue
            base = int(start)
            out.extend(range(base, base + k))
        return tuple(out)
