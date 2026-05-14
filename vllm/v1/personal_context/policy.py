# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request reuse policy and cache-isolation invariant.

vLLM operates two KV cache tables side-by-side under this design:

    1. The prefix-rolling cache (``vllm.v1.core.block_pool``): keyed by a
       hash that folds in the parent block hash, so the same tokens at
       different prompt prefixes get different keys. Stores K, V computed
       at their final prompt positions.

    2. The personal-context store (``vllm.v1.personal_context``): keyed by
       a position-independent content hash over token IDs plus a domain
       separator. Stores K post-RoPE at the original encoding position;
       a delta rotation is applied at load time to relocate K to the new
       prompt position. V is position-independent.

The two schemas are incompatible — values stored under one schema are
meaningless under the other. The cache-isolation invariant:

    A request that reuses retrieved-chunk KV from the personal-context
    store MUST NOT write any block of that request into the prefix-rolling
    cache. ``assert_can_write_prefix_cache()`` is the guard at the write
    site; the scheduler / connector wrapping the prefix cache calls it
    before every block insertion.

The cache-isolation invariant matters because retrieved-chunk K is stored
post-RoPE at its original position. If such K ever leaked into the
prefix-rolling cache, a later request with the same prompt prefix would
read K rotated at the wrong position — silent correctness regression.
Loud rejection is preferred over silent pollution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.v1.personal_context.chunk import Chunk


class PrefixCachePollutionError(ValueError):
    """A mixed-KV request attempted to write into the prefix-rolling cache."""


@dataclass(frozen=True)
class ReusePlan:
    """Per-request plan for reusing retrieved-chunk KV.

    A request with at least one chunk in its plan is "mixed-KV" and must
    not pollute the prefix-rolling cache. A request with no chunks (or no
    plan at all) is vanilla and may use the prefix cache normally.
    """

    chunks: tuple[Chunk, ...] = field(default_factory=tuple)

    def is_mixed_kv(self) -> bool:
        return bool(self.chunks)


def assert_can_write_prefix_cache(plan: ReusePlan | None) -> None:
    """Guard for the prefix-rolling cache write site.

    Raises ``PrefixCachePollutionError`` if the request carries a mixed-KV
    reuse plan. Callers wrap prefix-cache inserts with this guard and
    fall back to skipping the insert (the prefix cache is best-effort
    anyway) rather than risking schema-incompatible pollution.
    """
    if plan is not None and plan.is_mixed_kv():
        raise PrefixCachePollutionError(
            "Refusing to write the prefix-rolling cache for a mixed-KV "
            "request: retrieved-chunk K is stored post-RoPE at its "
            "original position and is not interchangeable with prefix-"
            "cache K. See vllm.v1.personal_context.policy."
        )
