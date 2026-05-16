# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request lookup connector for the personal-context KV store.

The scheduler calls ``PersonalContextConnector.lookup(plan)`` to learn
which retrieved-chunk blocks are already in the store. Every hit is one
``block_size`` worth of tokens that the scheduler can deduct from the
prefill budget — those tokens will be filled in later by the load +
delta RoPE path rather than re-computed from the model.

The connector is read-only and single-request scoped: it does not mutate
the store, batch across requests, or know anything about scheduler
state. Wiring the result into the v1 scheduler is a separate change.

Alignment violations propagate as ``AlignmentError`` from
``Chunk.block_hashes()`` — callers catch it and fall back to oracle-
path full prefill.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.personal_context.chunk import Chunk
from vllm.v1.personal_context.entry import KVBlock
from vllm.v1.personal_context.policy import ReusePlan
from vllm.v1.personal_context.storage import InMemoryStorage


@dataclass(frozen=True)
class BlockLookup:
    """Lookup result for one block hash."""

    key: bytes
    block: KVBlock | None

    @property
    def hit(self) -> bool:
        return self.block is not None


@dataclass(frozen=True)
class ChunkLookup:
    """Per-chunk lookup result; blocks are in chunk order."""

    chunk: Chunk
    blocks: tuple[BlockLookup, ...]

    @property
    def hit_count(self) -> int:
        return sum(1 for b in self.blocks if b.hit)

    @property
    def all_hit(self) -> bool:
        return bool(self.blocks) and all(b.hit for b in self.blocks)

    @property
    def all_miss(self) -> bool:
        return all(not b.hit for b in self.blocks)


@dataclass(frozen=True)
class PlanLookup:
    """Plan-wide lookup result.

    ``free_tokens`` is the value the scheduler subtracts from the
    prefill token budget for this request.
    """

    chunks: tuple[ChunkLookup, ...]
    block_size: int

    @property
    def total_hit_blocks(self) -> int:
        return sum(c.hit_count for c in self.chunks)

    @property
    def free_tokens(self) -> int:
        return self.total_hit_blocks * self.block_size


class PersonalContextConnector:
    """Read-only single-request lookup for the personal-context KV store."""

    def __init__(self, storage: InMemoryStorage):
        self._storage = storage

    @property
    def block_size(self) -> int:
        return self._storage.config.block_size

    def lookup(self, plan: ReusePlan) -> PlanLookup:
        """Probe the store for every block of every chunk in ``plan``.

        Propagates ``AlignmentError`` from ``Chunk.block_hashes()`` so
        the scheduler can catch it and fall back to full prefill.
        """
        block_size = self.block_size
        results: list[ChunkLookup] = []
        for chunk in plan.chunks:
            hashes = chunk.block_hashes(block_size)
            blocks = self._storage.lookup(hashes)
            results.append(
                ChunkLookup(
                    chunk=chunk,
                    blocks=tuple(
                        BlockLookup(key=h, block=b)
                        for h, b in zip(hashes, blocks)
                    ),
                )
            )
        return PlanLookup(chunks=tuple(results), block_size=block_size)
