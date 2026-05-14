# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1 stub: in-memory KV block storage.

Concrete class, no abstract base. Phase 3 wraps this in a
``PersonalContextOffloadingManager`` (subclass of vLLM's
``OffloadingManager``). Phase 12 swaps the storage for a persistent
backend; the manager subclass is unchanged.

Intentional limits (deferred to later phases):
    - no eviction (Phase 11)
    - no persistence (Phase 12)
    - no thread safety (Phase 12)
    - no quantization (Phase 12)
"""

from __future__ import annotations

from typing import Optional

from vllm.v1.personal_context.entry import KVBlock, StoreConfig


class InMemoryStorage:
    def __init__(self, config: StoreConfig):
        self._config = config
        self._blocks: dict[bytes, KVBlock] = {}

    @property
    def config(self) -> StoreConfig:
        return self._config

    def put(self, key: bytes, block: KVBlock) -> None:
        self._blocks[key] = block

    def get(self, key: bytes) -> Optional[KVBlock]:
        return self._blocks.get(key)

    def lookup(self, keys: list[bytes]) -> list[Optional[KVBlock]]:
        """Batched ``get``; preserves order, returns ``None`` for misses."""
        return [self._blocks.get(k) for k in keys]

    def __contains__(self, key: bytes) -> bool:
        return key in self._blocks

    def __len__(self) -> int:
        return len(self._blocks)
