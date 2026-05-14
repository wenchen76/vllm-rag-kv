# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-memory KV block storage.

Concrete class, no abstract base. A later phase wraps this in a
``PersonalContextOffloadingManager`` (subclass of vLLM's
``OffloadingManager``) and swaps the backend for a persistent store;
the manager subclass is unchanged.

Intentional limits (deferred):
    - no eviction
    - no persistence
    - no thread safety
    - no quantization

``put()`` enforces the block-shape side of the alignment contract: tensor
list length, per-tensor shape, dtype, and ``old_pos_start`` must agree
with ``StoreConfig``. Mismatches raise so the caller can fall back to
oracle-path full prefill rather than silently storing a corrupt entry.
"""

from __future__ import annotations

from typing import Optional

from vllm.v1.personal_context.chunk import AlignmentError
from vllm.v1.personal_context.entry import KVBlock, StoreConfig


class InMemoryStorage:
    def __init__(self, config: StoreConfig):
        self._config = config
        self._blocks: dict[bytes, KVBlock] = {}

    @property
    def config(self) -> StoreConfig:
        return self._config

    def put(self, key: bytes, block: KVBlock) -> None:
        self._validate(block)
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

    def _validate(self, block: KVBlock) -> None:
        cfg = self._config
        if len(block.keys) != cfg.num_layers:
            raise ValueError(
                f"KVBlock has {len(block.keys)} key tensors, "
                f"expected {cfg.num_layers}"
            )
        if len(block.values) != cfg.num_layers:
            raise ValueError(
                f"KVBlock has {len(block.values)} value tensors, "
                f"expected {cfg.num_layers}"
            )
        if block.old_pos_start < 0 or block.old_pos_start % cfg.block_size != 0:
            raise AlignmentError(
                f"KVBlock.old_pos_start {block.old_pos_start} is not a "
                f"non-negative multiple of block_size {cfg.block_size}"
            )
        expected = (cfg.block_size, cfg.num_kv_heads, cfg.head_dim)
        for layer_idx, (k, v) in enumerate(zip(block.keys, block.values)):
            if tuple(k.shape) != expected:
                raise ValueError(
                    f"KVBlock layer {layer_idx} K shape {tuple(k.shape)}, "
                    f"expected {expected}"
                )
            if tuple(v.shape) != expected:
                raise ValueError(
                    f"KVBlock layer {layer_idx} V shape {tuple(v.shape)}, "
                    f"expected {expected}"
                )
            if k.dtype != cfg.dtype or v.dtype != cfg.dtype:
                raise ValueError(
                    f"KVBlock layer {layer_idx} dtype K={k.dtype} V={v.dtype}, "
                    f"expected {cfg.dtype}"
                )
