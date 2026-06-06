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

import torch

from vllm.v1.personal_context.chunk import AlignmentError
from vllm.v1.personal_context.entry import KVBlock, StoreConfig


def validate_block(config: StoreConfig, block: KVBlock) -> None:
    """Enforce KVBlock matches ``config`` (layer count, per-tensor shape,
    dtype, ``old_pos_start`` alignment).

    Module-level helper so every storage backend — in-memory, Redis,
    future disk or remote — runs the same shape-side of the alignment
    contract on ``put``. Raises on first failure; never returns False.
    """
    if len(block.keys) != config.num_layers:
        raise ValueError(
            f"KVBlock has {len(block.keys)} key tensors, "
            f"expected {config.num_layers}"
        )
    if len(block.values) != config.num_layers:
        raise ValueError(
            f"KVBlock has {len(block.values)} value tensors, "
            f"expected {config.num_layers}"
        )
    if block.old_pos_start < 0 or block.old_pos_start % config.block_size != 0:
        raise AlignmentError(
            f"KVBlock.old_pos_start {block.old_pos_start} is not a "
            f"non-negative multiple of block_size {config.block_size}"
        )
    # When quant="int8" the stored tensors are int8 (config.dtype stays the
    # compute/dequant dtype); otherwise they are config.dtype.
    storage_dtype = torch.int8 if config.quant == "int8" else config.dtype
    expected = (config.block_size, config.num_kv_heads, config.head_dim)
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
        if k.dtype != storage_dtype or v.dtype != storage_dtype:
            raise ValueError(
                f"KVBlock layer {layer_idx} dtype K={k.dtype} V={v.dtype}, "
                f"expected {storage_dtype} (quant={config.quant})"
            )
    if config.quant == "int8":
        if block.k_scales is None or block.v_scales is None:
            raise ValueError("quant=int8 requires k_scales and v_scales")
        if (len(block.k_scales) != config.num_layers
                or len(block.v_scales) != config.num_layers):
            raise ValueError(
                f"quant=int8 scales must have {config.num_layers} entries, got "
                f"K={len(block.k_scales)} V={len(block.v_scales)}"
            )
    elif block.k_scales is not None or block.v_scales is not None:
        raise ValueError("quant=none but KVBlock carries scales")


class InMemoryStorage:
    def __init__(self, config: StoreConfig):
        self._config = config
        self._blocks: dict[bytes, KVBlock] = {}

    @property
    def config(self) -> StoreConfig:
        return self._config

    def put(self, key: bytes, block: KVBlock) -> None:
        validate_block(self._config, block)
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
