# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Personal-context KV store for RAG retrieved-chunk reuse.

Phase 1 deliverable: in-memory storage stub + content hash + entry schema.
Phase 3 wires this into vLLM via an ``OffloadingManager`` subclass.
"""

from vllm.v1.personal_context.entry import KVBlock, StoreConfig
from vllm.v1.personal_context.hash import hash_block
from vllm.v1.personal_context.storage import InMemoryStorage

__all__ = [
    "InMemoryStorage",
    "KVBlock",
    "StoreConfig",
    "hash_block",
]
