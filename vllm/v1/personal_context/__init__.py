# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Personal-context KV store for RAG retrieved-chunk reuse."""

from vllm.v1.personal_context.chunk import (
    AlignmentError,
    Chunk,
    validate_chunk_alignment,
)
from vllm.v1.personal_context.entry import KVBlock, StoreConfig
from vllm.v1.personal_context.hash import hash_block
from vllm.v1.personal_context.policy import (
    PrefixCachePollutionError,
    ReusePlan,
    assert_can_write_prefix_cache,
)
from vllm.v1.personal_context.storage import InMemoryStorage

__all__ = [
    "AlignmentError",
    "Chunk",
    "InMemoryStorage",
    "KVBlock",
    "PrefixCachePollutionError",
    "ReusePlan",
    "StoreConfig",
    "assert_can_write_prefix_cache",
    "hash_block",
    "validate_chunk_alignment",
]
