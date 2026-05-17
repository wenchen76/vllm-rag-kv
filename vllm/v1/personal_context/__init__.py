# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Personal-context KV store for RAG retrieved-chunk reuse."""

from vllm.v1.personal_context.chunk import (
    AlignmentError,
    Chunk,
    validate_chunk_alignment,
)
from vllm.v1.personal_context.connector import (
    BlockLookup,
    ChunkLookup,
    PersonalContextConnector,
    PlanLookup,
)
from vllm.v1.personal_context.entry import KVBlock, StoreConfig
from vllm.v1.personal_context.hash import hash_block
from vllm.v1.personal_context.load import (
    LoadedBlock,
    LoadedChunk,
    LoadedPlan,
    load_plan,
)
from vllm.v1.personal_context.policy import (
    PrefixCachePollutionError,
    ReusePlan,
    assert_can_write_prefix_cache,
)
from vllm.v1.personal_context.prefill import (
    PrefillSetupSpec,
    build_chunk_aware_mask,
    build_paged_kv_metadata,
    run_chunk_aware_prefill,
    setup_chunk_aware_prefill,
)
from vllm.v1.personal_context.rope import (
    apply_delta_rope,
    apply_rope_at_positions,
)
from vllm.v1.personal_context.scatter import (
    scatter_loaded_block,
    scatter_loaded_plan,
)
from vllm.v1.personal_context.storage import InMemoryStorage

__all__ = [
    "AlignmentError",
    "BlockLookup",
    "Chunk",
    "ChunkLookup",
    "InMemoryStorage",
    "KVBlock",
    "LoadedBlock",
    "LoadedChunk",
    "LoadedPlan",
    "PersonalContextConnector",
    "PlanLookup",
    "PrefillSetupSpec",
    "PrefixCachePollutionError",
    "ReusePlan",
    "StoreConfig",
    "apply_delta_rope",
    "apply_rope_at_positions",
    "assert_can_write_prefix_cache",
    "build_chunk_aware_mask",
    "build_paged_kv_metadata",
    "hash_block",
    "load_plan",
    "run_chunk_aware_prefill",
    "scatter_loaded_block",
    "scatter_loaded_plan",
    "setup_chunk_aware_prefill",
    "validate_chunk_alignment",
]
