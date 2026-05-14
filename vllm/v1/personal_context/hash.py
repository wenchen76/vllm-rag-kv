# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Position-independent content hashing for retrieved KV blocks.

Distinct schema from vLLM's prefix-rolling block hash: this hash depends
ONLY on the token IDs inside a block, not on any preceding context.
Same chunk content at different prompt positions yields the same hash.

The two hash schemas MUST NOT share a cache table - see invariant locked
in Phase 3.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

# Domain separator: prevents accidental collision with any other vLLM hash schema.
_DOMAIN = b"vllm.personal_context.block.v1\x00"


def hash_block(token_ids: Sequence[int], salt: bytes = b"") -> bytes:
    """Position-independent block hash.

    Args:
        token_ids: ordered token IDs for this block. The caller is
            responsible for block alignment (len == store.block_size).
        salt: optional tenant / namespace salt.

    Returns:
        32-byte SHA-256 digest.
    """
    h = hashlib.sha256()
    h.update(_DOMAIN)
    h.update(len(salt).to_bytes(2, "big"))
    h.update(salt)
    h.update(len(token_ids).to_bytes(4, "big"))
    for tid in token_ids:
        h.update(int(tid).to_bytes(4, "big", signed=False))
    return h.digest()
