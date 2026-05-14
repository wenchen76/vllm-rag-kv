# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-alignment contract for retrieved chunks.

The store side does NOT pad mis-aligned chunks. Callers must guarantee:

    1. ``start_pos`` is a multiple of ``block_size``
    2. ``len(token_ids)`` is a positive multiple of ``block_size``

A chunk that fails these checks is rejected with ``AlignmentError`` at the
moment its block hashes are requested. The caller is expected to catch
``AlignmentError`` and fall back to treating the chunk as ordinary tokens
(full prefill). Silent padding would mask correctness regressions; loud
rejection is preferred.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.personal_context.hash import hash_block


class AlignmentError(ValueError):
    """A chunk does not satisfy the block-alignment contract."""


def validate_chunk_alignment(
    start_pos: int,
    length: int,
    block_size: int,
) -> None:
    """Raise ``AlignmentError`` if the chunk is not block-aligned."""
    if block_size <= 0:
        raise AlignmentError(f"block_size must be positive, got {block_size}")
    if length <= 0:
        raise AlignmentError(f"chunk length must be positive, got {length}")
    if start_pos < 0:
        raise AlignmentError(f"start_pos must be non-negative, got {start_pos}")
    if start_pos % block_size != 0:
        raise AlignmentError(
            f"start_pos {start_pos} is not a multiple of block_size {block_size}"
        )
    if length % block_size != 0:
        raise AlignmentError(
            f"length {length} is not a multiple of block_size {block_size}"
        )


@dataclass(frozen=True)
class Chunk:
    """A contiguous run of retrieved tokens with its source position.

    ``old_pos_start`` and ``len(token_ids)`` must both be multiples of
    the store's ``block_size``. Validation runs lazily inside
    ``block_hashes()``.
    """

    token_ids: tuple[int, ...]
    old_pos_start: int
    salt: bytes = b""

    def block_hashes(self, block_size: int) -> list[bytes]:
        """Split into block-sized content hashes, in order.

        Raises ``AlignmentError`` if alignment is violated.
        """
        validate_chunk_alignment(
            self.old_pos_start, len(self.token_ids), block_size
        )
        return [
            hash_block(self.token_ids[i : i + block_size], salt=self.salt)
            for i in range(0, len(self.token_ids), block_size)
        ]
