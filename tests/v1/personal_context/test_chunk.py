# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.personal_context import (
    AlignmentError,
    Chunk,
    hash_block,
    validate_chunk_alignment,
)

BLOCK_SIZE = 4


def test_aligned_chunk_block_hashes_match_independent_hash_block():
    tokens = tuple(range(BLOCK_SIZE * 3))
    chunk = Chunk(token_ids=tokens, old_pos_start=BLOCK_SIZE * 2)
    hashes = chunk.block_hashes(BLOCK_SIZE)
    assert len(hashes) == 3
    for i, h in enumerate(hashes):
        expected = hash_block(tokens[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE])
        assert h == expected


def test_single_block_chunk():
    tokens = tuple(range(BLOCK_SIZE))
    chunk = Chunk(token_ids=tokens, old_pos_start=0)
    hashes = chunk.block_hashes(BLOCK_SIZE)
    assert hashes == [hash_block(tokens)]


def test_salt_propagates_to_each_block_hash():
    tokens = tuple(range(BLOCK_SIZE * 2))
    salt = b"tenant-a"
    chunk = Chunk(token_ids=tokens, old_pos_start=0, salt=salt)
    hashes = chunk.block_hashes(BLOCK_SIZE)
    unsalted = Chunk(token_ids=tokens, old_pos_start=0).block_hashes(BLOCK_SIZE)
    assert hashes != unsalted
    for i, h in enumerate(hashes):
        assert h == hash_block(
            tokens[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE], salt=salt
        )


def test_misaligned_start_pos_raises():
    tokens = tuple(range(BLOCK_SIZE * 2))
    chunk = Chunk(token_ids=tokens, old_pos_start=BLOCK_SIZE + 1)
    with pytest.raises(AlignmentError, match="start_pos"):
        chunk.block_hashes(BLOCK_SIZE)


def test_misaligned_length_raises():
    tokens = tuple(range(BLOCK_SIZE + 1))
    chunk = Chunk(token_ids=tokens, old_pos_start=0)
    with pytest.raises(AlignmentError, match="length"):
        chunk.block_hashes(BLOCK_SIZE)


def test_zero_length_raises():
    chunk = Chunk(token_ids=(), old_pos_start=0)
    with pytest.raises(AlignmentError, match="length"):
        chunk.block_hashes(BLOCK_SIZE)


def test_negative_start_pos_raises():
    tokens = tuple(range(BLOCK_SIZE))
    chunk = Chunk(token_ids=tokens, old_pos_start=-BLOCK_SIZE)
    with pytest.raises(AlignmentError, match="start_pos"):
        chunk.block_hashes(BLOCK_SIZE)


def test_non_positive_block_size_raises():
    with pytest.raises(AlignmentError, match="block_size"):
        validate_chunk_alignment(start_pos=0, length=4, block_size=0)
    with pytest.raises(AlignmentError, match="block_size"):
        validate_chunk_alignment(start_pos=0, length=4, block_size=-1)


def test_validate_chunk_alignment_accepts_aligned():
    validate_chunk_alignment(start_pos=0, length=BLOCK_SIZE, block_size=BLOCK_SIZE)
    validate_chunk_alignment(
        start_pos=BLOCK_SIZE * 7, length=BLOCK_SIZE * 3, block_size=BLOCK_SIZE
    )


def test_alignment_error_is_value_error():
    assert issubclass(AlignmentError, ValueError)
