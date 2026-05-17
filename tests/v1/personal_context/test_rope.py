# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.personal_context import (
    apply_delta_rope,
    apply_rope_at_positions,
)


BLOCK_SIZE = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8


def _random_keys(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(
        BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, generator=g, dtype=torch.float32
    ).to(dtype)


def test_delta_zero_is_identity():
    k = _random_keys()
    out = apply_delta_rope(k, delta=0)
    torch.testing.assert_close(out, k)


def test_composition_law_matches_full_rope_at_new_position():
    """RoPE(P) then R(delta) == RoPE(P + delta), elementwise."""
    k0 = _random_keys()
    p_old = 16
    delta = 32

    old_positions = torch.arange(p_old, p_old + BLOCK_SIZE, dtype=torch.float32)
    new_positions = old_positions + delta

    rotated_at_old = apply_rope_at_positions(k0, old_positions)
    rotated_then_delta = apply_delta_rope(rotated_at_old, delta=delta)
    rotated_at_new = apply_rope_at_positions(k0, new_positions)

    torch.testing.assert_close(
        rotated_then_delta, rotated_at_new, rtol=1e-5, atol=1e-5
    )


def test_composition_law_with_negative_delta():
    k0 = _random_keys()
    p_old = 64
    delta = -32  # rewind

    old_positions = torch.arange(p_old, p_old + BLOCK_SIZE, dtype=torch.float32)
    new_positions = old_positions + delta

    rotated_at_old = apply_rope_at_positions(k0, old_positions)
    rotated_then_delta = apply_delta_rope(rotated_at_old, delta=delta)
    rotated_at_new = apply_rope_at_positions(k0, new_positions)

    torch.testing.assert_close(
        rotated_then_delta, rotated_at_new, rtol=1e-5, atol=1e-5
    )


def test_shape_and_dtype_preserved_fp16():
    k = _random_keys(dtype=torch.float16)
    out = apply_delta_rope(k, delta=8)
    assert out.shape == k.shape
    assert out.dtype == torch.float16


def test_shape_and_dtype_preserved_bf16():
    k = _random_keys(dtype=torch.bfloat16)
    out = apply_delta_rope(k, delta=8)
    assert out.shape == k.shape
    assert out.dtype == torch.bfloat16


def test_odd_head_dim_rejected():
    bad = torch.zeros(BLOCK_SIZE, NUM_KV_HEADS, 7)
    with pytest.raises(ValueError, match="head_dim"):
        apply_delta_rope(bad, delta=4)


def test_positions_must_be_1d():
    k = _random_keys()
    bad_positions = torch.zeros(BLOCK_SIZE, 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="1-D"):
        apply_rope_at_positions(k, bad_positions)


def test_positions_length_must_match():
    k = _random_keys()
    bad_positions = torch.zeros(BLOCK_SIZE + 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="length"):
        apply_rope_at_positions(k, bad_positions)


def test_determinism():
    k = _random_keys()
    a = apply_delta_rope(k, delta=17)
    b = apply_delta_rope(k, delta=17)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_input_not_mutated():
    k = _random_keys()
    k_copy = k.clone()
    _ = apply_delta_rope(k, delta=11)
    torch.testing.assert_close(k, k_copy, rtol=0, atol=0)


def test_rope_theta_is_wired_through():
    k = _random_keys()
    positions = torch.arange(BLOCK_SIZE, dtype=torch.float32)
    out_default = apply_rope_at_positions(k, positions, rope_theta=10000.0)
    out_other = apply_rope_at_positions(k, positions, rope_theta=500000.0)
    # Different bases must produce different rotations for non-zero positions.
    assert not torch.allclose(out_default, out_other)


def test_apply_rope_at_positions_broadcasts_over_heads():
    """cos/sin must broadcast across the head axis (single position acts identically per head)."""
    head_dim = HEAD_DIM
    k = torch.randn(BLOCK_SIZE, NUM_KV_HEADS, head_dim)
    positions = torch.arange(BLOCK_SIZE, dtype=torch.float32)
    out = apply_rope_at_positions(k, positions)
    # Per-head outputs computed independently must match the broadcast result.
    per_head = torch.stack(
        [
            apply_rope_at_positions(
                k[:, h : h + 1, :], positions
            ).squeeze(1)
            for h in range(NUM_KV_HEADS)
        ],
        dim=1,
    )
    torch.testing.assert_close(out, per_head, rtol=1e-6, atol=1e-6)
