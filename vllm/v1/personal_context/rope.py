# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Delta RoPE primitive for personal-context KV reuse.

K is stored post-RoPE at the chunk's *encoding* position
``old_pos_start``. To reuse it at a different *runtime* position
``new_pos_start`` we apply

    R(new_pos - old_pos) = R(delta)

elementwise. Because ``R(p_old) @ R(delta) = R(p_new)`` and ``delta``
is the same for every token inside a chunk (every block shifts by the
same amount), one rotation per layer per block is enough.

This file is the **reference** implementation: eager-mode PyTorch,
Llama-style rotate-half only, no inv-freq cache, internal compute in
fp32. A fused kernel can replace it later behind the same signature.
"""

from __future__ import annotations

import torch


def _compute_inv_freq(
    head_dim: int, rope_theta: float, device: torch.device
) -> torch.Tensor:
    """Standard RoPE inverse-frequency table in fp32. Shape [head_dim // 2]."""
    exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    return 1.0 / (rope_theta ** (exponent / head_dim))


def _compute_cos_sin(
    positions: torch.Tensor, inv_freq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) of shape [N, head_dim], duplicating the half table."""
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Llama-style: ``(x1, x2) -> (-x2, x1)`` on the last dim halves."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_at_positions(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Apply RoPE rotation to ``x`` at the given absolute ``positions``.

    Args:
        x: Tensor with shape ``[N, ..., head_dim]``. ``head_dim`` must be
            even. Last dim is rotated; leading dim ``N`` is one position
            per row; any middle dims (e.g. heads) are broadcast over.
        positions: Long/float tensor of shape ``[N]``.
        rope_theta: RoPE base. Default 10000.0.

    Returns:
        Rotated tensor with the same shape and dtype as ``x``.
        Input is not mutated.
    """
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if positions.dim() != 1:
        raise ValueError(
            f"positions must be 1-D, got shape {tuple(positions.shape)}"
        )
    n = x.shape[0]
    if positions.shape[0] != n:
        raise ValueError(
            f"positions length {positions.shape[0]} != x.shape[0] {n}"
        )

    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    inv_freq = _compute_inv_freq(head_dim, rope_theta, x.device)
    cos, sin = _compute_cos_sin(positions.to(torch.float32), inv_freq)
    # cos/sin: [N, head_dim]; need to broadcast over any middle dims of x.
    for _ in range(x_fp32.dim() - 2):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    rotated = x_fp32 * cos + _rotate_half(x_fp32) * sin
    return rotated.to(orig_dtype)


def apply_delta_rope(
    keys: torch.Tensor,
    delta: int,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Rotate one block of K by a constant ``delta`` along position axis.

    ``keys`` has shape ``[block_size, num_kv_heads, head_dim]`` and is
    interpreted as already-rotated at some ``old_pos_start``. The result
    is the same K rotated at ``old_pos_start + delta``.

    ``delta`` may be zero or negative. ``delta == 0`` is the identity
    (return value still equals input numerically; a fresh tensor is
    returned, not the original).
    """
    block_size = keys.shape[0]
    positions = torch.full(
        (block_size,), delta, dtype=torch.float32, device=keys.device
    )
    return apply_rope_at_positions(keys, positions, rope_theta=rope_theta)


def apply_delta_rope_batched(
    keys: torch.Tensor,
    delta: int,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Batched ``apply_delta_rope`` over a leading stack dimension.

    ``keys`` has shape ``[M, block_size, num_kv_heads, head_dim]`` — a
    stack of ``M`` blocks (e.g. every (block, layer) pair of one chunk)
    that all shift by the SAME ``delta``. Because the rotation depends
    only on ``delta`` (constant within a chunk), the cos/sin table is
    computed ONCE and broadcast across the whole stack, replacing ``M``
    separate ``apply_delta_rope`` calls — each of which otherwise
    recomputes the identical inv_freq + cos/sin and launches its own
    kernels. This is the hot path in ``load_plan`` for long contexts:
    14 chunks x ~4 blocks x 32 layers is ~1800 calls collapsed to a
    handful.

    Numerically identical to calling ``apply_delta_rope`` on each
    ``keys[i]`` (same fp32 compute, same cos/sin, same rotate-half).

    Returns a tensor of the same shape and dtype as ``keys``; input is
    not mutated. ``M == 0`` returns an empty tensor of the right shape.
    """
    if keys.dim() != 4:
        raise ValueError(
            f"keys must be 4-D [M, block_size, num_kv_heads, head_dim], "
            f"got shape {tuple(keys.shape)}"
        )
    head_dim = keys.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if keys.shape[0] == 0:
        return keys.clone()

    block_size = keys.shape[1]
    orig_dtype = keys.dtype
    x_fp32 = keys.to(torch.float32)
    inv_freq = _compute_inv_freq(head_dim, rope_theta, keys.device)
    positions = torch.full(
        (block_size,), delta, dtype=torch.float32, device=keys.device
    )
    cos, sin = _compute_cos_sin(positions, inv_freq)  # [block_size, head_dim]
    # Broadcast cos/sin over the leading stack dim (M) and the head dim:
    # x_fp32 is [M, block_size, num_kv_heads, head_dim], cos/sin become
    # [1, block_size, 1, head_dim].
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    rotated = x_fp32 * cos + _rotate_half(x_fp32) * sin
    return rotated.to(orig_dtype)
