# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV block schema for the personal-context KV store."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StoreConfig:
    """Store-wide constants. KV reuse requires exact match on every field."""

    model_id: str
    dtype: torch.dtype
    layout: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    # "none" (store K/V as ``dtype``) or "int8" (store per-tensor int8 +
    # scales, dequantized to ``dtype`` in load_plan before delta-RoPE).
    # ``dtype`` always remains the COMPUTE dtype (the dequant target).
    quant: str = "none"


@dataclass
class KVBlock:
    """One block of post-RoPE K and V at the original encoding position.

    K is post-RoPE applied at positions
    ``[old_pos_start, old_pos_start + block_size)``. On load, a delta
    rotation ``R(new_pos - old_pos)`` is applied to K. V is
    position-independent and reused as-is.
    """

    # length == num_layers; each tensor shape [block_size, num_kv_heads, head_dim]
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    old_pos_start: int
    # When quantized (StoreConfig.quant=="int8"), keys/values are int8 and
    # these hold the per-tensor dequant scales (one float per layer); both
    # None for the unquantized fp16/bf16 case.
    k_scales: list[float] | None = None
    v_scales: list[float] | None = None
