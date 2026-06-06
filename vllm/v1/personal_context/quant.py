# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Int8 quantization for stored chunk KV.

Stored KV can be quantized to int8 to shrink the store (~2x vs fp16/bf16).
Quantization is **per-tensor symmetric absmax**: one float32 scale per
``[block_size, num_kv_heads, head_dim]`` tensor, ``scale = absmax / 127`` and
``q = round(x / scale)`` clamped to ``[-127, 127]``.

The store keeps int8 + scales; ``load_plan`` dequantizes back to the model's
compute dtype on GPU right before delta-RoPE (which needs float). Per-tensor is
the simplest granularity (1 scale/tensor) and is what the mmap record layout and
``KVBlock.k_scales``/``v_scales`` assume.
"""

from __future__ import annotations

import torch

QUANT_NONE = "none"
QUANT_INT8 = "int8"
VALID_QUANT = (QUANT_NONE, QUANT_INT8)


def quantize_int8(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor symmetric int8. Returns ``(int8_tensor, scale)``.

    ``scale`` is a Python float so it serialises trivially (pickle / JSON /
    a float32 in the mmap record). ``t`` should be a CPU float tensor.
    """
    amax = float(t.detach().abs().max())
    scale = amax / 127.0 if amax > 0.0 else 1.0  # all-zero tensor -> scale 1
    q = (
        torch.round(t.detach().to(torch.float32) / scale)
        .clamp_(-127, 127)
        .to(torch.int8)
    )
    return q, scale


def dequantize_int8(
    q: torch.Tensor, scale: float, dtype: torch.dtype
) -> torch.Tensor:
    """Inverse of ``quantize_int8`` into ``dtype`` (the model compute dtype)."""
    return (q.to(torch.float32) * scale).to(dtype)
