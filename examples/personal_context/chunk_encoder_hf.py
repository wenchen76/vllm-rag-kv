# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline chunk-KV encoder backed by HuggingFace transformers.

Produces ``KVBlock`` entries that ``vllm.v1.personal_context`` can store
and reuse via the PersonalContext KV connector. The encoder runs the
target model on a single chunk (no surrounding context) and harvests
``past_key_values``, slices them per block, and permutes to PC's NHD
layout.

Why HF (not vLLM) for offline encoding:

    - vLLM's paged-cache K/V is not directly exposed for offline
      capture; extracting it would need a hook into the worker.
    - HF runs the same mathematical RoPE / attention; numerical drift
      vs vLLM is at the fp16 ULP level for standard-RoPE models
      (Llama-3, Llama-3.2, Qwen2.5 — anything without ``rope_scaling``).
    - For the standard RAG pattern (chunks at prefix positions, query
      suffix recomputed), drift is irrelevant: select-all sparse-Q
      overwrites chunk K/V before attention reads it. The drift only
      manifests under R < 1.0 partial selection, where it appears as
      a small quality floor next to the baseline.

Iteration model is Llama-3.2-1B-Instruct (small, fast, same
architecture family as Llama-3-8B-Instruct). Switching the production
target to 3-8B requires only changing ``MODEL_ID`` and the
``StoreConfig`` (num_layers, head_dim, num_kv_heads); the encoder
logic is unchanged.

Usage:
    python examples/personal_context/chunk_encoder_hf.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from vllm.v1.personal_context import (
    Chunk,
    InMemoryStorage,
    KVBlock,
    StoreConfig,
)


# ----------------------------- model presets -----------------------------


@dataclass(frozen=True)
class ModelPreset:
    """Architecture constants needed to construct a matching ``StoreConfig``.

    The values are duplicated here (not pulled from ``AutoConfig``)
    so the script can assert HF output shapes match what PC expects
    without an extra ``hf_config`` round-trip; mismatches surface as
    plain ``AssertionError`` rather than confusing tensor-shape errors
    deeper in the pipeline.
    """

    hf_id: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float


LLAMA_3_2_1B_INSTRUCT = ModelPreset(
    hf_id="meta-llama/Llama-3.2-1B-Instruct",
    num_layers=16,
    num_kv_heads=8,
    head_dim=64,
    rope_theta=500000.0,
)

LLAMA_3_8B_INSTRUCT = ModelPreset(
    hf_id="meta-llama/Meta-Llama-3-8B-Instruct",
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    rope_theta=500000.0,
)


def store_config_for(preset: ModelPreset, block_size: int = 16) -> StoreConfig:
    return StoreConfig(
        model_id=preset.hf_id,
        dtype=torch.float16,
        layout="NHD",
        num_layers=preset.num_layers,
        num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim,
        block_size=block_size,
    )


# ----------------------------- encoder -----------------------------


class HFChunkEncoder:
    """Encode chunk text → list of ``(block_hash, KVBlock)`` via HF forward.

    One instance holds one HF model. Reuse across many chunks.
    """

    def __init__(
        self,
        preset: ModelPreset,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "HFChunkEncoder requires `transformers`. "
                "Install via `uv pip install transformers`."
            ) from e

        self.preset = preset
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(preset.hf_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(preset.hf_id, dtype=dtype)
            .to(device)
            .eval()
        )

        # Cross-check the HF config matches the preset; catches the case
        # where the preset is stale or the HF checkpoint published a
        # variant with different num_kv_heads / head_dim.
        hf_cfg = self.model.config
        assert hf_cfg.num_hidden_layers == preset.num_layers, (
            f"preset says {preset.num_layers} layers, HF reports "
            f"{hf_cfg.num_hidden_layers}"
        )
        assert hf_cfg.num_key_value_heads == preset.num_kv_heads, (
            f"preset says {preset.num_kv_heads} KV heads, HF reports "
            f"{hf_cfg.num_key_value_heads}"
        )
        expected_head_dim = hf_cfg.hidden_size // hf_cfg.num_attention_heads
        assert expected_head_dim == preset.head_dim, (
            f"preset says head_dim={preset.head_dim}, HF reports "
            f"{expected_head_dim}"
        )
        assert float(hf_cfg.rope_theta) == preset.rope_theta, (
            f"preset says rope_theta={preset.rope_theta}, HF reports "
            f"{hf_cfg.rope_theta}"
        )
        # Loud refusal for Llama-3.1's NTK rope_scaling — PC's
        # apply_delta_rope does not implement that yet, so delta-RoPE
        # rotation would silently mis-rotate. Plain Llama-3 / Llama-3.2
        # have rope_scaling=None.
        if getattr(hf_cfg, "rope_scaling", None) is not None:
            raise NotImplementedError(
                f"Model {preset.hf_id} declares rope_scaling="
                f"{hf_cfg.rope_scaling}. PC's apply_delta_rope only "
                "supports standard RoPE; using this model would produce "
                "wrong K rotation on reuse. Pick a no-scaling variant "
                "(Llama-3, Llama-3.2, Qwen2.5)."
            )

    def encode_chunk(
        self,
        token_ids: list[int],
        old_pos_start: int,
        block_size: int = 16,
    ) -> tuple[Chunk, list[tuple[bytes, KVBlock]]]:
        """Encode one chunk and return Chunk + per-block (hash, KVBlock).

        Args:
            token_ids: Chunk tokens. Length must be a multiple of
                ``block_size``; the caller is responsible for chunking
                / padding upstream.
            old_pos_start: Absolute position the chunk's first token
                sits at during this encoding pass. Stored alongside
                each ``KVBlock`` so PC's ``apply_delta_rope`` can
                compute the rotation delta when the chunk is reused at
                a different position. ``0`` is the standard choice for
                context-free chunks; the value only has to be
                consistent between encoding and storage.
            block_size: Tokens per paged block. Must match the vLLM
                instance and the ``StoreConfig`` block_size.

        Returns:
            ``(chunk, entries)`` where:
              - ``chunk`` is the ``Chunk`` (token_ids, old_pos_start).
              - ``entries`` is a list of ``(block_hash, KVBlock)``
                ready for ``InMemoryStorage.put(hash, block)``.
        """
        n = len(token_ids)
        if n == 0 or n % block_size != 0:
            raise ValueError(
                f"chunk length {n} must be a positive multiple of "
                f"block_size {block_size}"
            )

        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )
        position_ids = torch.arange(
            old_pos_start,
            old_pos_start + n,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True,
            )

        layers = _extract_past_kv_layers(outputs.past_key_values)
        if len(layers) != self.preset.num_layers:
            raise RuntimeError(
                f"HF returned {len(layers)} layers of past KV, expected "
                f"{self.preset.num_layers}"
            )

        chunk = Chunk(
            token_ids=tuple(token_ids), old_pos_start=old_pos_start
        )
        hashes = list(chunk.block_hashes(block_size))
        num_blocks = n // block_size
        assert len(hashes) == num_blocks

        entries: list[tuple[bytes, KVBlock]] = []
        for i, block_hash in enumerate(hashes):
            block_start = i * block_size
            block_end = block_start + block_size
            keys_per_layer: list[torch.Tensor] = []
            values_per_layer: list[torch.Tensor] = []
            for layer_idx, (K, V) in enumerate(layers):
                # HF shape: [batch=1, num_kv_heads, seq, head_dim]
                # → slice seq → permute to NHD: [seq_slice, num_kv_heads, head_dim]
                k_slice = (
                    K[0, :, block_start:block_end, :]
                    .permute(1, 0, 2)
                    .contiguous()
                    .to(self.dtype)
                    .cpu()
                )
                v_slice = (
                    V[0, :, block_start:block_end, :]
                    .permute(1, 0, 2)
                    .contiguous()
                    .to(self.dtype)
                    .cpu()
                )
                _check_kv_shape(k_slice, layer_idx, "K", block_size, self.preset)
                _check_kv_shape(v_slice, layer_idx, "V", block_size, self.preset)
                keys_per_layer.append(k_slice)
                values_per_layer.append(v_slice)

            entries.append(
                (
                    block_hash,
                    KVBlock(
                        keys=keys_per_layer,
                        values=values_per_layer,
                        old_pos_start=old_pos_start + block_start,
                    ),
                )
            )
        return chunk, entries

    def encode_text(
        self,
        text: str,
        old_pos_start: int = 0,
        block_size: int = 16,
    ) -> tuple[Chunk, list[tuple[bytes, KVBlock]]]:
        """Convenience: tokenize ``text`` then call ``encode_chunk``.

        Pads (right-truncates) to the nearest multiple of ``block_size``
        rather than raising — keeps the demo path forgiving. For
        production you almost certainly want to chunk upstream so
        every chunk is exactly N * block_size tokens.
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        # Truncate to multiple of block_size; warn if we drop tokens.
        truncated = (len(token_ids) // block_size) * block_size
        if truncated == 0:
            raise ValueError(
                f"text tokenizes to {len(token_ids)} tokens, less than "
                f"block_size {block_size}; provide a longer chunk."
            )
        if truncated != len(token_ids):
            dropped = len(token_ids) - truncated
            print(
                f"[encode_text] dropped {dropped} trailing tokens to align "
                f"to block_size={block_size}"
            )
            token_ids = token_ids[:truncated]
        return self.encode_chunk(token_ids, old_pos_start, block_size)


# ----------------------------- internals -----------------------------


def _extract_past_kv_layers(past_kv) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Normalise transformers' evolving past_kv API to a flat list.

    Newer transformers returns ``DynamicCache`` with ``key_cache`` /
    ``value_cache`` lists; older returns ``tuple[layer] of (K, V)``.
    """
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        return list(zip(past_kv.key_cache, past_kv.value_cache))
    return [(k, v) for k, v in past_kv]


def _check_kv_shape(
    t: torch.Tensor,
    layer_idx: int,
    name: str,
    block_size: int,
    preset: ModelPreset,
) -> None:
    expected = (block_size, preset.num_kv_heads, preset.head_dim)
    if tuple(t.shape) != expected:
        raise RuntimeError(
            f"layer {layer_idx} {name}: expected shape {expected}, got "
            f"{tuple(t.shape)}"
        )


def populate_storage(
    storage: InMemoryStorage,
    entries: Iterable[tuple[bytes, KVBlock]],
) -> int:
    """Write every (hash, KVBlock) into storage. Returns count written."""
    count = 0
    for block_hash, kv_block in entries:
        storage.put(block_hash, kv_block)
        count += 1
    return count


# ----------------------------- __main__ demo -----------------------------


def _demo() -> None:
    """Encode one short chunk and verify it round-trips through storage."""
    preset = LLAMA_3_2_1B_INSTRUCT
    block_size = 16
    cfg = store_config_for(preset, block_size=block_size)

    print(f"Loading {preset.hf_id} (this may take a minute on first run)...")
    encoder = HFChunkEncoder(preset=preset, device="cuda", dtype=torch.float16)

    # 32 tokens = 2 blocks at block_size=16. Tokens are arbitrary integers
    # inside the model's vocab range; for production you'd encode real
    # document text.
    chunk_text = (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
        "Mars in Paris. It is named after the engineer Gustave Eiffel."
    )
    chunk, entries = encoder.encode_text(
        chunk_text, old_pos_start=0, block_size=block_size
    )
    print(
        f"Encoded {len(chunk.token_ids)} tokens into {len(entries)} blocks."
    )

    # Round-trip through storage.
    storage = InMemoryStorage(cfg)
    n_written = populate_storage(storage, entries)
    print(f"Wrote {n_written} blocks to InMemoryStorage.")

    # Sanity: every block hash returns the same KVBlock we put in.
    for block_hash, original in entries:
        roundtrip = storage.get(block_hash)
        assert roundtrip is not None, "store miss after put"
        assert len(roundtrip.keys) == len(original.keys) == preset.num_layers
        for k_orig, k_back in zip(original.keys, roundtrip.keys):
            torch.testing.assert_close(k_orig, k_back, rtol=0, atol=0)
    print(
        f"Round-trip verified: {n_written} blocks × {preset.num_layers} layers "
        f"× ({block_size}, {preset.num_kv_heads}, {preset.head_dim}) fp16."
    )


if __name__ == "__main__":
    _demo()
