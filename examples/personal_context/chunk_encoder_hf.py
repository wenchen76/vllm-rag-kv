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


QWEN_2_5_0_5B_INSTRUCT = ModelPreset(
    hf_id="Qwen/Qwen2.5-0.5B-Instruct",
    num_layers=24,
    num_kv_heads=2,  # GQA: 14 attn heads ÷ 7
    head_dim=64,
    rope_theta=1000000.0,
)

QWEN_2_5_1_5B_INSTRUCT = ModelPreset(
    hf_id="Qwen/Qwen2.5-1.5B-Instruct",
    num_layers=28,
    num_kv_heads=2,  # GQA: 12 attn heads ÷ 6
    head_dim=128,
    rope_theta=1000000.0,
)

LLAMA_3_8B_INSTRUCT = ModelPreset(
    hf_id="meta-llama/Meta-Llama-3-8B-Instruct",
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    rope_theta=500000.0,
)

# NOTE: Meta-Llama-3.1 / 3.2 series are intentionally NOT listed here.
# They ship with ``rope_scaling = {"rope_type": "llama3", ...}`` to
# extend context to 128K via NTK-style piecewise frequency rescaling.
# PC's ``apply_delta_rope`` only implements standard RoPE; using a
# scaled-RoPE model would silently mis-rotate K on delta != 0 and
# corrupt attention. Re-add those presets only after implementing
# the scaling correction in ``vllm/v1/personal_context/rope.py``.


# Registry of presets keyed by HF id. Demo / CLI tools look up by id
# rather than passing the dataclass directly so a single ``--model``
# string flag works end-to-end.
KNOWN_PRESETS: dict[str, ModelPreset] = {
    p.hf_id: p
    for p in (
        QWEN_2_5_0_5B_INSTRUCT,
        QWEN_2_5_1_5B_INSTRUCT,
        LLAMA_3_8B_INSTRUCT,
    )
}


def preset_for(hf_id: str) -> ModelPreset:
    """Look up a ``ModelPreset`` by its HF id.

    Raises ``KeyError`` with a helpful list of registered ids when the
    requested model has no preset — register a new ``ModelPreset`` in
    this file rather than removing this guard, so the constructor's
    architecture-vs-preset cross-checks still run.
    """
    try:
        return KNOWN_PRESETS[hf_id]
    except KeyError as e:
        known = "\n  ".join(sorted(KNOWN_PRESETS.keys()))
        raise KeyError(
            f"No ModelPreset registered for {hf_id!r}.\n"
            f"Known presets:\n  {known}\n"
            f"To add a new one, declare a ModelPreset in "
            f"chunk_encoder_hf.py and append it to KNOWN_PRESETS."
        ) from e


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
        lazy_model: bool = False,
    ):
        """
        Args:
            preset: Architecture / model id descriptor.
            device: Where the model lives (``cuda`` / ``cpu``).
            dtype: Model dtype; should match the StoreConfig.
            lazy_model: If True, skip loading the model until the first
                ``encode_chunk`` call (or explicit ``.model`` access).
                The tokenizer always loads up-front since it's tiny and
                callers usually need it for hash precomputation. Use
                this when ingestion may hit a fully-warm cache and the
                heavy model load can be avoided entirely.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "HFChunkEncoder requires `transformers`. "
                "Install via `uv pip install transformers`."
            ) from e

        self.preset = preset
        self.device = device
        self.dtype = dtype
        # Tokenizer is always loaded (small, ~10MB cached) — callers
        # need it to compute block hashes for cache hit checks before
        # deciding whether to pay the model load cost.
        self.tokenizer = AutoTokenizer.from_pretrained(preset.hf_id)
        self._model = None
        if not lazy_model:
            self._ensure_model()

    @property
    def model(self):
        """Accessing ``.model`` triggers lazy load if needed."""
        if self._model is None:
            self._ensure_model()
        return self._model

    @property
    def is_model_loaded(self) -> bool:
        """Whether the heavy HF model is actually in memory yet."""
        return self._model is not None

    def _ensure_model(self) -> None:
        """Idempotent: load + arch-check the HF model on first call.

        Architecture checks (layer count, head dim, rope_theta,
        rope_type) intentionally live here rather than ``__init__``
        because they all read ``self.model.config``. Lazy mode never
        runs them until the model is actually needed.
        """
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "HFChunkEncoder requires `transformers`. "
                "Install via `uv pip install transformers`."
            ) from e

        print(f"[encoder] loading model {self.preset.hf_id} (lazy)")
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self.preset.hf_id, dtype=self.dtype
            )
            .to(self.device)
            .eval()
        )

        # Cross-check the HF config matches the preset; catches the
        # case where the preset is stale or the HF checkpoint
        # published a variant with different num_kv_heads / head_dim.
        hf_cfg = self._model.config
        assert hf_cfg.num_hidden_layers == self.preset.num_layers, (
            f"preset says {self.preset.num_layers} layers, HF reports "
            f"{hf_cfg.num_hidden_layers}"
        )
        assert hf_cfg.num_key_value_heads == self.preset.num_kv_heads, (
            f"preset says {self.preset.num_kv_heads} KV heads, HF reports "
            f"{hf_cfg.num_key_value_heads}"
        )
        expected_head_dim = hf_cfg.hidden_size // hf_cfg.num_attention_heads
        assert expected_head_dim == self.preset.head_dim, (
            f"preset says head_dim={self.preset.head_dim}, HF reports "
            f"{expected_head_dim}"
        )
        hf_rope_theta = _read_rope_theta(hf_cfg)
        assert hf_rope_theta == self.preset.rope_theta, (
            f"preset says rope_theta={self.preset.rope_theta}, HF reports "
            f"{hf_rope_theta}"
        )
        # Loud refusal for any non-default rope_type — PC's
        # apply_delta_rope only implements standard RoPE, so any
        # scaled variant (NTK piecewise, linear interpolation, YaRN,
        # dynamic NTK) would silently mis-rotate K on reuse.
        #
        # Newer transformers (~4.50+) ALWAYS populates rope_scaling
        # even for standard-RoPE models, with ``rope_type="default"``
        # as the "no actual scaling" sentinel. Old transformers leaves
        # rope_scaling=None for the same case. Both shapes are safe.
        rope_scaling = getattr(hf_cfg, "rope_scaling", None)
        rope_type = (
            rope_scaling.get("rope_type")
            if isinstance(rope_scaling, dict)
            else None
        )
        if rope_type not in (None, "default"):
            raise NotImplementedError(
                f"Model {self.preset.hf_id} uses rope_type={rope_type!r} "
                f"(scaling={rope_scaling}). PC's apply_delta_rope only "
                "supports standard RoPE (rope_type='default' or None); "
                "any other scaling would silently mis-rotate K on reuse. "
                "Pick a no-scaling variant (Llama-3, Qwen2.5, TinyLlama)."
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

    Tried in order:
        1. Plain ``tuple[layer] of (K, V)`` — oldest format.
        2. ``DynamicCache`` with ``.key_cache`` / ``.value_cache`` lists.
        3. ``DynamicCache`` with ``.layers`` (each layer object has
           ``.keys`` / ``.values`` or ``.key`` / ``.value``) — newest.
        4. Iterable fall-through: each row's first two elements are K, V.

    Returns ``[(K, V), ...]`` one per layer, in layer order.
    """
    # 1. Plain tuple (old transformers)
    if isinstance(past_kv, tuple) and past_kv and isinstance(
        past_kv[0], tuple
    ):
        return [(k, v) for k, v in past_kv]

    # 2. DynamicCache w/ key_cache / value_cache parallel lists
    if (
        hasattr(past_kv, "key_cache")
        and hasattr(past_kv, "value_cache")
        and len(past_kv.key_cache) > 0
    ):
        return list(zip(past_kv.key_cache, past_kv.value_cache))

    # 3. DynamicCache w/ .layers (transformers ~4.55+)
    if hasattr(past_kv, "layers"):
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in past_kv.layers:
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                out.append((layer.keys, layer.values))
            elif hasattr(layer, "key") and hasattr(layer, "value"):
                out.append((layer.key, layer.value))
            else:
                raise RuntimeError(
                    f"unknown DynamicCache layer type: "
                    f"{type(layer).__name__}; attrs="
                    f"{[a for a in dir(layer) if not a.startswith('_')]}"
                )
        if out:
            return out

    # 4. Iterable, just take first two of each row
    try:
        return [(row[0], row[1]) for row in past_kv]
    except (TypeError, IndexError) as e:
        raise RuntimeError(
            f"could not extract per-layer K/V from past_kv of type "
            f"{type(past_kv).__name__}; checked tuple, .key_cache/"
            f".value_cache, .layers, iter"
        ) from e


def _read_rope_theta(hf_cfg) -> float:
    """Read ``rope_theta`` from a HF config across transformers versions.

    transformers used to expose ``rope_theta`` as a direct attribute on
    every model config that uses RoPE. ~4.50 consolidated RoPE settings
    into a ``rope_parameters`` dict on some configs; other variants nest
    it inside ``rope_scaling``. This helper checks all three layouts and
    returns the first hit.
    """
    val = getattr(hf_cfg, "rope_theta", None)
    if val is not None:
        return float(val)
    for attr in ("rope_parameters", "rope_scaling"):
        nested = getattr(hf_cfg, attr, None) or {}
        if isinstance(nested, dict) and "rope_theta" in nested:
            return float(nested["rope_theta"])
    raise AttributeError(
        f"could not find rope_theta on {type(hf_cfg).__name__}; "
        f"checked .rope_theta, .rope_parameters.rope_theta, "
        f".rope_scaling.rope_theta"
    )


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
    preset = QWEN_2_5_0_5B_INSTRUCT
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
