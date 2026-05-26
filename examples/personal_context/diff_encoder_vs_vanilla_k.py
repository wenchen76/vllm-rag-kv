#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diff K computed by HFChunkEncoder vs K computed by vanilla vLLM.

Goal: localize the r=0.9 catastrophic breakdown observed in the prefill
bench (output collapses into number sequences even with only ~10% stale
K/V in the cache). Previous HF-only run showed encoder K aligns with
vanilla HF K within FP16 rounding at layer 0 and stays cos > 0.9 at the
top layer, which doesn't explain the breakdown — so we widen the
comparison to the K that vLLM **actually** writes into the paged cache
at inference time.

Three K sources are compared:

  - **encoder**: ``HFChunkEncoder.encode_chunk`` → ``load_plan``
    (delta-RoPE'd to the online position). Same path the PC store
    populates from at bench time.
  - **vanilla HF**: a fresh HF ``Qwen2ForCausalLM.forward`` over
    ``sys + chunk``; ``past_key_values`` sliced at chunk positions.
  - **vanilla vLLM**: a fresh vLLM ``LLM.generate`` over the same
    ``sys + chunk`` prompt, with ``VLLM_DIFF_DUMP_K=/path`` so
    ``GpuModelRunner._maybe_diff_dump_kv`` pickles the first N physical
    blocks of every layer's K (and V) after the forward. Block IDs are
    deterministic for a cold-start single-request prefill: vLLM v1
    reserves block 0 (null) and allocates 1..total_blocks contiguously,
    so the chunk's blocks live at ``sys_len/block_size + 1`` onward.

Layer 0 K depends only on token embedding + RoPE (no attention), so all
three should agree to FP16 ULP there. Higher-layer drift between
encoder and vanilla is the CacheBlend approximation; any vLLM-vs-HF
divergence at higher layers would isolate a vLLM kernel-side issue.

Usage:
    .venv/bin/python examples/personal_context/diff_encoder_vs_vanilla_k.py

Env overrides:
    DIFF_MODEL=Qwen/Qwen2.5-1.5B-Instruct   (default)
    DIFF_BLOCK_SIZE=16                       (default)
    DIFF_SKIP_HF=1                           (skip HF arm, encoder vs vLLM only)
    DIFF_VLLM_DUMP=/tmp/vllm_diff_k.pkl      (vLLM dump path)
"""

from __future__ import annotations

import gc
import os
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the encoder + PC stack the bench uses.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "personal_context"))

from chunk_encoder_hf import HFChunkEncoder, preset_for  # noqa: E402

from vllm.v1.personal_context import (  # noqa: E402
    InMemoryStorage,
    PersonalContextConnector,
    ReusePlan,
    StoreConfig,
    load_plan,
)


# --------------------------------------------------------------------------
# Inputs — keep the comparison realistic by using bench-like text.
# --------------------------------------------------------------------------

SYS_TEXT = (
    "You are a helpful assistant. Today is 2026-05-25. Help the user keep "
    "track of their personal calendar, contacts, and recent messages. "
    "Answer concisely and only from the context provided below."
)

CHUNK_TEXT = (
    "Calendar event Saturday May 17 11:00-12:00. Subject: Coffee chat with "
    "Alice. Location: Bluebottle SOMA, 101 S Park St, San Francisco. "
    "Notes: bring laptop and project deck. Status: confirmed."
)


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------


def _extract_layered_kv(past_key_values):
    """Normalise HF ``past_key_values`` (DynamicCache or tuple-of-tuple)
    into a list ``[(K, V), ...]`` per layer. K/V HF shape:
    ``[batch, num_kv_heads, seq, head_dim]``."""
    if hasattr(past_key_values, "key_cache") and hasattr(
        past_key_values, "value_cache"
    ):
        return list(
            zip(past_key_values.key_cache, past_key_values.value_cache)
        )
    if hasattr(past_key_values, "layers"):
        out = []
        for layer in past_key_values.layers:
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                out.append((layer.keys, layer.values))
            else:
                out.append((layer[0], layer[1]))
        return out
    return [tuple(layer) for layer in past_key_values]


def _pad_to_block(token_ids, tokenizer, block_size):
    pad_needed = (-len(token_ids)) % block_size
    if pad_needed == 0:
        return list(token_ids)
    pool = []
    pad_text = "\n"
    while len(pool) < pad_needed:
        pool.extend(tokenizer.encode(pad_text, add_special_tokens=False))
        pad_text += "\n"
    return list(token_ids) + pool[:pad_needed]


def _truncate_to_block(token_ids, block_size):
    n = (len(token_ids) // block_size) * block_size
    if n == 0:
        raise ValueError("chunk text too short to fill one block")
    return list(token_ids[:n])


def _compare(label, A, B, device="cuda"):
    """Print max_abs, mean_abs, norms, cos_sim per layer."""
    A_dev = A.to(device).float().reshape(-1)
    B_dev = B.to(device).float().reshape(-1)
    diff = (A_dev - B_dev).abs()
    cos = F.cosine_similarity(
        A_dev.unsqueeze(0), B_dev.unsqueeze(0)
    ).item()
    return (
        f"{label:<18} "
        f"max={diff.max().item():>9.3e} "
        f"mean={diff.mean().item():>9.3e} "
        f"|A|={A_dev.norm().item():>9.3e} "
        f"|B|={B_dev.norm().item():>9.3e} "
        f"cos={cos:>7.4f}"
    )


# --------------------------------------------------------------------------
# Arms.
# --------------------------------------------------------------------------


def get_encoder_KV(model_id, dtype, device, chunk_ids, chunk_start_pos,
                   block_size, preset):
    """Encode chunk in isolation, delta-RoPE to the online position,
    return per-layer ``(K, V)`` each shaped
    ``[chunk_len, num_kv_heads, head_dim]``."""
    print("[arm] encoder: HFChunkEncoder.encode_chunk(old_pos_start=0)")
    encoder = HFChunkEncoder(
        preset=preset, device=device, dtype=dtype, lazy_model=False
    )
    chunk, entries = encoder.encode_chunk(
        chunk_ids, old_pos_start=0, block_size=block_size
    )
    storage = InMemoryStorage(
        StoreConfig(
            model_id=model_id,
            dtype=dtype,
            layout="NHD",
            num_layers=preset.num_layers,
            num_kv_heads=preset.num_kv_heads,
            head_dim=preset.head_dim,
            block_size=block_size,
        )
    )
    for h, blk in entries:
        storage.put(h, blk)
    connector = PersonalContextConnector(storage)
    plan_lookup = connector.lookup(ReusePlan(chunks=(chunk,)))
    loaded = load_plan(
        plan_lookup,
        new_pos_starts=(chunk_start_pos,),
        rope_theta=preset.rope_theta,
    )
    num_blocks = len(loaded.chunks[0].blocks)
    K_per_layer, V_per_layer = [], []
    for L in range(preset.num_layers):
        k_blocks = [loaded.chunks[0].blocks[i].keys[L] for i in range(num_blocks)]
        v_blocks = [loaded.chunks[0].blocks[i].values[L] for i in range(num_blocks)]
        K_per_layer.append(torch.cat(k_blocks, dim=0))
        V_per_layer.append(torch.cat(v_blocks, dim=0))
    del encoder
    torch.cuda.empty_cache()
    gc.collect()
    return K_per_layer, V_per_layer


def get_hf_KV(model_id, dtype, device, full_ids, chunk_start_pos, chunk_len,
              num_layers):
    """Fresh HF forward over sys+chunk; slice K and V at chunk positions."""
    print("[arm] vanilla HF: AutoModelForCausalLM.forward(sys+chunk)")
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    position_ids = torch.arange(
        len(full_ids), dtype=torch.long, device=device
    ).unsqueeze(0)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=True,
        )
    kv_layers = _extract_layered_kv(outputs.past_key_values)
    assert len(kv_layers) == num_layers, (
        f"HF returned {len(kv_layers)} layers, expected {num_layers}"
    )
    K_per_layer, V_per_layer = [], []
    for L in range(num_layers):
        K_hf, V_hf = kv_layers[L]  # both [1, nh, S, hd]
        K_chunk = K_hf[0, :, chunk_start_pos : chunk_start_pos + chunk_len, :]
        V_chunk = V_hf[0, :, chunk_start_pos : chunk_start_pos + chunk_len, :]
        K_per_layer.append(K_chunk.permute(1, 0, 2).contiguous())
        V_per_layer.append(V_chunk.permute(1, 0, 2).contiguous())
    del model, outputs, kv_layers
    torch.cuda.empty_cache()
    gc.collect()
    return K_per_layer, V_per_layer


def get_vllm_KV(model_id, dtype, full_ids, block_size, sys_len, chunk_len,
                num_layers, dump_path):
    """Fresh vLLM cold-start prefill over sys+chunk; load the pickle
    dumped by ``GpuModelRunner._maybe_diff_dump_kv`` and slice K and V
    at the chunk's physical blocks."""
    print(f"[arm] vanilla vLLM: LLM.generate with VLLM_DIFF_DUMP_K={dump_path}")
    if os.path.exists(dump_path):
        os.unlink(dump_path)
    os.environ["VLLM_DIFF_DUMP_K"] = dump_path

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from vllm.inputs import TokensPrompt  # noqa: PLC0415

    dtype_str = "float16" if dtype == torch.float16 else "bfloat16"
    llm = LLM(
        model=model_id,
        dtype=dtype_str,
        block_size=block_size,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_model_len=2048,
    )
    try:
        llm.generate(
            [TokensPrompt(prompt_token_ids=list(full_ids))],
            sampling_params=[
                SamplingParams(max_tokens=1, temperature=0.0)
            ],
        )
    finally:
        del llm
        torch.cuda.empty_cache()
        gc.collect()
        os.environ.pop("VLLM_DIFF_DUMP_K", None)

    if not os.path.exists(dump_path):
        raise RuntimeError(
            f"vLLM did not write {dump_path}. "
            "Confirm the runner's _maybe_diff_dump_kv patch is in place."
        )
    with open(dump_path, "rb") as f:
        dump = pickle.load(f)
    assert len(dump) == num_layers, (
        f"vLLM dump has {len(dump)} layers, expected {num_layers}"
    )

    # Cold-start single-request prefill on vLLM v1: block 0 is the null
    # block; subsequent blocks are allocated contiguously from block 1.
    # First chunk block = sys_len // block_size + 1.
    sys_blocks = sys_len // block_size
    num_chunk_blocks = chunk_len // block_size
    chunk_block_ids = list(
        range(sys_blocks + 1, sys_blocks + 1 + num_chunk_blocks)
    )

    K_per_layer, V_per_layer = [], []
    for L in range(num_layers):
        entry = dump[L]
        K_blocks = entry["K"]  # [N, block_size, num_kv_heads, head_dim]
        V_blocks = entry["V"]
        max_blk = chunk_block_ids[-1]
        if max_blk >= K_blocks.shape[0]:
            raise RuntimeError(
                f"vLLM dump layer {L} has {K_blocks.shape[0]} blocks, "
                f"chunk reaches block {max_blk}. Increase "
                "VLLM_DIFF_DUMP_K_NBLOCKS or shorten the prompt."
            )
        K_per_layer.append(
            torch.cat([K_blocks[b] for b in chunk_block_ids], dim=0)
        )
        V_per_layer.append(
            torch.cat([V_blocks[b] for b in chunk_block_ids], dim=0)
        )
    return K_per_layer, V_per_layer


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------


def main():
    model_id = os.environ.get("DIFF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    block_size = int(os.environ.get("DIFF_BLOCK_SIZE", "16"))
    dtype = torch.float16
    device = "cuda"
    skip_hf = os.environ.get("DIFF_SKIP_HF") == "1"
    dump_path = os.environ.get(
        "DIFF_VLLM_DUMP", "/tmp/vllm_diff_k.pkl"
    )

    print(
        f"[diff] model={model_id} block_size={block_size} dtype={dtype} "
        f"skip_hf={skip_hf}"
    )
    preset = preset_for(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sys_ids = tokenizer.encode(SYS_TEXT, add_special_tokens=False)
    sys_ids = _pad_to_block(sys_ids, tokenizer, block_size)
    chunk_ids = tokenizer.encode(CHUNK_TEXT, add_special_tokens=False)
    chunk_ids = _truncate_to_block(chunk_ids, block_size)

    sys_len = len(sys_ids)
    chunk_len = len(chunk_ids)
    chunk_start_pos = sys_len
    full_ids = sys_ids + chunk_ids
    print(
        f"[diff] sys={sys_len} tok ({sys_len // block_size} blk), "
        f"chunk={chunk_len} tok ({chunk_len // block_size} blk), "
        f"start_pos={chunk_start_pos}, total={len(full_ids)} tok"
    )

    # ---- ARMS ----
    encoder_K, encoder_V = get_encoder_KV(
        model_id, dtype, device, chunk_ids, chunk_start_pos,
        block_size, preset
    )
    hf_K, hf_V = (None, None)
    if not skip_hf:
        hf_K, hf_V = get_hf_KV(
            model_id, dtype, device, full_ids, chunk_start_pos,
            chunk_len, preset.num_layers
        )
    vllm_K, vllm_V = get_vllm_KV(
        model_id, dtype, full_ids, block_size, sys_len, chunk_len,
        preset.num_layers, dump_path
    )

    # ---- COMPARE ----
    print("\n[diff] per-layer comparison "
          "(K rows then V rows for each layer):")
    layer_n = preset.num_layers
    note_for = {
        0: "GOLD",
        max(1, layer_n // 4): "Q1",
        layer_n // 2: "MID",
        (3 * layer_n) // 4: "Q3",
        layer_n - 1: "TOP",
    }
    for L in range(layer_n):
        note = note_for.get(L, "")
        header = f"L{L:02d}" + (f" [{note}]" if note else "")
        print(f"--- {header} ---")
        print("  K " + _compare("encoder-vs-vllm", encoder_K[L], vllm_K[L], device))
        if hf_K is not None:
            print("  K " + _compare("hf-vs-vllm     ", hf_K[L], vllm_K[L], device))
        print("  V " + _compare("encoder-vs-vllm", encoder_V[L], vllm_V[L], device))
        if hf_V is not None:
            print("  V " + _compare("hf-vs-vllm     ", hf_V[L], vllm_V[L], device))

    # Layer 0 detail.
    print("\n[diff] layer 0 detail (chunk pos 0, head 0, first 8 head-dim slots):")
    print("  K:")
    print(f"    encoder: {[f'{x:+.5f}' for x in encoder_K[0][0, 0, :8].float().tolist()]}")
    if hf_K is not None:
        print(f"    hf:      {[f'{x:+.5f}' for x in hf_K[0][0, 0, :8].float().tolist()]}")
    print(f"    vllm:    {[f'{x:+.5f}' for x in vllm_K[0][0, 0, :8].float().tolist()]}")
    print("  V:")
    print(f"    encoder: {[f'{x:+.5f}' for x in encoder_V[0][0, 0, :8].float().tolist()]}")
    if hf_V is not None:
        print(f"    hf:      {[f'{x:+.5f}' for x in hf_V[0][0, 0, :8].float().tolist()]}")
    print(f"    vllm:    {[f'{x:+.5f}' for x in vllm_V[0][0, 0, :8].float().tolist()]}")


if __name__ == "__main__":
    main()
