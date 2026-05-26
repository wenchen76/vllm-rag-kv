#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diff sparse-Q-written fresh K at r=0.9 vs r=1.0.

Background:
    ``test_pc_full_selection_matches_vanilla`` proves that PC + sparse-Q
    at r=1.0 produces K byte-identical to vanilla vLLM (whole chunk is
    recomputed; scattered stale K never gets read). We have NOT verified
    that sparse-Q at intermediate r writes the same K at the positions
    it does recompute.

    Mechanism it could go wrong: sparse-Q's fresh recompute at layer L
    is driven by Q[L] @ K[<L] where K[<L] is the paged cache, which has
    been polluted by scatter at unselected positions. Even though the
    selected positions get fresh recompute, their X[L] reads polluted
    attention from L-1, so the resulting K[L] at selected positions
    *drifts* from vanilla. CacheBlend's whole premise is "this drift is
    small", but for Qwen2.5 1.5B the bench shows r=0.9 already
    catastrophic — so it's worth measuring.

Test:
    Two vLLM runs over the same prompt (sys + chunk + query) with PC
    enabled, identical chunk encoder state, only the selector differs:

      run A: SelectFirstR(1.0)  → all chunk positions recomputed
                                   → K_A == vanilla K (per existing test)
      run B: SelectFirstR(0.9)  → first 90 % of each chunk recomputed,
                                   tail 10 % stays scattered (encoder K)

    For each chunk position p:
      - if p ∈ r=0.9 selected: K_A[p] == vanilla; K_B[p] = sparse-Q's
        recompute from the polluted cache. Their delta isolates the
        CacheBlend recompute approximation error AT SELECTED positions.
      - if p ∉ r=0.9 selected: K_A[p] == vanilla; K_B[p] == encoder K.
        Their delta is the same encoder-vs-vanilla number we already
        measured in ``diff_encoder_vs_vanilla_k.py``.

Setup:
    - HFChunkEncoder encodes a fixed chunk text → InMemoryStorage
    - Storage pickle-bound via VLLM_PERSONAL_CONTEXT_TEST_BIND so the
      vLLM worker auto-binds it on connector init (same channel the GPU
      e2e tests use)
    - Each vLLM run does sys-only warmup + real prefill; dump fires on
      the second forward (VLLM_DIFF_DUMP_K_SKIP=1)

Usage:
    .venv/bin/python examples/personal_context/diff_sparse_q_r.py
"""

from __future__ import annotations

import gc
import os
import pickle
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "personal_context"))

from chunk_encoder_hf import HFChunkEncoder, preset_for  # noqa: E402

from vllm.v1.personal_context import (  # noqa: E402
    InMemoryStorage,
    StoreConfig,
)


# --------------------------------------------------------------------------
# Prompt setup — same shape as the bench's instance_001.
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
QUERY_TEXT = "When is my coffee chat with Alice?"


def _pad_to_block(tokens, tokenizer, block_size):
    pad_needed = (-len(tokens)) % block_size
    if pad_needed == 0:
        return list(tokens)
    pool = []
    pad_text = "\n"
    while len(pool) < pad_needed:
        pool.extend(tokenizer.encode(pad_text, add_special_tokens=False))
        pad_text += "\n"
    return list(tokens) + pool[:pad_needed]


def _truncate_to_block(tokens, block_size):
    n = (len(tokens) // block_size) * block_size
    if n == 0:
        raise ValueError("text too short for one block")
    return list(tokens[:n])


def _bind_storage(storage):
    """Pickle ``storage`` and set VLLM_PERSONAL_CONTEXT_TEST_BIND so the
    spawned worker auto-binds it on connector init. Returns the temp
    path (caller cleans it up)."""
    fd, path = tempfile.mkstemp(suffix=".pkl", prefix="pc_diff_")
    os.close(fd)
    with open(path, "wb") as f:
        pickle.dump({"storage": storage}, f)
    return path


def _run_vllm_dump(model_id, dtype_str, block_size, selector_r,
                   prompt_ids_warmup, prompt_ids_real, chunk_token_ids,
                   dump_path):
    """Boot a fresh vLLM, run sys-only warmup then a real call with
    reuse_plan, and dump K (and V) from the paged cache after the real
    forward."""
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from vllm.config import KVTransferConfig  # noqa: PLC0415
    from vllm.inputs import TokensPrompt  # noqa: PLC0415

    if os.path.exists(dump_path):
        os.unlink(dump_path)
    os.environ["VLLM_DIFF_DUMP_K"] = dump_path
    os.environ["VLLM_DIFF_DUMP_K_SKIP"] = "1"  # skip sys-only warmup

    kv_transfer_config = KVTransferConfig(
        kv_connector="PersonalContextKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "selector": {"type": "SelectFirstR", "r": selector_r},
        },
    )

    llm = LLM(
        model=model_id,
        dtype=dtype_str,
        block_size=block_size,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_model_len=2048,
        kv_transfer_config=kv_transfer_config,
        # PC's sparse-Q ``custom_mask`` path lives in
        # ``flashinfer.FlashInferMetadataBuilder._build_pc_custom_mask`` —
        # force this backend so non-contiguous Q positions at r<1 get
        # the right causal mask. Without it (default FLASH_ATTN on
        # Llama), FA treats Q row index as absolute position and the
        # mask drifts wherever sparse-Q skips positions.
        attention_config={"backend": "FLASHINFER"},
    )
    try:
        # Warmup primes sys into vLLM's prefix cache so PC's placement
        # check passes on the real call. Skipped from the dump.
        llm.generate(
            [TokensPrompt(prompt_token_ids=list(prompt_ids_warmup))],
            sampling_params=[
                SamplingParams(max_tokens=1, temperature=0.0)
            ],
        )

        reuse_plan = {
            "chunks": [
                {
                    "token_ids": list(chunk_token_ids),
                    "old_pos_start": 0,
                    "salt_hex": "",
                }
            ]
        }
        sp = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            extra_args={"kv_transfer_params": {"reuse_plan": reuse_plan}},
        )
        llm.generate(
            [TokensPrompt(prompt_token_ids=list(prompt_ids_real))],
            sampling_params=[sp],
        )
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        os.environ.pop("VLLM_DIFF_DUMP_K", None)
        os.environ.pop("VLLM_DIFF_DUMP_K_SKIP", None)

    if not os.path.exists(dump_path):
        raise RuntimeError(
            f"vLLM did not write {dump_path}. Check the runner patch + "
            f"SKIP env var."
        )
    with open(dump_path, "rb") as f:
        return pickle.load(f)


def _chunk_K_from_dump(dump, num_layers, chunk_block_ids):
    """Slice K at chunk physical blocks, return per-layer
    ``[chunk_len, num_kv_heads, head_dim]``."""
    out = []
    for L in range(num_layers):
        entry = dump[L]
        K_blocks = entry["K"]
        max_blk = chunk_block_ids[-1]
        if max_blk >= K_blocks.shape[0]:
            raise RuntimeError(
                f"layer {L} dump has {K_blocks.shape[0]} blocks, need "
                f"{max_blk}; increase VLLM_DIFF_DUMP_K_NBLOCKS."
            )
        out.append(
            torch.cat([K_blocks[b] for b in chunk_block_ids], dim=0)
        )
    return out


def main():
    model_id = os.environ.get("DIFF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    block_size = int(os.environ.get("DIFF_BLOCK_SIZE", "16"))
    dtype_str = "float16"
    dtype = torch.float16
    device = "cuda"

    preset = preset_for(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sys_ids = _pad_to_block(
        tokenizer.encode(SYS_TEXT, add_special_tokens=False),
        tokenizer, block_size,
    )
    chunk_ids = _truncate_to_block(
        tokenizer.encode(CHUNK_TEXT, add_special_tokens=False),
        block_size,
    )
    query_ids = _pad_to_block(
        tokenizer.encode(QUERY_TEXT, add_special_tokens=False),
        tokenizer, block_size,
    )

    sys_len = len(sys_ids)
    chunk_len = len(chunk_ids)
    query_len = len(query_ids)
    chunk_start_pos = sys_len
    num_chunk_blocks = chunk_len // block_size
    sys_blocks = sys_len // block_size
    chunk_block_ids = list(
        range(sys_blocks + 1, sys_blocks + 1 + num_chunk_blocks)
    )

    print(
        f"[diff-rB] model={model_id}, sys={sys_len} ({sys_blocks} blk), "
        f"chunk={chunk_len} ({num_chunk_blocks} blk), "
        f"query={query_len} ({query_len // block_size} blk)"
    )
    print(f"[diff-rB] chunk physical block ids = {chunk_block_ids}")

    # ----- Encode chunk + bind storage -----
    print("[diff-rB] encoding chunk via HFChunkEncoder...")
    encoder = HFChunkEncoder(
        preset=preset, device=device, dtype=dtype, lazy_model=False
    )
    chunk, entries = encoder.encode_chunk(
        chunk_ids, old_pos_start=0, block_size=block_size
    )
    del encoder
    torch.cuda.empty_cache()
    gc.collect()

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
    bind_path = _bind_storage(storage)
    os.environ["VLLM_PERSONAL_CONTEXT_TEST_BIND"] = bind_path

    prompt_warmup = sys_ids
    prompt_real = sys_ids + chunk_ids + query_ids

    try:
        print("[diff-rB] === run A: r=1.0 (full recompute baseline) ===")
        dump_A = _run_vllm_dump(
            model_id, dtype_str, block_size, 1.0,
            prompt_warmup, prompt_real, chunk_ids,
            "/tmp/diff_rB_A.pkl",
        )
        print("[diff-rB] === run B: r=0.9 (partial recompute) ===")
        dump_B = _run_vllm_dump(
            model_id, dtype_str, block_size, 0.9,
            prompt_warmup, prompt_real, chunk_ids,
            "/tmp/diff_rB_B.pkl",
        )
    finally:
        os.environ.pop("VLLM_PERSONAL_CONTEXT_TEST_BIND", None)
        try:
            os.unlink(bind_path)
        except OSError:
            pass

    K_A = _chunk_K_from_dump(dump_A, preset.num_layers, chunk_block_ids)
    K_B = _chunk_K_from_dump(dump_B, preset.num_layers, chunk_block_ids)

    # ----- Direct value inspection at the suspicious UNSEL position -----
    # Compute encoder K for chunk-local position 47 (last position), then
    # show side-by-side with K_A (r=1.0 cache) and K_B (r=0.9 cache). If
    # encoder K matches K_B closely but K_A is the outlier → scatter is
    # writing the expected encoder K but sparse-Q at r=1.0 didn't
    # overwrite. If K_B differs from encoder K → scatter wrote something
    # other than what we computed.
    import math as _math  # noqa: PLC0415
    from chunk_encoder_hf import HFChunkEncoder as _Enc  # noqa: PLC0415
    print("\n[diff-rB] direct value inspection at chunk-local pos 47 "
          "(L=0, head 0, first 8 dims):")
    enc_re = _Enc(preset=preset, device="cuda", dtype=torch.float16,
                  lazy_model=False)
    _, _entries = enc_re.encode_chunk(
        list(chunk_ids), old_pos_start=0, block_size=block_size
    )
    # entries[i] = (hash, KVBlock). Block 2 has 16 slots (chunk-local 32..47).
    enc_blk2_K_L0 = _entries[2][1].keys[0]  # shape [16, nh, hd]
    enc_K_pos47_pre = enc_blk2_K_L0[15, 0, :8].float().cpu()  # PRE delta-RoPE
    # Apply delta-RoPE to align to online position 47 + chunk_start_pos.
    from vllm.v1.personal_context.rope import apply_delta_rope as _ddr  # noqa: PLC0415
    delta = chunk_start_pos  # old_pos_start=0 → new_pos_start=chunk_start_pos
    enc_blk2_K_L0_rotated = _ddr(
        enc_blk2_K_L0, delta, rope_theta=preset.rope_theta,
    )
    enc_K_pos47_post = enc_blk2_K_L0_rotated[15, 0, :8].float().cpu()
    print(
        f"  encoder K pre  delta-RoPE: {[f'{x:+.5f}' for x in enc_K_pos47_pre.tolist()]}"
    )
    print(
        f"  encoder K post delta-RoPE: {[f'{x:+.5f}' for x in enc_K_pos47_post.tolist()]}"
    )
    print(
        f"  cache K_A (r=1.0):         {[f'{x:+.5f}' for x in K_A[0][47, 0, :8].float().cpu().tolist()]}"
    )
    print(
        f"  cache K_B (r=0.9):         {[f'{x:+.5f}' for x in K_B[0][47, 0, :8].float().cpu().tolist()]}"
    )
    del enc_re
    torch.cuda.empty_cache()
    gc.collect()

    # ----- Per-position cos at each layer -----
    # SelectFirstR(r=0.9) selects positions [chunk_start_pos ..
    # chunk_start_pos + ceil(0.9 * chunk_len) - 1].
    import math
    k_selected = math.ceil(0.9 * chunk_len)
    sel_local = set(range(k_selected))  # local indices (0..chunk_len-1)

    print(
        f"\n[diff-rB] r=0.9 selects {k_selected}/{chunk_len} positions "
        f"(local indices 0..{k_selected - 1})"
    )

    layer_n = preset.num_layers
    note_for = {
        0: "GOLD",
        max(1, layer_n // 4): "Q1",
        layer_n // 2: "MID",
        (3 * layer_n) // 4: "Q3",
        layer_n - 1: "TOP",
    }

    print(
        f"\n{'L':>3} {'note':<5} "
        f"{'sel cos μ':>10} {'sel cos min':>12} {'sel mean|Δ|':>12} "
        f"{'unsel cos μ':>12} {'unsel cos min':>14} {'unsel mean|Δ|':>14}"
    )
    print("-" * 88)
    for L in range(layer_n):
        a = K_A[L].to(device).float()  # [chunk_len, nh, hd]
        b = K_B[L].to(device).float()
        # per-position cos: dot over (nh, hd) flattened
        a_flat = a.reshape(chunk_len, -1)
        b_flat = b.reshape(chunk_len, -1)
        pos_cos = F.cosine_similarity(a_flat, b_flat, dim=-1)  # [chunk_len]
        pos_diff = (a - b).abs().mean(dim=(1, 2))               # [chunk_len]

        sel_mask = torch.tensor(
            [i in sel_local for i in range(chunk_len)],
            device=device, dtype=torch.bool,
        )
        unsel_mask = ~sel_mask
        sel_cos = pos_cos[sel_mask]
        unsel_cos = pos_cos[unsel_mask]
        sel_diff = pos_diff[sel_mask]
        unsel_diff = pos_diff[unsel_mask]

        note = note_for.get(L, "")
        print(
            f"{L:>3} {note:<5} "
            f"{sel_cos.mean().item():>10.4f} "
            f"{sel_cos.min().item():>12.4f} "
            f"{sel_diff.mean().item():>12.3e} "
            f"{unsel_cos.mean().item():>12.4f} "
            f"{unsel_cos.min().item():>14.4f} "
            f"{unsel_diff.mean().item():>14.3e}"
        )

    # Per-position cos table at TOP layer for visual inspection.
    print(f"\n[diff-rB] top-layer (L{layer_n - 1}) per-position cos & sel:")
    L = layer_n - 1
    a = K_A[L].to(device).float().reshape(chunk_len, -1)
    b = K_B[L].to(device).float().reshape(chunk_len, -1)
    pos_cos = F.cosine_similarity(a, b, dim=-1)
    for i in range(chunk_len):
        flag = "SEL" if i in sel_local else "stl"
        print(
            f"  pos {i:>3d} ({flag}) "
            f"cos={pos_cos[i].item():+.4f}"
        )


if __name__ == "__main__":
    main()
