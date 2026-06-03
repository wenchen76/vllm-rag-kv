#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Throwaway probe: confirm RequestOutput.metrics is populated under
disable_log_stats=False, and that prefill_time / first_token_latency are
usable as the single-call TTFT source (replacing the two-call bench).

Runs ONE reuse request (memory backend) on the 2k dataset and dumps every
metrics field. Run on GPU:

    .venv/bin/python examples/personal_context/_verify_metrics.py

Delete after deciding whether to switch the bench to single-call metrics.
"""
import dataclasses
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.personal_context.chunk_encoder_hf import preset_for
from examples.personal_context.rag_demo import (
    RAGIndex,
    _pc_test_bind_storage,
    build_prompt_and_plan,
)

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA = Path(__file__).parent / "sample_data_2k.jsonl"
TOP_K = 14
BLOCK_SIZE = 16
R = 0.5

instances = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
instances = instances[:1]  # one instance is enough to inspect metrics

preset = preset_for(MODEL)
index = RAGIndex(
    preset=preset, embedder_model_id="sentence-transformers/all-MiniLM-L6-v2",
    block_size=BLOCK_SIZE, device="cuda", store_backend="memory",
)
index.ingest_instances(instances)
index.build_faiss()

inst = instances[0]
retrieved = index.retrieve(inst["query"], top_k=TOP_K)
prod_ids, reuse_params, _sys_raw, _sys_pad = build_prompt_and_plan(
    index.tokenizer, inst["sys"], retrieved, inst["query"],
    block_size=BLOCK_SIZE,
)

# Free the HF encoder (~15GB for an 8B model) BEFORE booting vLLM, or both
# 8B models sit in VRAM at once and OOM a 24GB card (mirrors bench main).
import gc  # noqa: E402

import torch  # noqa: E402

if getattr(index, "encoder", None) is not None:
    del index.encoder
    index.encoder = None
    gc.collect()
    torch.cuda.empty_cache()

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

kv_cfg = KVTransferConfig(
    kv_connector="PersonalContextKVConnector", kv_role="kv_both",
    kv_connector_extra_config={"selector": {"type": "SelectFirstR", "r": R}},
)

with _pc_test_bind_storage(index.kv_store):
    llm = LLM(
        model=MODEL, dtype="float16", block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9, enforce_eager=True,
        kv_transfer_config=kv_cfg,
        attention_config={"backend": "FLASHINFER"},
        disable_log_stats=False,  # <-- the key: enable metrics
    )
    # warmup sys so reuse can activate
    llm.generate([TokensPrompt(prompt_token_ids=list(prod_ids[:32]))],
                 sampling_params=[SamplingParams(max_tokens=1, temperature=0.0)])

    out = llm.generate(
        [TokensPrompt(prompt_token_ids=list(prod_ids))],
        sampling_params=[SamplingParams(
            max_tokens=160, temperature=0.0, extra_args=reuse_params)],
    )[0]

m = out.metrics
print("\n===== RequestOutput.metrics =====")
print(f"type: {type(m)}")
if m is None:
    print("!!! metrics is None — disable_log_stats=False did NOT populate it")
else:
    for f in dataclasses.fields(m):
        print(f"  {f.name} = {getattr(m, f.name)}")
    # The two we care about for single-call TTFT:
    print("\n--- candidate TTFT sources ---")
    for attr in ("prefill_time", "first_token_latency", "inference_time",
                 "decode_time", "e2e_latency"):
        v = getattr(m, attr, "MISSING")
        print(f"  {attr} = {v}")
print(f"\ngenerated[:80]: {out.outputs[0].text[:80]!r}")
