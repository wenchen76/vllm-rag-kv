# Personal-Context KV Reuse for RAG (vLLM)

Cross-request **KV-cache reuse for retrieval-augmented generation** on top of
vLLM v1.

In a RAG serving loop the same retrieved chunks (your calendar, your messages,
a knowledge-base passage) are fed back into the model on *every* request, and
vLLM re-prefills them from scratch every time. Prefill is the expensive part of
a RAG turn — it dominates **time-to-first-token (TTFT)**. This project encodes
each chunk's key/value tensors **once, offline**, stores them in a
content-addressed KV store, and **scatters them straight into vLLM's paged KV
cache** at query time so the model never recomputes them.

The result is a drop-in `KVConnectorBase_V1` implementation
(`PersonalContextKVConnector`) plus a small, fully-tested support library
(`vllm.v1.personal_context`) and a set of runnable demos, benchmarks, and
numerical-correctness diagnostics under this directory.

> ⚠️ **Research / experimental feature.** This is a fork of vLLM. It targets
> standard-RoPE decoder models (Qwen2.5, Llama-3) and is meant for
> experimentation and benchmarking, not production serving.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Usage reference](#usage-reference)
- [Technical deep dive](#technical-deep-dive)
- [Configuration reference](#configuration-reference)
- [Supported models](#supported-models)
- [Benchmarking](#benchmarking)
- [Correctness diagnostics](#correctness-diagnostics)
- [Testing](#testing)
- [Limitations & sharp edges](#limitations--sharp-edges)

---

## Why this exists

A RAG request looks like:

```
prompt = system_prompt  +  chunk_1 + chunk_2 + … + chunk_k  +  user_query
```

The retrieved chunks are usually the bulk of the tokens, and they recur across
requests. Vanilla vLLM recomputes attention over all of them on every prefill.
If a chunk's KV were already in the cache, prefill would only have to pay for
`system + query + (optionally) a fraction of the chunks`, cutting TTFT roughly
in proportion to how many chunk tokens are reused.

vLLM's built-in **prefix caching** can't help here: it is *position-dependent*
and *prefix-contiguous* — it only reuses a block if the entire token prefix up
to that block is byte-identical. Retrieved chunks appear at different offsets,
in different orders, interleaved with different system prompts, so the prefix
almost never matches. We need a **position-independent, content-addressed** KV
cache that can place a chunk anywhere in a new prompt. That requires solving the
one thing that makes KV position-dependent: **RoPE**.

---

## How it works

### Offline ingestion (once per chunk)

```
chunk text
  │  tokenize, pad to a block_size multiple
  ▼
HFChunkEncoder  ── runs the target model on the chunk in isolation
  │              harvests past_key_values, slices per block,
  │              permutes to NHD layout
  ▼
per-block (content_hash → KVBlock)   ──►  KV store (Redis)
                                          + vector  ──► vector DB
```

Each chunk is hashed **per block** with a position-independent SHA-256 over its
token ids (+ optional tenant salt), domain-separated from vLLM's own
prefix-rolling hashes. K is stored *post-RoPE at its encoding position*; V is
position-independent.

### Online query (per request)

```
user query
  │ embed → vector DB top-k → retrieved chunks
  ▼
build prompt = sys + chunks + query   and   a reuse_plan describing the chunks
  ▼
vLLM.generate(prompt, kv_transfer_params={"reuse_plan": …})
  │
  ├─ scheduler: get_num_new_matched_tokens()   discount prefill budget by reused tokens
  │             build_connector_meta() → Selector.select()
  │               → pick selected_positions to recompute (sparse-Q)
  │
  └─ worker:    start_load_kv()
                  load_plan()             fetch blocks, apply DELTA-RoPE to K
                  scatter_loaded_plan()   copy_() K/V into the paged cache blocks
                build_sparse_q_arrays()   expand query → (query ∪ selected_positions)
                chunk-aware prefill       forward only those positions
```

The two non-obvious pieces are **delta-RoPE** (re-rotating a chunk's stored K
from its encoding position to wherever it lands in the new prompt) and
**sparse-Q selective recompute** (recomputes only a
fraction `r` of each chunk to trade quality against prefill cost). Both are
covered in the [deep dive](#technical-deep-dive).

---

## Repository layout

| Path | What it is |
| --- | --- |
| `vllm/v1/personal_context/` | The support library: hashing, storage, delta-RoPE, load, scatter, selection, chunk-aware prefill, cache-isolation policy. ~1.3k LoC, fully unit-tested. |
| `vllm/distributed/kv_transfer/kv_connector/v1/personal_context_connector.py` | `PersonalContextKVConnector` — the vLLM-facing `KVConnectorBase_V1` glue (~1.1k LoC). Registered in the connector `factory.py`. |
| `examples/personal_context/rag_demo.py` | End-to-end RAG demo: ingest a JSONL corpus, retrieve from a vector DB, generate with KV reuse. |
| `examples/personal_context/chunk_encoder_hf.py` | Offline chunk→KVBlock encoder backed by HuggingFace transformers; model presets. |
| `examples/personal_context/bench_prefill.py` | TTFT + generation-quality benchmark sweeping the selective-recompute knob `r`. |
| `examples/personal_context/diff_encoder_vs_vanilla_k.py` | Numerical diagnostic: encoder K vs vanilla HF K vs vanilla vLLM K, layer by layer. |
| `examples/personal_context/diff_sparse_q_r.py` | Numerical diagnostic: K written by sparse-Q at `r=0.9` vs the `r=1.0` baseline. |
| `examples/personal_context/sample_data.jsonl` | 5 personal-context RAG instances (calendar/messages/notes/email/contacts) with gold answers. |
| `tests/v1/personal_context/` | Unit + integration + CPU/GPU e2e tests (15 files). |

---

## Installation

This repo follows the vLLM contributor workflow — use `uv`, never bare `pip`.

```bash
# Build vLLM (Python-only changes, precompiled kernels):
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# Add the personal-context extras (vector DB, embeddings, Redis, scoring):
uv pip install -e ".[personal-context]"
```

The `personal-context` extra pulls in:

| Package | Scope | Used for |
| --- | --- | --- |
| `faiss-cpu>=1.8.0` | Demo | Vector DB for the example's retrieval index |
| `sentence-transformers>=3.0.0` | Demo | Query/chunk embeddings for retrieval |
| `redis>=5.0.0` | Store backend | Optional cross-process KV store (`store_backend=redis`) |
| `fakeredis>=2.20.0` | Tests | In-process Redis for unit tests |
| `rouge-score>=0.1.2` | Benchmark | ROUGE-L quality scoring |

None of these are required by the core connector or the
`vllm.v1.personal_context` library itself — with the default in-memory store it
imports nothing beyond vLLM/torch. They exist only to run the **demo**
(`faiss-cpu` + `sentence-transformers` drive retrieval), the optional Redis store
backend, and the tests/benchmark.

A CUDA GPU is required to run the demos/benchmarks (they boot a real vLLM
engine). The pure-library unit tests run on CPU.

---

## Quickstart

```bash
# In-memory KV store (single process), Qwen2.5-1.5B:
.venv/bin/python examples/personal_context/rag_demo.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --top_k 4 \
    --selector_r 0.8
```

The demo ingests `sample_data.jsonl`, builds a vector DB index, reports retrieval
quality against the labelled gold chunks, then runs the augmented prompt through
vLLM with KV reuse and prints the generated answer next to the gold answer.

### With a Redis-backed store (cross-process)

```bash
docker run -d --name pc-redis -p 6379:6379 redis:7

.venv/bin/python examples/personal_context/rag_demo.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --store_backend redis \
    --store_url redis://localhost:6379
```

In Redis mode the encoded blocks live in Redis and the vLLM worker pulls them in
via `kv_connector_extra_config` — the same path a real deployment would use,
where ingestion and serving are separate processes.

---

## Usage reference

### `rag_demo.py`

| Flag | Default | Description |
| --- | --- | --- |
| `--data` | `sample_data.jsonl` | JSONL corpus (`id`, `sys`, `query`, `answer`, `chunks`). |
| `--model` | `meta-llama/Meta-Llama-3-8B-Instruct` | HF model id; must have a registered preset. |
| `--top_k` | `4` | Vector DB retrieval depth (top-k). |
| `--selector_r` | `1.0` | Fraction of each chunk to recompute (sparse-Q). `1.0` = full recompute (lossless), `0.0` = pure reuse. |
| `--max_tokens` | `128` | Tokens generated per query. |
| `--embedder` | `sentence-transformers/all-MiniLM-L6-v2` | Retrieval embedding model. |
| `--store_backend` | `memory` | `memory` or `redis`. |
| `--store_url` | `redis://localhost:6379` | Redis URL (redis backend only). |
| `--gpu_memory_utilization` | `0.85` | vLLM VRAM fraction. |

### Programmatic API

KV reuse is driven entirely through vLLM's existing connector + sampling APIs —
no engine surgery:

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.inputs import TokensPrompt

# 1) Turn on the connector and pick a selective-recompute policy.
kv_transfer_config = KVTransferConfig(
    kv_connector="PersonalContextKVConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "selector": {"type": "SelectFirstR", "r": 0.8},
        # Optional cross-process store:
        "store_backend": "redis",
        "store_url": "redis://localhost:6379",
    },
)

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="float16",
    block_size=16,
    kv_transfer_config=kv_transfer_config,
)

# 2) Per request, attach a reuse_plan describing the retrieved chunks.
reuse_plan = {
    "chunks": [
        {
            "token_ids": chunk_token_ids,  # the chunk's tokens, block-aligned
            "old_pos_start": 0,            # position the chunk was encoded at
            "salt_hex": "",               # optional tenant/namespace salt
        },
    ]
}
sp = SamplingParams(
    max_tokens=128,
    temperature=0.0,
    extra_args={"kv_transfer_params": {"reuse_plan": reuse_plan}},
)

llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sampling_params=[sp])
```

The connector reads the `reuse_plan`, looks up each chunk's blocks in the bound
store, and (on a hit) loads + scatters them into the freshly allocated paged
cache blocks before the forward pass.

---

## Technical deep dive

### 1. Position-independent, content-addressed KV blocks

vLLM's prefix cache keys a block on *the entire token prefix leading up to it*,
which is why it can't reuse a chunk that appears at a different offset. The
personal-context store instead hashes **only the block's own token ids** (plus
an optional salt), via `hash_block()` — a 32-byte SHA-256 digest,
domain-separated from vLLM's rolling hashes. The same chunk therefore has the
same key no matter where it lands in a prompt. Blocks are `block_size`-aligned
(default 16 tokens); chunks are **padded**, not truncated,
so content is preserved.

### 2. Delta-RoPE: the key trick

RoPE rotates each key vector by an angle proportional to its absolute position.
A chunk encoded at position `old_pos_start` has K already rotated for *that*
position. Reused at runtime position `new_pos_start`, those rotations are wrong.

Because RoPE rotations compose additively, we don't need the un-rotated K — we
just apply the **difference**:

```
R(old_pos) · R(delta) = R(old_pos + delta) = R(new_pos),   delta = new_pos − old_pos
```

`apply_delta_rope(keys, delta, rope_theta)` rotates a whole block by one constant
`delta` along the position axis (one rotation per layer per block, fp32 internal,
Llama-style rotate-half). V is position-independent and merely cloned.

```python
def apply_delta_rope(keys: torch.Tensor,        # [block_size, num_kv_heads, head_dim]
                     delta: int,
                     rope_theta: float = 10000.0) -> torch.Tensor: ...
```

### 3. Sparse-Q selective recompute

Reusing a chunk's K verbatim is only approximate: a chunk encoded in isolation
never "saw" the surrounding system prompt, so its higher-layer K drifts from
what a full prefill would produce. [CacheBlend's](https://arxiv.org/abs/2405.16444)
insight is that recomputing a *small fraction* of positions recovers most of the
quality. The **selector** decides which positions to recompute:

- `NoSelection` / `r = 0.0` — pure reuse, recompute nothing (cheapest, most drift).
- `SelectFirstR(r)` — recompute the first `ceil(r · L)` positions of each chunk
  (early tokens anchor attention; the prefix recovers most accuracy).
- `r = 1.0` — recompute everything: **provably byte-identical to vanilla vLLM**
  , at no prefill savings.

`SelectFirstR` is the only selector implemented today (it always recomputes a
contiguous prefix). A future selector, **HKVD** (High KV Deviation), is planned:
rather than the prefix, it picks the positions whose stored KV deviates most from
a full prefill.

At `r < 1` the recomputed Q rows are **non-contiguous**, so the connector forces
the **FlashInfer** backend and builds a custom causal mask
(`build_pc_prefill_custom_mask`) — otherwise FlashAttention treats the Q row
index as the absolute position and the causal mask drifts wherever sparse-Q
skips a position.

### 4. Cache-isolation invariant

A mixed-KV request (one carrying a `reuse_plan`) must **never** write its blocks
back into vLLM's position-dependent prefix cache: a later request sharing that
prefix would then read K rotated for the wrong position.

### 5. Connector lifecycle

`PersonalContextKVConnector` implements the `KVConnectorBase_V1` contract:

| Step | Method | Role |
| --- | --- | --- |
| Scheduler | `get_num_new_matched_tokens` | Trust the retriever (chunks came from this store), so report the full reused token count and discount the prefill budget. Lookup runs defensively — any miss falls back to full prefill. |
| Scheduler | `update_state_after_alloc` | Stash `(plan, num_external_tokens)` once blocks are allocated. |
| Scheduler | `build_connector_meta` → `_invoke_selector` → `Selector.select` | Pair each plan with its scheduler-assigned block ids, and pick `selected_positions` to recompute (sparse-Q). |
| Worker | `start_load_kv` | `load_plan()` (fetch + delta-RoPE) then `scatter_loaded_plan()` into the paged cache. |
| Worker (runner) | `_pc_build_sparse_q_arrays` | Expand the forwarded Q rows to `query ∪ selected_positions`; FlashInfer's `_build_pc_custom_mask` masks the non-contiguous Q. |

---

## Configuration reference

All knobs live in `KVTransferConfig.kv_connector_extra_config`.

### Selector

| `selector` value | Behaviour |
| --- | --- |
| omitted / `null` | Default: no recompute, stale K reused as-is. |
| `"NoSelection"` | Same as null. |
| `"SelectFirstR"` | `SelectFirstR(r=1.0)` — full recompute (lossless, no savings). |
| `{"type": "SelectFirstR", "r": 0.5}` | Recompute the first 50% of each chunk. |

> **`SelectFirstR` is currently the only implemented selective-recompute
> policy.** It recomputes a contiguous prefix of each chunk. A second policy,
> **HKVD** (High KV Deviation), is planned — instead of always taking the
> prefix, it will recompute the positions whose stored KV deviates most from a
> full prefill.

### Store backend

| key | values |
| --- | --- |
| `store_backend` | `"redis"` to bind a `RedisKVStorage` (schema auto-verified on connect). Omit for the in-process path. |
| `store_url` | e.g. `redis://localhost:6379` (required when `store_backend="redis"`). |

> For tests and the in-memory demo path, a populated `InMemoryStorage` is
> pickled to the worker via the `VLLM_PERSONAL_CONTEXT_TEST_BIND` env var. This is
> a test/dev channel, not a production mechanism — use Redis for cross-process.

---

## Supported models

Encoding and reuse require the model's exact architecture constants. Registered
presets (`chunk_encoder_hf.KNOWN_PRESETS`):

| Model | Layers | KV heads | head_dim | rope_theta |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | 24 | 2 | 64 | 1,000,000 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 28 | 2 | 128 | 1,000,000 |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 32 | 8 | 128 | 500,000 |

Add a model by declaring a `ModelPreset` and appending it to `KNOWN_PRESETS`.

> **Llama-3.1 / 3.2 are intentionally excluded.** They use `rope_scaling` (NTK)
> piecewise-rescaled frequencies, which `apply_delta_rope` does not implement.
> The encoder refuses scaled-RoPE models loudly rather than corrupting K.

---

## Benchmarking

`bench_prefill.py` sweeps the recompute knob `r` and reports TTFT and generation
quality.

```bash
.venv/bin/python examples/personal_context/bench_prefill.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --r_values 1.0,0.75,0.5,0.25,0.0
```

**Methodology.** For each `(r, query)` it makes two `generate` calls:

1. **TTFT call** (`max_tokens=1`) over the production `sys + chunks + query`
   prompt — sparse-Q activates here, so the wall-clock is the latency a real
   user would see, and it leaves the `r`-specific chunk K/V in the cache.
2. **Generation call** (`max_tokens=128`) over the same prompt — the prefix
   cache hits the prefill, so this measures the quality of text produced from
   the `r`-specific cache state end-to-end.

Quality uses two complementary metrics on the *generated* text:

- **ROUGE-L** (lexical) — catches single-token factual flips.
- **Embedding cosine similarity** (semantic).

and a **recovery** score that normalises each metric between the pure-reuse and
full-recompute endpoints:

```
recovery(r) = (q(r) − q@r=0.0) / (q@r=1.0 − q@r=0.0)
```

> The benchmark deliberately scores *generation*, not teacher-forced
> perplexity: requesting `prompt_logprobs` disables prefix caching in vLLM,
> which would prevent KV reuse from activating at all and collapse every `r` to
> vanilla prefill.

---

## Testing

The support library has CPU-runnable unit tests plus CPU/GPU end-to-end tests:

```bash
# Pure-library unit tests (CPU):
.venv/bin/python -m pytest tests/v1/personal_context/ -v \
    --ignore=tests/v1/personal_context/test_e2e_gpu.py

# GPU end-to-end (requires a CUDA device):
.venv/bin/python -m pytest tests/v1/personal_context/test_e2e_gpu.py -v
```

---

## Limitations & sharp edges

- **Standard RoPE only.** No `rope_scaling` / NTK (Llama-3.1/3.2). Scaled-RoPE
  models are refused at encode time.
- **First-query warmup.** The demo's first query may fall back to full prefill
  until the system prompt is primed into the prefix cache; subsequent queries
  reuse normally.
- **Tokenisation drift.** Chunks are tokenised once and stored; BPE merges
  across a concatenation boundary can re-tokenise differently, so chunks are
  padded to block boundaries to keep the stored hash matching the prompt tokens
  (a few "wasted" tokens).
- **Block size must match** between the offline encoder and the serving engine
  (default 16).

