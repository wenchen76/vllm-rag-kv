# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 13 bench: TTFT + perplexity + recovery vs selector_r.

Measures, per ``(selector_r, query)`` pair, **two** things — split
into two separate ``llm.generate`` calls because they need different
prompts:

    1. **TTFT (Time To First Token)** — wall-clock around a
       ``max_tokens=1`` generate on the *production* prompt
       ``sys + chunks + query``. This is the prefill latency a user
       would actually see; sampling happens right after the query
       token (not after gold), so this number is comparable to a real
       deployment's TTFT.

    2. **Perplexity over the gold answer** — teacher-forcing via a
       SECOND ``max_tokens=1`` generate whose prompt is
       ``sys + chunks + query + gold``, with ``prompt_logprobs=1``.
       vLLM's prefix cache reuses Call 1's K/V state (which carries
       the r-specific PC reuse fingerprint), so Call 2 only
       forward-passes the gold tail, returning logprobs at each gold
       position. Mean NLL → ``exp`` → perplexity.

The two calls must be in this order — otherwise Call 2 would prime
the prefix cache and Call 1 would measure a near-zero prefill
(invalid).

From the per-query perplexities we compute **recovery** at each
intermediate r:

    recovery(r) = (ppl_stale - ppl_hybrid(r)) / (ppl_stale - ppl_gold)

where ``ppl_stale`` = ppl at r=0.0 (full reuse, no recomputation)
and ``ppl_gold`` = ppl at r=1.0 (full recomputation, vanilla
equivalent). 1.0 = full quality recovered with partial recompute;
0.0 = no better than stale. recovery is intentionally only computed
for intermediate r in (0, 1); the endpoints are 0 and 1 by
construction.

CAVEAT — prefix cache pollution
-------------------------------
PC's scatter writes chunk K/V into the worker's paged cache; vLLM's
prefix cache then captures those blocks (it hashes only on token
sequence, not K/V content). A second request with the same prompt
would hit the FIRST request's K/V state regardless of its own r
value, silently disabling per-request selector tuning.

This bench works around the issue by **tearing down the LLM between
each r value** — fresh LLM → fresh prefix cache → no cross-r
contamination. Static-r production deployments are unaffected, but
dynamic-r needs a real fix (a known open issue, not addressed by
this bench).

CAVEAT — perplexity blind spots
-------------------------------
Perplexity averages over all gold tokens, so a single catastrophic
``"Yes"`` ↔ ``"No"`` flip is diluted by 50+ surrounding tokens that
read similarly under both stale and fresh K/V. Recovery will look
"middling" even when actual generation is wrong. To catch those
cases, layer ROUGE / cosine-sim / LLM-judge on top of this bench
(future work).

Usage:
    # vanilla redis works (we only use KV ops, no vector module)
    redis-server --daemonize yes --port 6379
    .venv/bin/python examples/personal_context/bench_prefill.py \\
        --store_backend redis \\
        --r_values 1.0,0.75,0.5,0.25,0.0
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch


# Repo-root on path so we can import the sibling demo helpers.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.personal_context.chunk_encoder_hf import (  # noqa: E402
    preset_for,
)
from examples.personal_context.rag_demo import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_EMBEDDER,
    DEFAULT_REDIS_URL,
    RAGIndex,
    _pad_tokens_to_block_size,
    _pc_test_bind_storage,
    build_prompt_and_plan,
)


DEFAULT_R_VALUES = "1.0,0.75,0.5,0.25,0.0"


# ----------------------------- data classes -----------------------------


@dataclasses.dataclass
class QuerySpec:
    """All per-instance bench inputs, pre-computed once."""

    instance_id: str
    # Production-shaped prompt: sys + chunks + query. Drives TTFT.
    prod_prompt_ids: list[int]
    # Extended prompt: prod + gold answer tokens. Drives perplexity.
    eval_prompt_ids: list[int]
    # Connector reuse_plan riding in SamplingParams.extra_args.
    reuse_params: dict
    # Slice of eval_prompt_ids that holds the gold answer.
    gold_start: int
    gold_len: int
    # Provenance for the report header.
    sys_len_padded: int
    chunks_total: int


@dataclasses.dataclass
class Measurement:
    instance_id: str
    ttft_ms: float
    ppl: Optional[float]


# ----------------------------- helpers -----------------------------


def _build_query_specs(
    instances: list[dict],
    index: RAGIndex,
    args: argparse.Namespace,
) -> list[QuerySpec]:
    """Pre-build prompts + reuse_plans + gold positions for every query.

    Done up-front (before any LLM boot) so the per-r loop only spends
    time on inference, and so tokenisation surprises (e.g., a gold
    answer that exceeds max_model_len) surface immediately rather
    than after the first ~30s LLM load.
    """
    tokenizer = index.tokenizer
    specs: list[QuerySpec] = []
    for inst in instances:
        retrieved = index.retrieve(inst["query"], top_k=args.top_k)
        (
            prod_prompt_ids,
            reuse_params,
            _sys_len_raw,
            sys_len_padded,
        ) = build_prompt_and_plan(
            tokenizer,
            inst["sys"],
            retrieved,
            inst["query"],
            block_size=args.block_size,
        )

        gold_ids = tokenizer.encode(
            inst["answer"], add_special_tokens=False
        )
        if len(gold_ids) == 0:
            print(
                f"[bench] WARN: gold for {inst['id']} tokenises to 0 "
                "tokens; perplexity will be N/A."
            )
        eval_prompt_ids = list(prod_prompt_ids) + list(gold_ids)
        chunks_total = sum(len(e.token_ids) for e in retrieved)

        specs.append(
            QuerySpec(
                instance_id=inst["id"],
                prod_prompt_ids=prod_prompt_ids,
                eval_prompt_ids=eval_prompt_ids,
                reuse_params=reuse_params,
                gold_start=len(prod_prompt_ids),
                gold_len=len(gold_ids),
                sys_len_padded=sys_len_padded,
                chunks_total=chunks_total,
            )
        )
    return specs


def _compute_perplexity(
    prompt_logprobs: Optional[list],
    prompt_token_ids: list[int],
    gold_start: int,
    gold_len: int,
) -> Optional[float]:
    """Mean NLL over gold positions → ``exp`` = perplexity.

    Returns ``None`` when no gold logprobs are recoverable (e.g.,
    every gold position was somehow prefix-cached pre-prefill, which
    our two-call design should prevent in practice).
    """
    if gold_len == 0 or prompt_logprobs is None:
        return None
    nll_sum = 0.0
    count = 0
    for i in range(gold_start, gold_start + gold_len):
        if i >= len(prompt_logprobs):
            break
        lp_dict = prompt_logprobs[i]
        if lp_dict is None:
            continue
        token_id = prompt_token_ids[i]
        lp_obj = lp_dict.get(token_id)
        if lp_obj is None:
            continue
        # vLLM Logprob has .logprob (float); defensive against future
        # API drift where the value might land directly on lp_obj.
        lp_val = getattr(lp_obj, "logprob", lp_obj)
        nll_sum += -float(lp_val)
        count += 1
    if count == 0:
        return None
    return math.exp(nll_sum / count)


# ----------------------------- per-r bench loop -----------------------------


def _bench_for_r(
    r: float,
    query_specs: list[QuerySpec],
    sys_padded_tokens: list[int],
    args: argparse.Namespace,
) -> list[Measurement]:
    """Boot LLM at this r, warmup, run each query (2 calls), tear down."""
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.inputs import TokensPrompt

    extra: dict = {
        "selector": {"type": "SelectFirstR", "r": r},
    }
    if args.store_backend == "redis":
        extra["store_backend"] = "redis"
        extra["store_url"] = args.store_url

    kv_transfer_config = KVTransferConfig(
        kv_connector="PersonalContextKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra,
    )

    print(
        f"\n{'=' * 72}\n"
        f"[bench] r={r}: booting LLM\n"
        f"{'=' * 72}"
    )
    llm = LLM(
        model=args.model,
        dtype="float16",
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        kv_transfer_config=kv_transfer_config,
        attention_config={"backend": "FLASHINFER"},
    )

    measurements: list[Measurement] = []
    try:
        # Warmup: prime sys prefix-cache + JIT-compile the Triton
        # kernels that fire on first inference. No reuse_plan in
        # extra_args so the PC connector stays out of the way.
        print(f"[bench] r={r}: warmup (sys-only prompt)")
        llm.generate(
            [TokensPrompt(prompt_token_ids=list(sys_padded_tokens))],
            sampling_params=[
                SamplingParams(max_tokens=1, temperature=0.0)
            ],
        )

        for spec in query_specs:
            # ---- Call 1: TTFT (production-shaped prompt, no gold) ----
            sp_ttft = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                extra_args=spec.reuse_params,
            )
            t_start = time.perf_counter()
            _ = llm.generate(
                [TokensPrompt(prompt_token_ids=list(spec.prod_prompt_ids))],
                sampling_params=[sp_ttft],
            )
            t_end = time.perf_counter()
            ttft_ms = (t_end - t_start) * 1000.0

            # ---- Call 2: perplexity (prod + gold, prompt_logprobs=1) ----
            # vLLM's prefix cache covers Call 1's prompt, so this
            # call only forward-passes the gold tail. Logprobs come
            # back for the freshly-computed gold positions.
            sp_ppl = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                prompt_logprobs=1,
                extra_args=spec.reuse_params,
            )
            out = llm.generate(
                [TokensPrompt(prompt_token_ids=list(spec.eval_prompt_ids))],
                sampling_params=[sp_ppl],
            )
            ppl = _compute_perplexity(
                out[0].prompt_logprobs,
                spec.eval_prompt_ids,
                spec.gold_start,
                spec.gold_len,
            )

            ppl_str = f"{ppl:.4f}" if ppl is not None else "N/A"
            print(
                f"[bench] r={r}: {spec.instance_id}: "
                f"TTFT={ttft_ms:7.1f}ms  ppl={ppl_str}"
            )
            measurements.append(
                Measurement(
                    instance_id=spec.instance_id,
                    ttft_ms=ttft_ms,
                    ppl=ppl,
                )
            )
    finally:
        # Hard teardown so the next r value boots into a clean vLLM
        # prefix cache (the whole point of per-r isolation given the
        # scatter-pollutes-prefix-cache issue).
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)  # let the spawned worker process actually die

    return measurements


# ----------------------------- report -----------------------------


def _print_report(
    results: dict[float, list[Measurement]],
    query_specs: list[QuerySpec],
) -> None:
    """Pretty tables: prompt context, TTFT, perplexity, recovery."""
    r_values = sorted(results.keys(), reverse=True)

    # ---- Prompt context ----
    print("\n\n" + "=" * 100)
    print("BENCH RESULTS")
    print("=" * 100)
    print("Prompt makeup per instance (tokens):")
    for spec in query_specs:
        query_len = (
            len(spec.prod_prompt_ids)
            - spec.sys_len_padded
            - spec.chunks_total
        )
        print(
            f"  {spec.instance_id:<16}  "
            f"prod_prompt={len(spec.prod_prompt_ids):>4}  "
            f"(sys={spec.sys_len_padded} + chunks={spec.chunks_total} "
            f"+ query={query_len})  gold={spec.gold_len}"
        )

    # ---- TTFT table ----
    print("\n" + "-" * 100)
    print(
        "TTFT (ms) per (r, instance)  — production-shaped prompt "
        "(sys+chunks+query)"
    )
    print("-" * 100)
    header_cells = [f"r={r}" for r in r_values]
    print(f"  {'instance':<20}" + "".join(f"{c:>12}" for c in header_cells))
    for i, spec in enumerate(query_specs):
        row = f"  {spec.instance_id:<20}"
        for r in r_values:
            m = results[r][i]
            row += f"{m.ttft_ms:>10.1f}  "
        print(row)
    # Per-r averages + delta vs r=1.0 baseline.
    print(f"  {'AVG':<20}", end="")
    for r in r_values:
        avg = sum(m.ttft_ms for m in results[r]) / len(results[r])
        print(f"{avg:>10.1f}  ", end="")
    print()
    if 1.0 in results:
        baseline = sum(m.ttft_ms for m in results[1.0]) / len(results[1.0])
        print(f"  {'Δ vs r=1.0':<20}", end="")
        for r in r_values:
            avg = sum(m.ttft_ms for m in results[r]) / len(results[r])
            pct = (avg - baseline) / baseline * 100 if baseline else 0
            print(f"{pct:>+9.1f}%  ", end="")
        print()

    # ---- Perplexity table ----
    print("\n" + "-" * 100)
    print(
        "Perplexity per (r, instance)  — teacher-forced over gold answer"
    )
    print(
        "  Lower = model assigns higher probability to gold tokens (better)"
    )
    print("-" * 100)
    print(f"  {'instance':<20}" + "".join(f"{c:>12}" for c in header_cells))
    for i, spec in enumerate(query_specs):
        row = f"  {spec.instance_id:<20}"
        for r in r_values:
            m = results[r][i]
            cell = f"{m.ppl:>10.4f}" if m.ppl is not None else f"{'N/A':>10}"
            row += f"{cell}  "
        print(row)
    print(f"  {'AVG':<20}", end="")
    for r in r_values:
        ppls = [m.ppl for m in results[r] if m.ppl is not None]
        if ppls:
            print(f"{sum(ppls) / len(ppls):>10.4f}  ", end="")
        else:
            print(f"{'N/A':>10}  ", end="")
    print()

    # ---- Recovery (intermediate r only) ----
    if 1.0 not in results or 0.0 not in results:
        print(
            "\n[bench] Recovery skipped — need both r=1.0 and r=0.0 in "
            "--r_values to define the gold/stale anchors."
        )
        return
    intermediate = [r for r in r_values if 0.0 < r < 1.0]
    if not intermediate:
        print(
            "\n[bench] Recovery skipped — no intermediate r values "
            "(need r in (0, 1))."
        )
        return

    print("\n" + "-" * 100)
    print("Recovery per (intermediate r, instance)")
    print(
        "  recovery(r) = (ppl_stale - ppl_hybrid(r)) / (ppl_stale - ppl_gold)"
    )
    print(
        "  1.0  hybrid matches gold (full quality recovered "
        "with partial recompute)"
    )
    print(
        "  0.0  hybrid no better than stale (selection didn't help)"
    )
    print(
        "  >1.0 hybrid better than gold (rare; fp16 noise or "
        "favourable tokens)"
    )
    print(
        "  <0   hybrid worse than stale (bad selection strategy)"
    )
    print("-" * 100)
    print(
        f"  {'instance':<20}"
        + "".join(f"{f'r={r}':>12}" for r in intermediate)
    )
    sum_rec: dict[float, float] = defaultdict(float)
    cnt_rec: dict[float, int] = defaultdict(int)
    for i, spec in enumerate(query_specs):
        row = f"  {spec.instance_id:<20}"
        ppl_gold = results[1.0][i].ppl
        ppl_stale = results[0.0][i].ppl
        for r in intermediate:
            ppl_hyb = results[r][i].ppl
            if (
                ppl_gold is None
                or ppl_stale is None
                or ppl_hyb is None
            ):
                row += f"{'N/A':>12}"
                continue
            denom = ppl_stale - ppl_gold
            if abs(denom) < 1e-9:
                # Endpoints collapsed (stale ≈ gold) — recovery undefined.
                row += f"{'N/A':>12}"
                continue
            rec = (ppl_stale - ppl_hyb) / denom
            sum_rec[r] += rec
            cnt_rec[r] += 1
            row += f"{rec:>11.3f} "
        print(row)
    print(f"  {'AVG':<20}", end="")
    for r in intermediate:
        if cnt_rec[r]:
            print(f"{sum_rec[r] / cnt_rec[r]:>11.3f} ", end="")
        else:
            print(f"{'N/A':>12}", end="")
    print()

    # ---- Caveat footer ----
    print("\n" + "=" * 100)
    print("CAVEATS")
    print("=" * 100)
    print(
        "1. TTFT numbers rely on per-r LLM teardown to keep PC's K/V scatter\n"
        "   from leaking through vLLM's prefix cache into later r values."
    )
    print(
        "2. Perplexity averages over ~50 gold tokens, so single-token factual\n"
        "   flips (e.g., \"Yes\" ↔ \"No\") get diluted. Recovery may look\n"
        "   middling while actual generation is catastrophically wrong. Layer\n"
        "   ROUGE / cosine-sim / LLM-judge on top if you need factual checks."
    )


# ----------------------------- main -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 13 bench: TTFT + perplexity + recovery vs selector_r."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to JSONL of PC instances.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HF model id; must have a registered ModelPreset.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="FAISS retrieval top-K per query.",
    )
    parser.add_argument(
        "--r_values",
        type=str,
        default=DEFAULT_R_VALUES,
        help=(
            "Comma-separated SelectFirstR.r values to sweep. Include "
            "both 1.0 and 0.0 to enable recovery computation."
        ),
    )
    parser.add_argument(
        "--store_backend",
        type=str,
        default="redis",
        choices=["memory", "redis"],
        help=(
            "PC store backend. 'redis' is recommended so the encoder "
            "runs only once and all r values share the warm cache."
        ),
    )
    parser.add_argument(
        "--store_url",
        type=str,
        default=DEFAULT_REDIS_URL,
        help="Redis URL when --store_backend=redis.",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=DEFAULT_EMBEDDER,
        help="sentence-transformers model id (used by RAGIndex).",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="vLLM cache block_size (must match PC store config).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization (each LLM boot).",
    )
    args = parser.parse_args()

    r_values = [float(x) for x in args.r_values.split(",") if x.strip()]
    if not r_values:
        raise SystemExit("--r_values is empty")

    # ----- 1. Load JSONL + build index -----
    with open(args.data, encoding="utf-8") as f:
        instances = [json.loads(line) for line in f if line.strip()]
    print(f"[bench] loaded {len(instances)} instances from {args.data}")
    print(f"[bench] r values: {r_values}")

    preset = preset_for(args.model)
    print(
        f"[bench] target model = {preset.hf_id}  "
        f"({preset.num_layers} layers, {preset.num_kv_heads} KV heads, "
        f"head_dim={preset.head_dim})"
    )

    index = RAGIndex(
        preset=preset,
        embedder_model_id=args.embedder,
        block_size=args.block_size,
        device="cuda",
        store_backend=args.store_backend,
        store_url=args.store_url,
    )
    index.ingest_instances(instances)
    index.build_faiss()

    # ----- 2. Pre-compute everything we need per query -----
    query_specs = _build_query_specs(instances, index, args)

    # Sys is the same string across all sample instances; padded once.
    tokenizer = index.tokenizer
    sys_padded_tokens = _pad_tokens_to_block_size(
        tokenizer.encode(instances[0]["sys"], add_special_tokens=False),
        tokenizer,
        args.block_size,
    )

    # Capture the kv_store before freeing the encoder/embedder — needed
    # for the memory-backend pickle bridge.
    kv_store = index.kv_store
    index.free_index_models()

    # ----- 3. Per-r bench loop -----
    if args.store_backend == "memory":
        # Pickle bridge stays open across every LLM boot so each
        # worker process can unpickle the same store on init.
        outer_ctx = _pc_test_bind_storage(kv_store)
    else:
        outer_ctx = contextlib.nullcontext()

    results: dict[float, list[Measurement]] = {}
    with outer_ctx:
        for r in r_values:
            results[r] = _bench_for_r(
                r, query_specs, sys_padded_tokens, args
            )

    # ----- 4. Report -----
    _print_report(results, query_specs)


if __name__ == "__main__":
    main()
