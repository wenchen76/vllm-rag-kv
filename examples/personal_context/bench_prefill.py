# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 13 bench: TTFT + generation-quality + recovery vs selector_r.

Per ``(selector_r, query)`` we make two ``llm.generate`` calls:

    1. **Call 1 — TTFT measurement** (``max_tokens=1``):
       prompt is the *production* ``sys + chunks + query``. Wall-clock
       around the call is the latency a real user would see. PC's
       sparse-Q activates here at the configured ``r``, leaving the
       r-specific chunk K/V in vLLM's prefix cache.

    2. **Call 2 — generation for quality scoring** (``max_tokens=128``):
       same prompt as Call 1. vLLM's prefix cache hits the full
       prefill, so this call only runs decode against the r-specific
       cached K/V. Generated text reflects the r-specific cache state
       end-to-end.

Quality is scored on the generated text via two complementary metrics:

    - **ROUGE-L** (lexical): catches single-token factual flips
      (``"Yes"`` ↔ ``"No"``, ``"May 16"`` ↔ ``"May 6"``) that
      perplexity averaging would dilute. Costs ~ms.
    - **Cosine sim** on sentence-transformer embeddings (semantic):
      tolerant to paraphrasing, catches "totally off topic" cases
      ROUGE might miss. Reuses the existing demo embedder.

Recovery at each intermediate r (where 1.0 = full quality recovered,
0.0 = no better than stale):

    recovery_metric(r) = (q(r) - q_stale) / (q_gold - q_stale)

with ``q`` being either rouge_l or cos_sim, ``q_stale`` at r=0.0
(worst), ``q_gold`` at r=1.0 (best).

WHY NOT PERPLEXITY
------------------
The natural metric is perplexity over the gold answer
(``prompt_logprobs=1`` on ``sys+chunks+query+gold``). It doesn't
work: vLLM disables prefix caching for ``prompt_logprobs`` requests
(cached blocks have no stored logprobs, so the engine must fresh-
prefill the whole prompt). PC's placement validation then sees
``num_computed_tokens=0`` and refuses to activate, so all r values
collapse to vanilla full prefill — identical perplexity, useless
recovery. Verified empirically (see commit history).

Generation-based quality lets PC activate normally (no
prompt_logprobs in the request), at the cost of a slower second
call.

CAVEAT — prefix cache pollution
-------------------------------
PC's scatter writes chunk K/V into the worker's paged cache; vLLM's
prefix cache then captures those blocks. A second request with the
same prompt would inherit the first request's K/V state regardless
of its own r. This bench tears down the LLM between r values to
keep that contamination from leaking across r. Static-r production
is unaffected.

Usage:
    redis-server --daemonize yes --port 6379  # or docker
    .venv/bin/python examples/personal_context/bench_prefill.py \\
        --r_values 1.0,0.75,0.5,0.25,0.0
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch


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
DEFAULT_GEN_TOKENS = 128


# ----------------------------- data classes -----------------------------


@dataclasses.dataclass
class QuerySpec:
    """Pre-computed per-instance bench inputs."""

    instance_id: str
    query_text: str
    gold_text: str
    prod_prompt_ids: list[int]    # sys + chunks + query (PC-active prompt)
    reuse_params: dict             # extra_args for the connector
    sys_len_padded: int
    chunks_total: int


@dataclasses.dataclass
class Measurement:
    instance_id: str
    ttft_ms: float
    generated_text: str
    rouge_l: float
    cos_sim: float


# ----------------------------- helpers -----------------------------


def _build_query_specs(
    instances: list[dict],
    index: RAGIndex,
    args: argparse.Namespace,
) -> list[QuerySpec]:
    """Pre-build the prompt + reuse_plan for every instance.

    Done up-front so the per-r loop only does inference, and so any
    tokenisation surprises surface before paying the LLM boot cost.
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
        chunks_total = sum(len(e.token_ids) for e in retrieved)

        specs.append(
            QuerySpec(
                instance_id=inst["id"],
                query_text=inst["query"],
                gold_text=inst["answer"],
                prod_prompt_ids=prod_prompt_ids,
                reuse_params=reuse_params,
                sys_len_padded=sys_len_padded,
                chunks_total=chunks_total,
            )
        )
    return specs


def _score_generation(
    gold_text: str,
    gen_text: str,
    embedder,
    rouge_scorer,
) -> tuple[float, float]:
    """Compute (ROUGE-L F1, cosine similarity) of gen against gold."""
    rouge_l = rouge_scorer.score(gold_text, gen_text)["rougeL"].fmeasure

    # normalize_embeddings=True so we can use plain dot product as
    # cosine similarity (avoids an explicit division).
    vecs = embedder.encode(
        [gold_text, gen_text],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    cos_sim = float((vecs[0] * vecs[1]).sum())
    return float(rouge_l), cos_sim


# ----------------------------- per-r bench loop -----------------------------


def _bench_for_r(
    r: float,
    query_specs: list[QuerySpec],
    sys_padded_tokens: list[int],
    embedder,
    rouge_scorer,
    args: argparse.Namespace,
) -> list[Measurement]:
    """Boot LLM at this r, warmup, run 2-call pattern per query, tear down."""
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
        # Warmup: sys-only prompt with no reuse_plan, primes vLLM
        # prefix-cache for sys and triggers Triton kernel JIT compile.
        print(f"[bench] r={r}: warmup (sys-only prompt)")
        llm.generate(
            [TokensPrompt(prompt_token_ids=list(sys_padded_tokens))],
            sampling_params=[
                SamplingParams(max_tokens=1, temperature=0.0)
            ],
        )

        for spec in query_specs:
            # ---- Call 1: TTFT on production prompt ----
            # PC connector activates here; sparse-Q at this r writes
            # the r-specific chunk K/V into vLLM's paged cache, which
            # the prefix-cache then captures for Call 2 to consume.
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

            # ---- Call 2: generation for quality scoring ----
            # Same prompt → vLLM prefix-cache hits the full prefill;
            # this call effectively just decodes 128 tokens against
            # the r-specific cached K/V from Call 1. PC connector
            # logs a placement-overflow warning (harmless — there are
            # 0 new chunk tokens to match here).
            sp_gen = SamplingParams(
                max_tokens=args.gen_tokens,
                temperature=0.0,
                extra_args=spec.reuse_params,
            )
            out = llm.generate(
                [TokensPrompt(prompt_token_ids=list(spec.prod_prompt_ids))],
                sampling_params=[sp_gen],
            )
            gen_text = out[0].outputs[0].text

            rouge_l, cos_sim = _score_generation(
                spec.gold_text, gen_text, embedder, rouge_scorer
            )

            print(
                f"[bench] r={r}: {spec.instance_id}: "
                f"TTFT={ttft_ms:7.1f}ms  "
                f"rougeL={rouge_l:.3f}  cos={cos_sim:.3f}  "
                f"| {gen_text[:80]!r}{'...' if len(gen_text) > 80 else ''}"
            )
            measurements.append(
                Measurement(
                    instance_id=spec.instance_id,
                    ttft_ms=ttft_ms,
                    generated_text=gen_text,
                    rouge_l=rouge_l,
                    cos_sim=cos_sim,
                )
            )
    finally:
        # Hard teardown so the next r value boots into a clean vLLM
        # prefix cache (else the first r's K/V leaks into all later r).
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)  # let the spawned worker actually die

    return measurements


# ----------------------------- report -----------------------------


def _print_report(
    results: dict[float, list[Measurement]],
    query_specs: list[QuerySpec],
) -> None:
    """Pretty tables + recovery + raw generation dump."""
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
            f"+ query={query_len})  gold={len(spec.gold_text.split()):>3} words"
        )

    header_cells = [f"r={r}" for r in r_values]

    # ---- TTFT table ----
    print("\n" + "-" * 100)
    print(
        "TTFT (ms) per (r, instance)  — production-shaped prompt "
        "(sys+chunks+query, max_tokens=1)"
    )
    print("-" * 100)
    print(f"  {'instance':<20}" + "".join(f"{c:>12}" for c in header_cells))
    for i, spec in enumerate(query_specs):
        row = f"  {spec.instance_id:<20}"
        for r in r_values:
            row += f"{results[r][i].ttft_ms:>10.1f}  "
        print(row)
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

    # ---- ROUGE-L table ----
    _print_quality_table(
        "ROUGE-L F1 per (r, instance)  — lexical overlap with gold",
        "Higher = generated text shares more n-grams with gold (better)",
        results,
        query_specs,
        r_values,
        header_cells,
        getter=lambda m: m.rouge_l,
    )

    # ---- Cosine sim table ----
    _print_quality_table(
        "Cosine sim per (r, instance)  — embedding similarity to gold",
        "Higher = closer in semantic embedding space (better)",
        results,
        query_specs,
        r_values,
        header_cells,
        getter=lambda m: m.cos_sim,
    )

    # ---- Recovery ----
    _print_recovery(results, query_specs, r_values)

    # ---- Generation dump ----
    print("\n" + "=" * 100)
    print("GENERATION DUMP")
    print("=" * 100)
    for i, spec in enumerate(query_specs):
        print(f"\n[{spec.instance_id}]")
        print(f"  Query: {spec.query_text}")
        print(f"  Gold:  {spec.gold_text}")
        for r in r_values:
            m = results[r][i]
            print(
                f"\n  r={r}  (rougeL={m.rouge_l:.3f}, cos={m.cos_sim:.3f})"
            )
            # Indent the generation for readability.
            for line in m.generated_text.strip().splitlines() or [""]:
                print(f"    {line}")

    # ---- Caveat footer ----
    print("\n" + "=" * 100)
    print("CAVEATS")
    print("=" * 100)
    print(
        "1. TTFT numbers rely on per-r LLM teardown to keep PC's K/V scatter\n"
        "   from leaking through vLLM's prefix cache into later r values."
    )
    print(
        "2. Perplexity is NOT measured — vLLM's prompt_logprobs disables\n"
        "   prefix caching, which prevents PC from activating, collapsing\n"
        "   all r values to vanilla. ROUGE / cosine on generations work\n"
        "   because no prompt_logprobs is requested."
    )
    print(
        "3. ROUGE catches lexical / single-token errors that perplexity\n"
        "   averaging dilutes. Cosine catches paraphrasing-but-correct\n"
        "   cases ROUGE penalises. Use both, not either alone."
    )


def _print_quality_table(
    title: str,
    subtitle: str,
    results: dict[float, list[Measurement]],
    query_specs: list[QuerySpec],
    r_values: list[float],
    header_cells: list[str],
    getter,
) -> None:
    print("\n" + "-" * 100)
    print(title)
    print(f"  {subtitle}")
    print("-" * 100)
    print(f"  {'instance':<20}" + "".join(f"{c:>12}" for c in header_cells))
    for i, spec in enumerate(query_specs):
        row = f"  {spec.instance_id:<20}"
        for r in r_values:
            row += f"{getter(results[r][i]):>10.3f}  "
        print(row)
    print(f"  {'AVG':<20}", end="")
    for r in r_values:
        avg = sum(getter(m) for m in results[r]) / len(results[r])
        print(f"{avg:>10.3f}  ", end="")
    print()


def _print_recovery(
    results: dict[float, list[Measurement]],
    query_specs: list[QuerySpec],
    r_values: list[float],
) -> None:
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
        "  recovery_metric(r) = (q(r) - q_stale) / (q_gold - q_stale)"
    )
    print(
        "  1.0  hybrid matches gold (full quality with partial recompute)"
    )
    print("  0.0  hybrid no better than stale (selection didn't help)")
    print("  >1.0 hybrid better than gold (rare; lucky paraphrasing)")
    print("  <0   hybrid worse than stale (bad strategy)")
    print(
        "  N/A  endpoints collapsed (q_stale ≈ q_gold) — recovery undefined"
    )
    print("-" * 100)

    for metric_name, getter in [
        ("ROUGE-L", lambda m: m.rouge_l),
        ("Cosine ", lambda m: m.cos_sim),
    ]:
        print(f"\n  [{metric_name}]")
        print(
            f"  {'instance':<20}"
            + "".join(f"{f'r={r}':>12}" for r in intermediate)
        )
        sum_rec: dict[float, float] = defaultdict(float)
        cnt_rec: dict[float, int] = defaultdict(int)
        for i, spec in enumerate(query_specs):
            row = f"  {spec.instance_id:<20}"
            q_gold = getter(results[1.0][i])
            q_stale = getter(results[0.0][i])
            for r in intermediate:
                q_hyb = getter(results[r][i])
                denom = q_gold - q_stale
                if abs(denom) < 1e-9:
                    row += f"{'N/A':>12}"
                    continue
                rec = (q_hyb - q_stale) / denom
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


# ----------------------------- main -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 13 bench: TTFT + generation quality (ROUGE-L, cosine) "
            "+ recovery vs selector_r."
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
            "Comma-separated SelectFirstR.r values. Include 1.0 and "
            "0.0 to enable recovery computation."
        ),
    )
    parser.add_argument(
        "--gen_tokens",
        type=int,
        default=DEFAULT_GEN_TOKENS,
        help="Max generation length (Call 2). Should comfortably exceed "
        "gold answer length so ROUGE recall isn't truncated.",
    )
    parser.add_argument(
        "--store_backend",
        type=str,
        default="redis",
        choices=["memory", "redis"],
        help=(
            "PC store backend. 'redis' is recommended so the encoder "
            "runs only once and all r values share warm cache."
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
        help="sentence-transformers model id (used by index + scoring).",
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
    print(f"[bench] gen_tokens (Call 2 max_tokens): {args.gen_tokens}")

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

    tokenizer = index.tokenizer
    sys_padded_tokens = _pad_tokens_to_block_size(
        tokenizer.encode(instances[0]["sys"], add_special_tokens=False),
        tokenizer,
        args.block_size,
    )

    # Keep embedder alive for scoring (it's small, ~80MB) but release
    # the heavy HF encoder model before booting any LLM.
    kv_store = index.kv_store
    embedder = index.embedder
    if hasattr(index, "encoder") and index.encoder is not None:
        del index.encoder
        index.encoder = None
        gc.collect()
        torch.cuda.empty_cache()

    # ----- 3. Build the rouge scorer (cheap, no model) -----
    try:
        from rouge_score import rouge_scorer as _rouge_scorer_mod
    except ImportError as e:
        raise SystemExit(
            "rouge-score is required. Install via "
            "`uv pip install \"rouge-score>=0.1.2\"` or "
            "`uv pip install -e \".[personal-context]\"`."
        ) from e
    rouge_scorer = _rouge_scorer_mod.RougeScorer(
        ["rougeL"], use_stemmer=True
    )

    # ----- 4. Per-r bench loop -----
    if args.store_backend == "memory":
        outer_ctx = _pc_test_bind_storage(kv_store)
    else:
        outer_ctx = contextlib.nullcontext()

    results: dict[float, list[Measurement]] = {}
    with outer_ctx:
        for r in r_values:
            results[r] = _bench_for_r(
                r,
                query_specs,
                sys_padded_tokens,
                embedder,
                rouge_scorer,
                args,
            )

    # ----- 5. Report -----
    _print_report(results, query_specs)


if __name__ == "__main__":
    main()
