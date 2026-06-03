# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal check: gold-answer perplexity under PersonalContext KV reuse.

WHY
---
ROUGE/cosine came out flat across r. Perplexity over the gold answer is a
finer, *generation-free* quality metric — but vLLM disables KV reuse when you
request ``prompt_logprobs`` (it sets ``skip_reading_prefix_cache=True``). The
escape hatch is to pass ``skip_reading_prefix_cache=False`` explicitly; the
documented caveat (reused tokens have no computed logits) bites only the CHUNK
span, while the GOLD ANSWER is appended AFTER the query — freshly prefilled, so
its logits are real and reflect the reused-KV context.

This verifies that empirically on vanilla vs r=0.5 over a few instances:

  PROBE 1 (PC stayed active): with the memory backend, r=0.5 TTFT should be
          clearly BELOW vanilla. If it equals vanilla, prompt_logprobs
          collapsed reuse to full prefill.
  PROBE 2 (logits usable):    answer-span prompt_logprobs are present & finite,
          giving a sane PPL, with r=0.5 PPL close to vanilla.

Memory backend by default — it's the only backend where "PC active" reads as
"r=0.5 faster" (mmap's lookup overhead would make active runs *slower* than
vanilla on these short prompts, inverting the probe).

Throwaway / underscore-prefixed; delete after the question is settled.

Run:
    .venv/bin/python examples/personal_context/_verify_ppl.py --n 5
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.personal_context.chunk_encoder_hf import preset_for  # noqa: E402
from examples.personal_context.rag_demo import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_EMBEDDER,
    RAGIndex,
    _pad_tokens_to_block_size,
    _pc_test_bind_storage,
    build_prompt_and_plan,
)


def _answer_ppl(out, answer_start, answer_token_ids):
    """exp(-mean logprob) over the appended gold-answer tokens.

    Returns ``(ppl, note)``. ``ppl`` is None when any answer-span logprob is
    missing — which is exactly the failure mode we're probing for.
    """
    pls = out.prompt_logprobs
    if pls is None:
        return None, "prompt_logprobs is None"
    logps = []
    for offset, tok in enumerate(answer_token_ids):
        i = answer_start + offset
        if i >= len(pls):
            return None, f"answer idx {i} >= prompt_logprobs len {len(pls)}"
        entry = pls[i]
        if entry is None:
            return None, f"None logprob at answer pos {offset} (prompt idx {i})"
        lp = entry.get(tok)
        if lp is None:
            return None, f"actual token {tok} absent at answer pos {offset}"
        logps.append(lp.logprob)
    if not logps:
        return None, "empty answer span"
    return math.exp(-sum(logps) / len(logps)), f"{len(logps)} answer tokens"


def _ppl_sampling(reuse_params=None):
    """SamplingParams kwargs: 1-token gen + prompt_logprobs + escape hatch."""
    kw = dict(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
        # Escape hatch: keep prefix-cache / PC reuse ON despite prompt_logprobs
        # (default would force skip_reading_prefix_cache=True -> reuse off).
        skip_reading_prefix_cache=False,
    )
    if reuse_params is not None:
        kw["extra_args"] = reuse_params
    return kw


def _run_mode(r, specs, sys_padded_tokens, tok, args):
    """Boot one LLM (vanilla if r is None, else PC@SelectFirstR=r), score
    answer-PPL + TTFT per spec with prompt_logprobs, tear down. Caller holds
    the storage-bind context (memory backend)."""
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.inputs import TokensPrompt

    is_vanilla = r is None
    label = "vanilla" if is_vanilla else f"r={r}"

    kv_transfer_config = None
    if not is_vanilla:
        extra: dict = {"selector": {"type": "SelectFirstR", "r": r}}
        if args.store_backend in ("redis", "mmap"):
            extra["store_backend"] = args.store_backend
            extra["store_url"] = args.store_url
        kv_transfer_config = KVTransferConfig(
            kv_connector="PersonalContextKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config=extra,
        )

    # Per-mode dump file for the in-runner answer-logprob dump. Set the env
    # var BEFORE LLM(): the spawned worker inherits it at spawn time.
    dump_path = f"/tmp/pc_ppl_{label.replace('=', '').replace('.', 'p')}.jsonl"
    os.environ["VLLM_PC_PPL_DUMP"] = dump_path
    open(dump_path, "w", encoding="utf-8").close()  # truncate per mode

    print(f"\n{'=' * 72}\n[verify] booting LLM: {label}  (dump={dump_path})"
          f"\n{'=' * 72}")
    llm = LLM(
        model=args.model,
        dtype="float16",
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        kv_transfer_config=kv_transfer_config,
        attention_config={"backend": "FLASHINFER"},
        disable_log_stats=False,
    )

    # raw rows: (id, ttft, broken_ppl, note, num_prompt, answer_len)
    raw: list[tuple] = []
    try:
        # Warmup 1: sys-only -> caches sys prefix (so real requests get
        # num_computed_tokens=sys_len and PC can place chunks) + JITs kernels.
        llm.generate(
            [TokensPrompt(prompt_token_ids=list(sys_padded_tokens))],
            sampling_params=[SamplingParams(max_tokens=1, temperature=0.0)],
        )
        # Warmup 2 (reuse path): one PPL-shaped call so the first measured
        # TTFT isn't a one-time JIT outlier, and to confirm the
        # prompt_logprobs + reuse path doesn't crash before the loop.
        if not is_vanilla:
            llm.generate(
                [TokensPrompt(prompt_token_ids=list(specs[0]["full_ids"]))],
                sampling_params=[
                    SamplingParams(**_ppl_sampling(specs[0]["reuse_params"]))
                ],
            )

        for s in specs:
            sp = _ppl_sampling(None if is_vanilla else s["reuse_params"])
            np_len, alen = len(s["full_ids"]), len(s["answer_ids"])
            try:
                out = llm.generate(
                    [TokensPrompt(prompt_token_ids=list(s["full_ids"]))],
                    sampling_params=[SamplingParams(**sp)],
                )[0]
            except Exception as e:  # noqa: BLE001
                raw.append((s["id"], None, None, f"GENERATE RAISED: {e!r}",
                            np_len, alen))
                continue

            m = out.metrics
            ttft = (
                m.first_token_latency * 1000.0
                if m is not None and m.first_token_latency is not None
                else None
            )
            # broken_ppl: read straight off RequestOutput.prompt_logprobs (the
            # head-sliced path) — kept for contrast; corrected_ppl below comes
            # from the in-runner tail-slice dump.
            broken_ppl, note = _answer_ppl(
                out, s["answer_start"], s["answer_ids"]
            )
            raw.append((s["id"], ttft, broken_ppl, note, np_len, alen))
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # Corrected answer-PPL from the in-runner dump: the dumped token_logprobs
    # are the last K suffix tokens (tail-sliced past any recompute hiddens), so
    # the answer is the last `answer_len` of them. Match dump lines to specs by
    # num_prompt (last write wins; warmup of spec[0] shares its num_prompt but
    # is identical).
    by_np: dict[int, list[float]] = {}
    try:
        with open(dump_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                by_np[int(d["num_prompt"])] = d["token_logprobs"]
    except FileNotFoundError:
        pass

    rows = []
    for sid, ttft, broken_ppl, note, np_len, alen in raw:
        tlp = by_np.get(np_len)
        if tlp and len(tlp) >= alen and alen > 0:
            ans = tlp[-alen:]
            corrected_ppl = math.exp(-sum(ans) / len(ans))
        else:
            corrected_ppl = None
        rows.append((sid, ttft, broken_ppl, corrected_ppl, note))
    return label, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify gold-answer PPL under PC.")
    ap.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--n", type=int, default=5, help="instances to verify")
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument(
        "--store_backend", default="memory",
        choices=["memory", "redis", "mmap"],
        help="memory is cleanest (PC active => r=0.5 faster than vanilla).",
    )
    ap.add_argument("--store_url", default=None)
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        instances = [json.loads(line) for line in f if line.strip()][: args.n]
    print(f"[verify] {len(instances)} instances from {args.data}; "
          f"backend={args.store_backend}; top_k={args.top_k}")

    preset = preset_for(args.model)
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
    tok = index.tokenizer

    # Build prod prompt (sys+chunks+query) per instance, then append the gold
    # answer. answer_start = where the appended answer begins in the full
    # sequence. (We concatenate two separate encodings; the Answer:|answer
    # boundary token may differ slightly from a single-string encoding, but
    # vanilla and r=0.5 score the SAME tokens, so the comparison is valid.)
    specs = []
    for inst in instances:
        retrieved = index.retrieve(inst["query"], top_k=args.top_k)
        prod_ids, reuse_params, _, _ = build_prompt_and_plan(
            tok, inst["sys"], retrieved, inst["query"],
            block_size=args.block_size,
        )
        answer_ids = tok.encode(" " + inst["answer"], add_special_tokens=False)
        specs.append(dict(
            id=inst["id"],
            reuse_params=reuse_params,
            full_ids=list(prod_ids) + list(answer_ids),
            answer_start=len(prod_ids),
            answer_ids=answer_ids,
        ))

    sys_padded = _pad_tokens_to_block_size(
        tok.encode(instances[0]["sys"], add_special_tokens=False),
        tok, args.block_size,
    )

    kv_store = index.kv_store
    if getattr(index, "encoder", None) is not None:
        del index.encoder
        index.encoder = None
        gc.collect()
        torch.cuda.empty_cache()

    outer = (
        _pc_test_bind_storage(kv_store)
        if args.store_backend == "memory" else contextlib.nullcontext()
    )
    results = {}
    with outer:
        for r in (None, 0.5):
            label, rows = _run_mode(r, specs, sys_padded, tok, args)
            results[label] = rows

    # ---------------- report ----------------
    print(f"\n{'=' * 72}\n[verify] RESULTS  (TTFT ms; cPPL = corrected "
          f"answer-PPL from in-runner tail-slice dump)\n{'=' * 72}")
    print(f"  {'instance':<16}{'TTFT van':>10}{'TTFT r0.5':>11}"
          f"{'cPPL van':>10}{'cPPL r0.5':>11}  note(r0.5)")
    van = {row[0]: row for row in results["vanilla"]}
    r05 = {row[0]: row for row in results["r=0.5"]}
    tv, tr, vc, rc, rb = [], [], [], [], []

    def fmt(x):
        return f"{x:.1f}" if x is not None else "--"

    for s in specs:
        sid = s["id"]
        _, vt, _vbp, vcp, _ = van[sid]
        _, rt, rbp, rcp, rnote = r05[sid]
        for acc, val in ((tv, vt), (tr, rt), (vc, vcp), (rc, rcp), (rb, rbp)):
            if val is not None:
                acc.append(val)
        print(f"  {sid:<16}{fmt(vt):>10}{fmt(rt):>11}"
              f"{fmt(vcp):>10}{fmt(rcp):>11}  {rnote}")

    def med(x):
        return st.median(x) if x else float("nan")

    print(f"  {'MEDIAN':<16}{med(tv):>10.1f}{med(tr):>11.1f}"
          f"{med(vc):>10.2f}{med(rc):>11.2f}")

    # ---------------- verdicts ----------------
    print(f"\n{'-' * 72}\nVERDICTS\n{'-' * 72}")
    if tv and tr:
        mv, mr = med(tv), med(tr)
        if mr < mv * 0.9:
            print(f"  PROBE 1 PASS: r=0.5 TTFT {mr:.1f}ms < vanilla {mv:.1f}ms "
                  "-> PC stayed ACTIVE under prompt_logprobs.")
        else:
            print(f"  PROBE 1 FAIL: r=0.5 TTFT {mr:.1f}ms ~>= vanilla "
                  f"{mv:.1f}ms -> reuse likely COLLAPSED (memory backend "
                  "expects r << vanilla).")
    else:
        print("  PROBE 1 N/A: missing TTFT (see notes above).")

    n_ok = sum(1 for row in results["r=0.5"] if row[3] is not None)
    if n_ok == len(specs) and rc:
        print(f"  PROBE 2 PASS: {n_ok}/{len(specs)} r=0.5 corrected answer-PPLs "
              f"finite (median {med(rc):.2f}; vanilla {med(vc):.2f}).")
    else:
        print(f"  PROBE 2 FAIL: only {n_ok}/{len(specs)} r=0.5 corrected "
              "answer-PPLs finite -> in-runner dump missing (check stderr "
              "[PC_PPL_DUMP] and GENERATE RAISED notes).")

    # Contrast with the broken head-sliced path (RequestOutput.prompt_logprobs).
    if rb:
        print(f"  CONTRAST: broken head-slice r=0.5 PPL median = {med(rb):.1f} "
              f"vs corrected {med(rc):.2f} -> the tail-slice dump is what makes "
              "it usable.")

    print("\n  Sanity: a well-supported answer scores PPL ~1.5-15; corrected "
          "r=0.5 should sit close to vanilla. PROBE1+PROBE2 PASS and r=0.5 "
          "cPPL ~ vanilla => the in-runner dump gives trustworthy PPL.")


if __name__ == "__main__":
    main()
