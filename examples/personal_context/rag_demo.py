# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end RAG demo over personal-context data, with KV reuse.

Pipeline:

  Offline ingestion (per chunk in the JSONL corpus):
      1. tokenize chunk text (truncate to a block_size multiple)
      2. embed chunk text with sentence-transformers → vector
      3. encode chunk tokens with HFChunkEncoder → list[KVBlock]
      4. push (chunk_id, vector) into FAISS, (block_hash, kv_block) into PC store

  Online query loop (per instance):
      1. embed query → search FAISS top-K → list[IndexEntry]
      2. report retrieval quality vs the JSONL's labelled "gold" chunks
         (an indirect sanity check that the embedder is doing real work)
      3. build augmented prompt = sys_tokens + concat(chunk_tokens) + query_tokens
      4. build reuse_plan over retrieved chunks
      5. vLLM.generate(prompt, reuse_plan)
      6. print model output alongside the JSONL's labelled "answer"

Important caveats this demo intentionally does NOT solve:

  - **Prefix-cache priming for the system prompt.** The first query
    hits PC's placement validation (chunk tokens don't sit at offset
    0 — the system prompt does) and falls back to full prefill. From
    the second query onward (same sys prompt), vLLM's prefix cache
    has the sys block(s) and PC reuse activates cleanly. This is the
    expected production pattern (sys cached) but the demo doesn't
    pad / warmup explicitly.

  - **R<1.0 quality.** Default selector is SelectFirstR(r=1.0) — full
    recompute, equivalent to vanilla. To actually save prefill work
    you need r<1.0; quality may degrade. Override with --selector_r.

  - **Tokenisation drift.** Chunks are tokenised once and stored;
    when concatenated into a prompt they may not tokenise the same
    way as the raw concatenated text would (BPE merges across
    boundaries). For correctness this is required (chunk hashes must
    match prompt tokens at that offset); it costs a few "wasted"
    tokens.

Usage:
    .venv/bin/python examples/personal_context/rag_demo.py
    .venv/bin/python examples/personal_context/rag_demo.py --top_k 3 --selector_r 1.0
    .venv/bin/python examples/personal_context/rag_demo.py \
        --data examples/personal_context/my_data.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


# ----------------------- imports that bring in heavy deps -----------------------

# Repo-root on path so this script can import its sibling encoder.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.personal_context.chunk_encoder_hf import (  # noqa: E402
    QWEN_2_5_0_5B_INSTRUCT,
    HFChunkEncoder,
    store_config_for,
)
from vllm.v1.personal_context import InMemoryStorage  # noqa: E402


DEFAULT_DATA_PATH = Path(__file__).parent / "sample_data.jsonl"
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------- index entry -----------------------


@dataclasses.dataclass
class IndexEntry:
    """One chunk in the global FAISS + PC store index."""

    global_id: str
    """Unique across all instances: ``f'{instance_id}__{chunk_name}'``."""

    instance_id: str
    chunk_name: str
    source: str
    gold_rank: int
    """The labelled retrieval rank for the chunk's own instance query.
    Used only to report retrieval recall after FAISS search."""

    text: str
    token_ids: list[int]
    old_pos_start: int


# ----------------------- index -----------------------


class RAGIndex:
    """Bundles the encoder, embedder, FAISS index, and PC KV store.

    Holds everything index-side in one object so VRAM cleanup
    (``free_index_models``) before the vLLM engine starts is a single
    method call.
    """

    def __init__(
        self,
        preset,
        embedder_model_id: str = DEFAULT_EMBEDDER,
        block_size: int = 16,
        device: str = "cuda",
    ):
        try:
            import faiss  # noqa: F401  (validate availability)
        except ImportError as e:
            raise ImportError(
                "RAGIndex requires faiss-cpu (or faiss-gpu). "
                "Install via `uv pip install faiss-cpu`."
            ) from e
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "RAGIndex requires sentence-transformers. "
                "Install via `uv pip install sentence-transformers`."
            ) from e

        self.preset = preset
        self.block_size = block_size
        self.device = device

        print(f"[index] loading HF encoder: {preset.hf_id}")
        self.encoder = HFChunkEncoder(
            preset=preset, device=device, dtype=torch.float16
        )
        self.tokenizer = self.encoder.tokenizer

        print(f"[index] loading embedder: {embedder_model_id}")
        self.embedder = SentenceTransformer(embedder_model_id, device=device)

        self.kv_store = InMemoryStorage(
            store_config_for(preset, block_size=block_size)
        )
        self.entries: list[IndexEntry] = []
        self.faiss_index: Any = None

    # ----- ingestion -----

    def ingest_instances(self, instances: list[dict]) -> None:
        for inst in instances:
            self._ingest_instance(inst)

    def _ingest_instance(self, instance: dict) -> None:
        instance_id = instance["id"]
        for chunk_name, chunk_data in instance["chunks"].items():
            self._add_chunk(instance_id, chunk_name, chunk_data)

    def _add_chunk(
        self, instance_id: str, chunk_name: str, chunk_data: dict
    ) -> None:
        text = chunk_data["text"]
        raw_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(raw_tokens) == 0:
            print(
                f"[skip] {instance_id}/{chunk_name}: empty after tokenisation"
            )
            return

        # Pad up to the next block_size boundary rather than truncating.
        # Truncation drops trailing tokens — for short chunks that can
        # be 30–50% of the content (e.g., a 30-token note → 16). The
        # padding tokens become part of the stored chunk hash and of
        # every prompt that includes this chunk; both costs are paid in
        # exchange for full content preservation (high-value for short
        # personal-context chunks where every clause carries info).
        tokens = _pad_tokens_to_block_size(
            raw_tokens, self.tokenizer, self.block_size
        )
        pad_added = len(tokens) - len(raw_tokens)
        if pad_added > 0:
            print(
                f"[pad ] {instance_id}/{chunk_name}: "
                f"raw={len(raw_tokens)} → padded={len(tokens)} "
                f"(+{pad_added} newline tok)"
            )

        global_id = f"{instance_id}__{chunk_name}"

        # Encode → push every block into the PC store.
        _, block_entries = self.encoder.encode_chunk(
            tokens, old_pos_start=0, block_size=self.block_size
        )
        for block_hash, kv_block in block_entries:
            self.kv_store.put(block_hash, kv_block)

        self.entries.append(
            IndexEntry(
                global_id=global_id,
                instance_id=instance_id,
                chunk_name=chunk_name,
                source=chunk_data.get("source", ""),
                gold_rank=int(chunk_data.get("retrieval_rank", 0)),
                text=text,
                token_ids=list(tokens),
                old_pos_start=0,
            )
        )

    # ----- FAISS -----

    def build_faiss(self) -> None:
        import faiss

        if not self.entries:
            raise RuntimeError("no entries to index; call ingest_instances first")
        texts = [e.text for e in self.entries]
        vectors = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        dim = int(vectors.shape[1])
        index = faiss.IndexFlatIP(dim)  # inner product = cosine after L2-norm
        index.add(vectors)
        self.faiss_index = index
        print(f"[index] FAISS built: {len(self.entries)} entries × dim={dim}")

    def retrieve(self, query: str, top_k: int = 4) -> list[IndexEntry]:
        if self.faiss_index is None:
            raise RuntimeError("call build_faiss() before retrieve()")
        vec = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        _, idxs = self.faiss_index.search(vec, top_k)
        return [self.entries[i] for i in idxs[0]]

    # ----- cleanup -----

    def free_index_models(self) -> None:
        """Free encoder + embedder VRAM before the vLLM engine boots.

        ``tokenizer`` is the only object kept (small, CPU-only) because
        the prompt builder still needs it. ``kv_store`` survives because
        the worker will read from it via the pickle bind path.
        """
        del self.encoder
        del self.embedder
        torch.cuda.empty_cache()


# ----------------------- prompt + reuse plan assembly -----------------------


def _pad_tokens_to_block_size(
    tokens: list[int],
    tokenizer,
    block_size: int,
) -> list[int]:
    """Append newline tokens until ``len(tokens) % block_size == 0``.

    Idempotent — returns ``tokens`` unchanged when already aligned.

    Why this matters: PC's ``get_num_new_matched_tokens`` placement
    validation requires that ``num_computed_tokens`` (everything covered
    by prefix-cache + already-claimed external matches) plus the chunk
    coverage exactly partitions ``prompt_token_ids``. vLLM's prefix
    cache only stores complete blocks, so if the sys prompt isn't
    block-aligned, the trailing tail tokens never cache. Subsequent
    queries hit ``num_computed_tokens = floor(sys_len / block_size) *
    block_size`` and PC's chunk-tokens-at-this-offset check fails on
    the remaining sys tail bytes → PC drops the load → silent fallback
    to full prefill every single request.

    Padding with newlines is the least intrusive choice: Llama-family
    tokenizers turn each "\\n" into a single token, and the model has
    seen newline-after-instruction structure billions of times in
    training, so it doesn't perturb generation.
    """
    pad_needed = (-len(tokens)) % block_size
    if pad_needed == 0:
        return list(tokens)
    # Build a pool of pad tokens defensively — if a tokenizer ever
    # merged "\n" sequences into a single token (which Llama-3 does
    # not but defensiveness is cheap), keep adding until the pool is
    # at least ``pad_needed`` long, then slice.
    pool: list[int] = []
    pad_text = "\n"
    while len(pool) < pad_needed:
        pool.extend(tokenizer.encode(pad_text, add_special_tokens=False))
        pad_text += "\n"
    return list(tokens) + pool[:pad_needed]


def build_prompt_and_plan(
    tokenizer,
    sys_prompt: str,
    retrieved: list[IndexEntry],
    query: str,
    block_size: int = 16,
) -> tuple[list[int], dict, int, int]:
    """Build the augmented prompt token IDs and the matching reuse_plan.

    Layout: ``[sys_tokens_padded][chunk_tokens concatenated][query_tokens]``.

    ``sys_tokens_padded`` is the raw sys-prompt tokenisation padded up
    to a ``block_size`` boundary; without this padding PC's placement
    validation drops every request after the first (see
    ``_pad_tokens_to_block_size`` docstring). Chunk tokens come
    straight from the stored ``IndexEntry.token_ids`` so the placement
    check (chunk == prompt slice at the post-sys offset) holds.

    Returns:
        prompt_token_ids: flat list of ints to feed vLLM.
        reuse_params: dict shaped for ``SamplingParams.extra_args``.
        sys_len_raw: pre-pad sys token count (useful for surfacing the
            "how much was added" delta).
        sys_len_padded: post-pad sys token count, also = the offset
            where chunk_0 starts in the prompt.
    """
    sys_tokens_raw = tokenizer.encode(sys_prompt, add_special_tokens=False)
    sys_tokens = _pad_tokens_to_block_size(
        sys_tokens_raw, tokenizer, block_size
    )

    chunk_tokens_flat: list[int] = []
    for e in retrieved:
        chunk_tokens_flat.extend(e.token_ids)
    query_suffix = f"\n\n{query}\n\nAnswer:"
    query_tokens = tokenizer.encode(query_suffix, add_special_tokens=False)

    prompt_token_ids = sys_tokens + chunk_tokens_flat + query_tokens

    reuse_params = {
        "kv_transfer_params": {
            "reuse_plan": {
                "chunks": [
                    {
                        "token_ids": e.token_ids,
                        "old_pos_start": e.old_pos_start,
                        "salt_hex": "",
                    }
                    for e in retrieved
                ]
            }
        }
    }
    return (
        prompt_token_ids,
        reuse_params,
        len(sys_tokens_raw),
        len(sys_tokens),
    )


# ----------------------- PC test-bind for the worker -----------------------


@contextmanager
def _pc_test_bind_storage(storage):
    """Pickle the storage and point worker-side connector at it via env.

    This is the same channel ``test_e2e_gpu.py`` uses. It's labelled
    "test bind" because production would source the store from a
    persistent backend (Phase 12 native KV server), not pickle —
    but for a single-process demo the pickle bridge is the
    simplest cross-process delivery.
    """
    fd, path = tempfile.mkstemp(suffix=".pkl", prefix="pc_rag_demo_")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            pickle.dump({"storage": storage}, f)
        prev = os.environ.get("VLLM_PERSONAL_CONTEXT_TEST_BIND")
        os.environ["VLLM_PERSONAL_CONTEXT_TEST_BIND"] = path
        try:
            yield path
        finally:
            if prev is None:
                os.environ.pop("VLLM_PERSONAL_CONTEXT_TEST_BIND", None)
            else:
                os.environ["VLLM_PERSONAL_CONTEXT_TEST_BIND"] = prev
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ----------------------- main -----------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Personal-context RAG demo with KV reuse."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help=f"Path to JSONL of instances (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="FAISS retrieval top-K per query.",
    )
    parser.add_argument(
        "--selector_r",
        type=float,
        default=1.0,
        help=(
            "SelectFirstR.r. 1.0 = full recompute (byte-equal vanilla, "
            "no prefill saving). Lower values exercise partial reuse."
        ),
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Max tokens generated per query.",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=DEFAULT_EMBEDDER,
        help="sentence-transformers model id.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
        help="vLLM gpu_memory_utilization.",
    )
    args = parser.parse_args()

    # 1. Load JSONL.
    with open(args.data, encoding="utf-8") as f:
        instances = [json.loads(line) for line in f if line.strip()]
    print(f"[demo] loaded {len(instances)} instances from {args.data}")

    # 2. Build index (encoder + embedder + FAISS + PC store).
    preset = QWEN_2_5_0_5B_INSTRUCT
    index = RAGIndex(
        preset=preset,
        embedder_model_id=args.embedder,
        block_size=16,
        device="cuda",
    )
    index.ingest_instances(instances)
    index.build_faiss()

    # 3. Retrieve up-front (while embedder is still alive). Pre-compute
    #    everything that needs the index so we can free its VRAM
    #    before booting vLLM — the 1B model + KV cache will want it.
    queries = []
    for inst in instances:
        retrieved = index.retrieve(inst["query"], top_k=args.top_k)
        queries.append({"instance": inst, "retrieved": retrieved})

    _print_retrieval_quality(queries)

    # 4. Free encoder + embedder VRAM, hold on to tokenizer + store.
    tokenizer = index.tokenizer
    kv_store = index.kv_store
    index.free_index_models()

    # 5. Boot vLLM with PC connector.
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.inputs import TokensPrompt

    kv_transfer_config = KVTransferConfig(
        kv_connector="PersonalContextKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "selector": {"type": "SelectFirstR", "r": args.selector_r},
        },
    )

    with _pc_test_bind_storage(kv_store):
        llm = LLM(
            model=preset.hf_id,
            dtype="float16",
            block_size=16,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
            kv_transfer_config=kv_transfer_config,
            attention_config={"backend": "FLASHINFER"},
        )

        try:
            for q in queries:
                inst = q["instance"]
                retrieved: list[IndexEntry] = q["retrieved"]
                (
                    prompt_token_ids,
                    reuse_params,
                    sys_len_raw,
                    sys_len_padded,
                ) = build_prompt_and_plan(
                    tokenizer,
                    inst["sys"],
                    retrieved,
                    inst["query"],
                    block_size=16,
                )
                sampling = SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    extra_args=reuse_params,
                )
                out = llm.generate(
                    [TokensPrompt(prompt_token_ids=prompt_token_ids)],
                    sampling_params=[sampling],
                )
                generated = out[0].outputs[0].text

                _print_query_report(
                    inst,
                    retrieved,
                    generated,
                    sys_len_raw,
                    sys_len_padded,
                    len(prompt_token_ids),
                )
        finally:
            del llm
            torch.cuda.empty_cache()


# ----------------------- pretty printing -----------------------


def _print_retrieval_quality(queries: list[dict]) -> None:
    """How many of each instance's labelled chunks does FAISS surface?"""
    print()
    print("=" * 72)
    print("Retrieval quality (FAISS top-K vs labelled gold chunks)")
    print("=" * 72)
    for q in queries:
        inst = q["instance"]
        retrieved: list[IndexEntry] = q["retrieved"]
        retrieved_global_ids = {e.global_id for e in retrieved}
        gold_global_ids = {
            f"{inst['id']}__{name}" for name in inst["chunks"].keys()
        }
        hits = retrieved_global_ids & gold_global_ids
        print(
            f"[{inst['id']}] {len(hits):>2}/{len(retrieved):>2} retrieved are "
            f"labelled chunks for this query  |  "
            f"query: {inst['query'][:64]!r}"
        )


def _print_query_report(
    instance: dict,
    retrieved: list[IndexEntry],
    generated: str,
    sys_len_raw: int,
    sys_len_padded: int,
    prompt_len: int,
) -> None:
    pad_note = (
        f"raw={sys_len_raw}+pad={sys_len_padded - sys_len_raw}"
        if sys_len_padded != sys_len_raw
        else f"raw={sys_len_raw}, no pad needed"
    )
    print()
    print("=" * 72)
    print(
        f"[{instance['id']}]  prompt={prompt_len} tok  "
        f"(sys={sys_len_padded} tok — {pad_note})"
    )
    print(f"Query    : {instance['query']}")
    print(f"Gold ans : {instance['answer']}")
    print("Retrieved:")
    for e in retrieved:
        marker = "★" if e.instance_id == instance["id"] else " "
        print(
            f"  {marker} [{e.source:<8}] {e.global_id}  ({len(e.token_ids)} tok)"
        )
        print(f"      {e.text[:90]!r}")
    print(f"Output   : {generated.strip()}")


if __name__ == "__main__":
    main()
