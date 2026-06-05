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

Redis backend (Phase 12.2 — KV blocks shared via Redis instead of pickle):
    docker run -d --name pc-redis -p 6379:6379 redis/redis-stack-server
    .venv/bin/python examples/personal_context/rag_demo.py \
        --store_backend redis \
        --store_url redis://localhost:6379
"""

from __future__ import annotations

import argparse
import contextlib
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
    HFChunkEncoder,
    preset_for,
    store_config_for,
)
from vllm.v1.personal_context import Chunk, InMemoryStorage  # noqa: E402


DEFAULT_DATA_PATH = Path(__file__).parent / "sample_data.jsonl"
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REDIS_URL = "redis://localhost:6379"
DEFAULT_MMAP_DIR = "/tmp/pc_mmap"


def _default_store_url(backend: str) -> str:
    """Per-backend default for ``--store_url`` (resolved when the flag is
    omitted): a redis:// URL for redis, an on-disk directory for mmap.
    Returns "" for memory (the value is ignored there)."""
    if backend == "redis":
        return DEFAULT_REDIS_URL
    if backend == "mmap":
        return DEFAULT_MMAP_DIR
    return ""


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


# ----------------------- storage factory -----------------------


def _build_kv_store(
    preset, block_size: int, backend: str, url: str,
    dtype: torch.dtype = torch.float16,
):
    """Build the encoder-side KV store matching the worker-side backend.

    Worker-side connector chooses its own backend from
    ``kv_connector_extra_config`` (see ``_maybe_init_storage_from_config``
    in the connector). For the writes from the encoder process to be
    visible to the worker, both sides must point at the same backing
    store; for ``redis`` that's the Redis URL, for ``mmap`` it's the
    on-disk directory (``url``), for ``memory`` it's the pickle bridge.
    """
    cfg = store_config_for(preset, block_size=block_size, dtype=dtype)
    if backend == "memory":
        return InMemoryStorage(cfg)
    if backend == "redis":
        from vllm.v1.personal_context.redis_storage import RedisKVStorage

        print(f"[index] connecting to Redis at {url}")
        return RedisKVStorage(cfg, url=url)
    if backend == "mmap":
        from vllm.v1.personal_context.mmap_storage import MmapKVStorage

        print(f"[index] opening mmap store at {url}")
        return MmapKVStorage(cfg, root_dir=url)
    raise ValueError(
        f"unknown store_backend {backend!r}; must be 'memory', 'redis', "
        "or 'mmap'"
    )


# ----------------------- index -----------------------


class RAGIndex:
    """Bundles the encoder, embedder, FAISS index, and PC KV store.

    Holds everything index-side in one object so VRAM cleanup
    (``free_index_models``) before the vLLM engine starts is a single
    method call.

    ``store_backend`` selects how the encoded KV blocks reach the vLLM
    worker:

        - ``"memory"`` → ``InMemoryStorage`` here, pickle-bridged to the
          worker via ``VLLM_PERSONAL_CONTEXT_TEST_BIND`` env var. Test /
          quick-demo path.
        - ``"redis"`` → ``RedisKVStorage`` against ``store_url``. Worker
          builds its own ``RedisKVStorage`` against the same URL via
          ``kv_connector_extra_config``. Cross-process via Redis instead
          of pickle. Production / Phase 12 path.
    """

    def __init__(
        self,
        preset,
        embedder_model_id: str = DEFAULT_EMBEDDER,
        block_size: int = 16,
        device: str = "cuda",
        store_backend: str = "memory",
        store_url: str = DEFAULT_REDIS_URL,
        dtype: torch.dtype = torch.float16,
    ):
        # faiss is validated lazily in build_faiss() (the only place it's
        # used), so oracle-retrieval runs that never retrieve need no faiss.
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

        print(f"[index] preparing HF encoder (tokenizer-only init): {preset.hf_id}")
        # Lazy model load — defer ~30s of HF model loading until the
        # first cache miss. Tokenizer loads up-front (needed for cache
        # hit checks even on full-warm runs).
        self.dtype = dtype
        self.encoder = HFChunkEncoder(
            preset=preset,
            device=device,
            dtype=dtype,
            lazy_model=True,
        )
        self.tokenizer = self.encoder.tokenizer
        # Cache hit/miss counters, reset per ingest_instances() call.
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        print(f"[index] loading embedder: {embedder_model_id}")
        self.embedder = SentenceTransformer(embedder_model_id, device=device)

        self.store_backend = store_backend
        self.store_url = store_url
        self.kv_store = _build_kv_store(
            preset=preset,
            block_size=block_size,
            backend=store_backend,
            url=store_url,
            dtype=dtype,
        )
        self.entries: list[IndexEntry] = []
        self.faiss_index: Any = None

    # ----- ingestion -----

    def ingest_instances(self, instances: list[dict]) -> None:
        self._cache_hits = 0
        self._cache_misses = 0
        for inst in instances:
            self._ingest_instance(inst)
        total = self._cache_hits + self._cache_misses
        if total > 0:
            print(
                f"[index] ingest summary: {self._cache_hits}/{total} chunks "
                f"served from cache, {self._cache_misses} encoded fresh. "
                f"HF model loaded: {self.encoder.is_model_loaded}"
            )

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

        # Cache check (Phase 12.4): if every block hash for this chunk
        # is already in the KV store, skip encoding entirely. With a
        # warm Redis this avoids both the HF forward pass (~1s / chunk)
        # AND the first cache-miss-triggered HF model load (~30s for
        # Llama-3-8B). Cheap: ``__contains__`` is one EXISTS per block,
        # ~0.1ms localhost.
        cache_check_chunk = Chunk(token_ids=tuple(tokens), old_pos_start=0)
        cache_check_hashes = list(
            cache_check_chunk.block_hashes(self.block_size)
        )
        if all(h in self.kv_store for h in cache_check_hashes):
            print(
                f"[hit ] {instance_id}/{chunk_name}: "
                f"{len(cache_check_hashes)} blocks already cached, "
                f"skipping HF encode"
            )
            self._cache_hits += 1
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
            return

        # Cache miss → encode (triggers HF model lazy load on the first
        # miss) → push every block into the PC store.
        self._cache_misses += 1
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
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "RAGIndex.build_faiss() requires faiss-cpu (or faiss-gpu). "
                "Install via `uv pip install faiss-cpu`, or run with oracle "
                "retrieval (--oracle_retrieval), which needs no faiss."
            ) from e

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
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help=(
            "HF model id. Must have a matching ModelPreset registered "
            "in chunk_encoder_hf.KNOWN_PRESETS. Gated models (Llama-3) "
            "require ``huggingface-cli login`` first."
        ),
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
        default=160,
        help="Max tokens generated per query (160: the sys prompt asks for "
        "all details, so answers run a little longer than the ~70-tok gold).",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=DEFAULT_EMBEDDER,
        help="sentence-transformers model id.",
    )
    parser.add_argument(
        "--store_backend",
        type=str,
        default="mmap",
        choices=["memory", "redis", "mmap"],
        help=(
            "KV store backend (default: mmap). 'mmap' = MmapKVStorage, a "
            "same-host on-disk store the worker shares with no "
            "serialisation/socket overhead (--store_url is the directory, "
            f"default {DEFAULT_MMAP_DIR}). 'memory' = process-local "
            "InMemoryStorage bridged to the worker via pickle (test path). "
            "'redis' = RedisKVStorage shared via kv_connector_extra_config; "
            "requires a running Redis instance (e.g. `docker run -d -p "
            "6379:6379 redis/redis-stack-server`)."
        ),
    )
    parser.add_argument(
        "--store_url",
        type=str,
        default=None,
        help=(
            "Backing-store location: a redis:// URL when "
            "--store_backend=redis, or a filesystem directory when "
            "--store_backend=mmap. Ignored for memory. When omitted, "
            f"defaults per backend (redis: {DEFAULT_REDIS_URL}, mmap: "
            f"{DEFAULT_MMAP_DIR})."
        ),
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help=(
            "vLLM gpu_memory_utilization (fraction of TOTAL VRAM, "
            "including model weights). Default 0.85 fits an 8B fp16 "
            "model + KV cache on a 24GB GPU. For smaller models you "
            "can drop to 0.6 to leave room for other GPU users."
        ),
    )
    args = parser.parse_args()

    # Resolve the per-backend default store_url when the flag was omitted,
    # so `--store_backend mmap` alone doesn't inherit the redis URL.
    if args.store_url is None:
        args.store_url = _default_store_url(args.store_backend)

    # 1. Load JSONL.
    with open(args.data, encoding="utf-8") as f:
        instances = [json.loads(line) for line in f if line.strip()]
    print(f"[demo] loaded {len(instances)} instances from {args.data}")

    # 2. Build index (encoder + embedder + FAISS + PC store).
    preset = preset_for(args.model)
    print(
        f"[demo] target model = {preset.hf_id}  "
        f"({preset.num_layers} layers, {preset.num_kv_heads} KV heads, "
        f"head_dim={preset.head_dim}, rope_theta={preset.rope_theta})"
    )
    index = RAGIndex(
        preset=preset,
        embedder_model_id=args.embedder,
        block_size=16,
        device="cuda",
        store_backend=args.store_backend,
        store_url=args.store_url,
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

    # 5. Boot vLLM with PC connector. Worker-side storage binding
    #    depends on backend:
    #      memory → pickle bridge via VLLM_PERSONAL_CONTEXT_TEST_BIND
    #               (storage object travels in a tempfile)
    #      redis  → kv_connector_extra_config carries backend + URL;
    #               worker builds its own RedisKVStorage. No pickle.
    #      mmap   → same as redis but store_url is the on-disk dir; worker
    #               builds its own MmapKVStorage over the same directory.
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.inputs import TokensPrompt

    extra: dict = {
        "selector": {"type": "SelectFirstR", "r": args.selector_r},
    }
    if args.store_backend in ("redis", "mmap"):
        extra["store_backend"] = args.store_backend
        extra["store_url"] = args.store_url

    kv_transfer_config = KVTransferConfig(
        kv_connector="PersonalContextKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra,
    )

    if args.store_backend == "memory":
        worker_bind_ctx = _pc_test_bind_storage(kv_store)
    else:
        worker_bind_ctx = contextlib.nullcontext()

    with worker_bind_ctx:
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
