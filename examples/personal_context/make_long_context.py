#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate long-context variants of sample_data.jsonl.

The base corpus has ~200-token retrieved contexts (top_k=4 x ~50 tok),
which is too short for KV reuse to beat a vanilla full prefill: a ~200
token prefill is ~100 ms, less than the connector's per-request fixed
overhead, so reuse is a net loss at that length. KV reuse only pays off
once the reused context is long enough that the prefill it saves exceeds
that fixed overhead.

This script produces wide-recall, long-document variants WITHOUT hand
writing data, using two deterministic levers:

  1. Lengthen every chunk. The original text (which carries the gold
     answer's facts) is kept verbatim at the HEAD; same-source filler
     (realistic calendar/email/message/note/contact detail) is appended
     to reach a target per-chunk token budget. Gold answerability is
     preserved because the original clause is never truncated or diluted
     out of the chunk.

  2. Widen retrieval. Each instance keeps its own (gold) chunks at low
     retrieval_rank, then borrows chunks from OTHER instances as
     distractors at higher ranks until it has the target chunk count.
     This mirrors real wide-recall RAG (you pull many candidates and let
     the model sort them) and is why the prompt grows.

target_context ~= chunks_per_instance * per_chunk_tokens. Two presets are
emitted: ~2k and ~4k tokens.

Determinism: no RNG. Distractors are drawn by rotating through the pool
of all other instances' chunks, so re-running produces byte-identical
output (and therefore stable mmap/Redis block hashes).

Token counts here are CHARACTER-BASED ESTIMATES (~4 chars/token); the
real tokenizer count is what the demo/bench prints. Targets are
approximate by design.

Usage:
    python examples/personal_context/make_long_context.py
    # writes sample_data_2k.jsonl and sample_data_4k.jsonl next to the base
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).parent
_BASE = _HERE / "sample_data.jsonl"

# Rough chars-per-token for sizing filler. The real count comes from the
# model tokenizer at run time; this only needs to be in the ballpark.
_CHARS_PER_TOK = 4

# Two output presets: (filename, chunks_per_instance, per_chunk_tokens).
# ~2k:  14 * ~150 = ~2100 tok ; ~4k: 26 * ~150 = ~3900 tok.
_PRESETS = [
    ("sample_data_2k.jsonl", 14, 150),
    ("sample_data_4k.jsonl", 26, 150),
]


# Same-source filler sentences. Appended (cycling) after the original
# chunk text to pad it out while staying in-domain and plausible. These
# carry no gold facts — they are realistic surrounding detail.
_FILLER = {
    "email": [
        "Please let me know if anything here needs to change.",
        "This message and any attachments are intended for the named "
        "recipient only.",
        "Sent from my phone, apologies for any brevity or typos.",
        "Reply-all is enabled; loop in whoever else should be aware.",
        "A calendar hold has been sent separately for your convenience.",
        "For reference, the prior thread on this topic is quoted below.",
        "No action is required if the details above already look correct.",
        "Thanks again for your patience while we sorted this out.",
    ],
    "calendar": [
        "Reminder set for 30 minutes before the start time.",
        "Attendees: organizer plus invited guests; optional attendees may "
        "join remotely.",
        "Add to your calendar and set travel time if the location is far.",
        "Notes: bring any relevant documents and arrive a few minutes early.",
        "Video link will be attached to the invite if held remotely.",
        "This event repeats on the usual cadence unless cancelled.",
        "Free/busy shows this slot as confirmed and blocking.",
        "Updated automatically from the latest accepted invitation.",
    ],
    "messages": [
        "[me] sounds good, thanks for the heads up",
        "[them] no worries, talk soon",
        "[me] ok will keep you posted if anything changes",
        "[them] perfect, appreciate it",
        "[me] got it, adding a reminder now",
        "[them] cool, let me know either way",
        "[me] yeah that works for me",
        "[them] great, see you then",
    ],
    "notes": [
        "Follow up later if the situation changes.",
        "Double-check the details before acting on this.",
        "Cross-referenced with the calendar entry to be safe.",
        "Low priority unless something else comes up.",
        "Keep this handy for the next related conversation.",
        "Tagged for review at the end of the week.",
        "No further action needed for now.",
        "Saved a copy in the relevant project folder.",
    ],
    "contacts": [
        "Preferred contact method: email during business hours.",
        "Timezone: Pacific. Usually responsive within a day.",
        "Notes: relationship established a few years back.",
        "Backup phone and address on file if needed.",
        "Do not share these details outside the household.",
        "Last interaction was pleasant and productive.",
        "Tagged as a frequent and trusted contact.",
        "Add to favorites for quicker access.",
    ],
}
# Fallback filler for any unexpected source value.
_FILLER_DEFAULT = _FILLER["notes"]


def _lengthen(text: str, source: str, target_tok: int) -> str:
    """Append same-source filler until ~target_tok (char-estimated).

    The original ``text`` is kept verbatim at the head so the gold facts
    are never diluted out; filler only extends the tail.
    """
    target_chars = target_tok * _CHARS_PER_TOK
    pool = _FILLER.get(source, _FILLER_DEFAULT)
    out = text
    i = 0
    while len(out) < target_chars:
        out = out + " " + pool[i % len(pool)]
        i += 1
    return out


def _build_pool(rows: list[dict]) -> list[tuple[str, dict]]:
    """Flat list of (owner_instance_id, chunk_data) over ALL chunks, used
    as the distractor source. Order is deterministic (corpus order)."""
    pool: list[tuple[str, dict]] = []
    for r in rows:
        for _cname, cdata in r["chunks"].items():
            pool.append((r["id"], cdata))
    return pool


def _make_variant(
    rows: list[dict],
    chunks_per_instance: int,
    per_chunk_tokens: int,
) -> list[dict]:
    pool = _build_pool(rows)
    pool_len = len(pool)
    out_rows: list[dict] = []

    # Global rotating cursor into the distractor pool so borrowing is
    # spread across the corpus and deterministic across runs.
    cursor = 0

    for r in rows:
        own_chunks = list(r["chunks"].items())  # [(name, data), ...]
        n_own = len(own_chunks)

        new_chunks: dict[str, dict] = {}
        rank = 1

        # 1) Keep this instance's own (gold) chunks at the lowest ranks,
        #    lengthened. These carry the answer.
        for _name, cdata in own_chunks:
            new_chunks[f"chunk_{rank - 1}"] = {
                "text": _lengthen(
                    cdata["text"], cdata.get("source", "notes"),
                    per_chunk_tokens,
                ),
                "source": cdata.get("source", "notes"),
                "retrieval_rank": rank,
            }
            rank += 1

        # 2) Borrow distractor chunks from OTHER instances until we hit
        #    the target count. Skip any chunk owned by this instance so a
        #    gold chunk is never duplicated as a distractor.
        need = chunks_per_instance - n_own
        added = 0
        scanned = 0
        while added < need and scanned < pool_len:
            owner, cdata = pool[cursor % pool_len]
            cursor += 1
            scanned += 1
            if owner == r["id"]:
                continue  # don't borrow our own chunk as a distractor
            new_chunks[f"chunk_{rank - 1}"] = {
                "text": _lengthen(
                    cdata["text"], cdata.get("source", "notes"),
                    per_chunk_tokens,
                ),
                "source": cdata.get("source", "notes"),
                "retrieval_rank": rank,
            }
            rank += 1
            added += 1

        out_rows.append(
            {
                "id": r["id"],
                "sys": r["sys"],
                "query": r["query"],
                "answer": r["answer"],
                "chunks": new_chunks,
            }
        )
    return out_rows


def main() -> None:
    rows = [
        json.loads(line)
        for line in _BASE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"[make-long] base: {len(rows)} instances from {_BASE.name}")

    for fname, n_chunks, per_tok in _PRESETS:
        variant = _make_variant(rows, n_chunks, per_tok)
        out_path = _HERE / fname
        with out_path.open("w", encoding="utf-8") as f:
            for row in variant:
                f.write(json.dumps(row) + "\n")

        # Report the realised sizing (char-estimated) so the user can
        # sanity-check before paying the encode cost.
        est = []
        for row in variant:
            tok = sum(
                len(c["text"]) // _CHARS_PER_TOK
                for c in row["chunks"].values()
            )
            est.append(tok)
        est.sort()
        median = est[len(est) // 2]
        print(
            f"[make-long] wrote {fname}: {n_chunks} chunks/instance, "
            f"~{per_tok} tok/chunk → est context median ~{median} tok "
            f"(min ~{est[0]}, max ~{est[-1]})"
        )


if __name__ == "__main__":
    main()
