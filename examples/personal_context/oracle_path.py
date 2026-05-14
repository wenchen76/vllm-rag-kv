# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 0 oracle path: full prefill, no KV reuse.

Treats retrieved chunks as ordinary tokens through vanilla vLLM. Captures
generated tokens and optional per-step logprobs to JSON for diffing
against later phases' KV-reuse outputs (correctness ground truth).

Usage:
    python examples/personal_context/oracle_path.py \\
        --model meta-llama/Llama-3.2-1B \\
        --output oracle.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from vllm import LLM, SamplingParams


@dataclass
class RAGPrompt:
    system: str
    chunks: list[str]
    query: str

    def assemble(self) -> str:
        return (
            self.system
            + "\n\n"
            + "\n\n".join(self.chunks)
            + "\n\n"
            + self.query
        )


DEMO_PROMPTS: list[RAGPrompt] = [
    RAGPrompt(
        system="You are a helpful assistant. Use the context to answer.",
        chunks=[
            "Context A: vLLM is a fast LLM inference engine that uses "
            "paged attention to manage KV cache memory efficiently.",
            "Context B: KV cache reuse across requests can dramatically "
            "reduce TTFT for retrieval-augmented generation workloads.",
        ],
        query="Question: in one sentence, what does vLLM do?",
    ),
]


def run_oracle(
    model: str,
    prompts: list[RAGPrompt],
    max_tokens: int,
    logprobs: int | None,
) -> list[dict]:
    llm = LLM(model=model, enforce_eager=True)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=logprobs,
    )
    assembled = [p.assemble() for p in prompts]
    outputs = llm.generate(assembled, sampling)

    records = []
    for prompt, out in zip(prompts, outputs):
        completion = out.outputs[0]
        records.append(
            {
                "prompt": asdict(prompt),
                "assembled": out.prompt,
                "prompt_token_ids": list(out.prompt_token_ids),
                "generated_token_ids": list(completion.token_ids),
                "generated_text": completion.text,
                "logprobs": _serialize_logprobs(completion.logprobs)
                if logprobs
                else None,
            }
        )
    return records


def _serialize_logprobs(steps):
    if steps is None:
        return None
    out = []
    for step in steps:
        out.append(
            {
                str(tid): {"logprob": lp.logprob, "rank": lp.rank}
                for tid, lp in step.items()
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help="Capture top-k logprobs per step. Omit to skip.",
    )
    args = parser.parse_args()

    records = run_oracle(args.model, DEMO_PROMPTS, args.max_tokens, args.logprobs)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} oracle records to {args.output}")


if __name__ == "__main__":
    main()
