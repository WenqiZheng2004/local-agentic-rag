"""Tiny evaluation harness.

Runs a set of questions through the agent and reports a keyword-coverage score
(what fraction of expected keywords appear in each answer). It's deliberately
simple and transparent — the point is to show you measure quality, not to claim
a perfect metric. Swap in LLM-as-judge or exact-match if you want.

Usage (from project root):

    python -m eval.evaluate --ingest                       # ingest data/sample then eval
    python -m eval.evaluate --questions eval/questions.jsonl
"""

from __future__ import annotations

import os
import json
import argparse
import warnings
from typing import List, Dict

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
except Exception:
    pass

from src.agent import RAGAgent
from src.ingest import ingest_dir
from src.vectorstore import VectorStore


def load_questions(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def keyword_score(answer: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    a = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in a)
    return hits / len(keywords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--ingest", action="store_true",
                    help="(re)ingest the data dir before evaluating")
    args = ap.parse_args()

    store = VectorStore()
    if args.ingest or store.count() == 0:
        n = ingest_dir(args.data, store)
        print(f"Ingested {n} chunks from {args.data}\n")

    agent = RAGAgent(store=store)
    items = load_questions(args.questions)

    total = 0.0
    print(f"{'score':>6}  {'route':>8}  question")
    print("-" * 70)
    for it in items:
        out = agent.run(it["question"])
        s = keyword_score(out["answer"], it.get("keywords", []))
        total += s
        print(f"{s:>6.2f}  {out['route']:>8}  {it['question']}")

    avg = total / max(len(items), 1)
    print("-" * 70)
    print(f"Average keyword-coverage: {avg:.3f}  over {len(items)} questions")


if __name__ == "__main__":
    main()
