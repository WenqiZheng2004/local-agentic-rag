"""Cross-encoder reranker (second-stage ranking).

Bi-encoder retrieval (embeddings + ANN) is fast but approximate: it scores the
query and each chunk independently. A cross-encoder reads the query and a chunk
*together*, so it judges relevance much more accurately — but it's too slow to
run over the whole corpus. The standard pattern is therefore:

    retrieve many candidates with the fast bi-encoder  ->  rerank with the
    slow-but-accurate cross-encoder  ->  keep the best top_k.

We use sentence-transformers' CrossEncoder, which ships with that package, so no
extra dependency is needed — only a small model download.
"""

from __future__ import annotations

from typing import List, Dict, Any

from config import config


class Reranker:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name or config.reranker_model,
                                  device=device or config.device)

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not docs:
            return docs
        pairs = [(query, d["text"]) for d in docs]
        scores = self.model.predict(pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        ranked = sorted(docs, key=lambda d: d.get("rerank_score", 0.0), reverse=True)
        return ranked[:top_k]
