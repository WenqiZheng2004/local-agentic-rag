"""Persistent vector store backed by ChromaDB.

Embeddings are computed by our LocalEmbedder (GPU) and passed in explicitly, so
Chroma is used purely as an ANN index + metadata store.
"""

from __future__ import annotations

from typing import List, Dict, Any

import chromadb

from config import config
from .embeddings import LocalEmbedder


class VectorStore:
    def __init__(self, embedder: LocalEmbedder | None = None):
        self.embedder = embedder or LocalEmbedder()
        self.client = chromadb.PersistentClient(path=config.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        if not chunks:
            return
        embeddings = self.embedder.embed_documents(chunks)
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, text: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if self.count() == 0:
            return []
        q = self.embedder.embed_query(text)
        res = self.collection.query(
            query_embeddings=[q],
            n_results=top_k or config.top_k,
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out = []
        for d, m, dist in zip(docs, metas, dists):
            out.append({"text": d, "metadata": m or {}, "distance": dist})
        return out

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection (clears all indexed documents)."""
        try:
            self.client.delete_collection(config.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
