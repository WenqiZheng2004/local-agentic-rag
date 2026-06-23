"""Local embedding model running on the GPU via sentence-transformers.

We keep this independent of any LangChain embedding wrapper so the project has
no hidden dependency on fast-moving integration packages. Vectors are L2
normalized so cosine similarity == dot product.
"""

from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

from config import config


class LocalEmbedder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or config.embed_model
        self.device = device or config.device
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        # bge-*-zh-v1.5 works well without an instruction prefix for short queries.
        # If you switch to bge-large or see weak recall, prepend the recommended
        # retrieval instruction here.
        vec = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec.tolist()
