"""Central configuration. All values can be overridden via environment variables
(see .env.example). Defaults are tuned for an 8 GB GPU (e.g. RTX 4060)."""

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional
    pass


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # --- Models (all run locally) ---
    # 3B fits comfortably on 8 GB. Bump to Qwen/Qwen2.5-7B-Instruct with 4-bit if you want.
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    load_in_4bit: bool = _bool("LOAD_IN_4BIT", True)
    device: str = os.getenv("DEVICE", "cuda")

    # --- Generation ---
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))

    # --- Chunking / retrieval ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("TOP_K", "4"))            # chunks finally sent to the LLM
    retrieve_k: int = int(os.getenv("RETRIEVE_K", "12"))  # candidates pulled before reranking

    # --- Reranker (cross-encoder second-stage ranking) ---
    use_reranker: bool = _bool("USE_RERANKER", True)
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    # --- Vector store ---
    persist_dir: str = os.getenv("PERSIST_DIR", "./storage/chroma")
    collection_name: str = os.getenv("COLLECTION", "docs")

    # --- Agent ---
    max_rewrites: int = int(os.getenv("MAX_REWRITES", "1"))


config = Config()
