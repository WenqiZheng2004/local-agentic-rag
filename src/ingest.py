"""Document loading, chunking, and ingestion into the vector store.

Supported file types: .pdf, .txt, .md
"""

from __future__ import annotations

import os
import glob
import hashlib
from typing import List

from config import config
from .vectorstore import VectorStore

SUPPORTED = (".pdf", ".txt", ".md")


def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader  # imported lazily so non-PDF users don't need it
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
    """Paragraph-aware greedy chunking with a character overlap between neighbors.

    Splits on blank lines first, packs paragraphs up to ``chunk_size``, hard-splits
    any oversized paragraph, then stitches a small tail of the previous chunk onto
    the next one to preserve context across boundaries.
    """
    chunk_size = chunk_size or config.chunk_size
    overlap = overlap or config.chunk_overlap

    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) + 2 <= chunk_size:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append(current)
                current = ""
            if len(p) <= chunk_size:
                current = p
            else:
                start = 0
                step = max(1, chunk_size - overlap)
                while start < len(p):
                    chunks.append(p[start:start + chunk_size])
                    start += step
    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        stitched = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            stitched.append((tail + "\n" + chunks[i]).strip())
        chunks = stitched

    return chunks


def ingest_file(path: str, store: VectorStore | None = None) -> int:
    """Load, chunk, embed and store a single file. Returns number of chunks added."""
    store = store or VectorStore()
    text = load_text(path)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    base = os.path.basename(path)
    ids, metas = [], []
    for i, c in enumerate(chunks):
        uid = hashlib.md5(f"{base}-{i}-{c[:64]}".encode("utf-8")).hexdigest()
        ids.append(uid)
        metas.append({"source": base, "chunk": i})

    store.add(chunks, metas, ids)
    return len(chunks)


def ingest_dir(folder: str, store: VectorStore | None = None) -> int:
    """Recursively ingest every supported file under ``folder``."""
    store = store or VectorStore()
    total = 0
    for path in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in SUPPORTED:
            total += ingest_file(path, store)
    return total
