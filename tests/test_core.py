"""Unit tests for the pure-logic pieces (no GPU / models needed).

Run from the project root:

    pip install pytest
    pytest -q
"""

from src.ingest import chunk_text
from eval.evaluate import keyword_score


def test_chunk_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_short_text_single_chunk():
    chunks = chunk_text("hello world", chunk_size=500, overlap=80)
    assert len(chunks) == 1
    assert "hello world" in chunks[0]


def test_chunk_splits_long_text():
    text = ("一段文字。" * 60 + "\n\n") * 4
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and c for c in chunks)


def test_chunk_hard_splits_oversized_paragraph():
    big = "x" * 1200
    chunks = chunk_text(big, chunk_size=300, overlap=50)
    assert len(chunks) >= 4


def test_keyword_score():
    assert keyword_score("核心工作时间是 10:00 到 16:00", ["10:00", "16:00"]) == 1.0
    assert keyword_score("只有 10:00", ["10:00", "16:00"]) == 0.5
    assert keyword_score("nothing matches", ["10:00"]) == 0.0
    assert keyword_score("anything", []) == 0.0
