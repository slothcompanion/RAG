import pytest

from versarag.chunking import split_text


def test_short_text_single_chunk():
    assert split_text("hello world", 100, 10) == ["hello world"]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, 200, 40)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    # overlap: consecutive chunks share some text
    assert any(chunks[i][-15:] in chunks[i + 1] for i in range(len(chunks) - 1))


def test_hard_split_without_separators():
    chunks = split_text("x" * 1050, 100, 0)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == "x" * 1050


def test_overlap_must_be_smaller():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)
