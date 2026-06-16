"""Tests for rag.chunker — semantic C chunking + AI marker block extraction."""

from __future__ import annotations

from rag.chunker import chunk_c_source
from tests.rag.conftest import SAMPLE_C


def _by_kind(chunks):
    out = {}
    for c in chunks:
        out.setdefault(c.kind, []).append(c)
    return out


def test_extracts_all_semantic_kinds():
    chunks = chunk_c_source(SAMPLE_C)
    kinds = _by_kind(chunks)

    # Each top-level semantic unit must appear at least once.
    assert "function" in kinds
    assert "struct" in kinds  # bare `struct point_s {...};`
    assert "enum" in kinds  # bare `enum color_e {...};`
    assert "typedef" in kinds  # `typedef struct {...} item_t;`
    assert "define" in kinds  # MAX_ITEMS / SQUARE macro
    assert "ai_block" in kinds


def test_function_symbols_are_extracted():
    chunks = chunk_c_source(SAMPLE_C)
    func_symbols = {c.symbol for c in chunks if c.kind == "function"}
    assert "draw_header" in func_symbols
    assert "on_start_clicked" in func_symbols


def test_ai_blocks_captured_with_tags():
    chunks = chunk_c_source(SAMPLE_C)
    ai = [c for c in chunks if c.kind == "ai_block"]
    tags = {c.symbol for c in ai}
    assert {"DRAW", "EVENTS"} <= tags
    # The block text must contain both delimiters.
    draw = next(c for c in ai if c.symbol == "DRAW")
    assert "[AI_START_DRAW]" in draw.text
    assert "[AI_END_DRAW]" in draw.text


def test_chunks_have_line_ranges_and_no_fixed_split():
    chunks = chunk_c_source(SAMPLE_C)
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line
        assert c.text.strip()  # non-empty semantic unit
