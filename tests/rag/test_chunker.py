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


def test_file_context_captures_includes_and_globals():
    # A file with ONLY includes + global state vars + an assignment: none of it
    # is a function/struct, but it must NOT be dropped.
    source = (
        "#include <stdio.h>\n"
        "#include \"lvgl.h\"\n"
        "\n"
        "static lv_obj_t *g_label;\n"
        "int g_counter = 0;\n"
        "void prototype_only(void);\n"
        "g_counter = 5;\n"
    )
    chunks = chunk_c_source(source)
    fc = [c for c in chunks if c.kind == "file_context"]
    assert len(fc) == 1  # grouped into one synthetic chunk
    text = fc[0].text
    assert "#include <stdio.h>" in text
    assert "#include \"lvgl.h\"" in text
    assert "static lv_obj_t *g_label;" in text
    assert "int g_counter = 0;" in text
    assert "void prototype_only(void);" in text


def test_two_ai_blocks_sharing_a_tag():
    source = (
        "void a(void) {\n"
        "    // [AI_START_DRAW]\n"
        "    int x = 1;\n"
        "    // [AI_END_DRAW]\n"
        "}\n"
        "void b(void) {\n"
        "    // [AI_START_DRAW]\n"
        "    int y = 2;\n"
        "    // [AI_END_DRAW]\n"
        "}\n"
    )
    chunks = chunk_c_source(source)
    draws = [c for c in chunks if c.kind == "ai_block" and c.symbol == "DRAW"]
    assert len(draws) == 2  # repeated tag -> two distinct chunks
    assert "int x = 1;" in draws[0].text
    assert "int y = 2;" in draws[1].text
    # The two blocks must not be merged into one another.
    assert "int y = 2;" not in draws[0].text


def test_missing_ai_end_does_not_swallow_file():
    source = (
        "void a(void) {\n"
        "    // [AI_START_DRAW]\n"
        "    int x = 1;\n"  # no AI_END for DRAW
        "}\n"
        "void after(void) { return; }\n"
    )
    chunks = chunk_c_source(source)
    # The unterminated block is dropped...
    assert not [c for c in chunks if c.kind == "ai_block"]
    # ...and the rest of the file is still chunked normally.
    funcs = {c.symbol for c in chunks if c.kind == "function"}
    assert {"a", "after"} <= funcs


def test_leading_doc_comment_attached_to_function():
    source = (
        "/**\n"
        " * Draws the header label.\n"
        " */\n"
        "void draw_header(lv_obj_t *parent) {\n"
        "    return;\n"
        "}\n"
    )
    chunks = chunk_c_source(source)
    fn = next(c for c in chunks if c.kind == "function" and c.symbol == "draw_header")
    assert "Draws the header label." in fn.text
    assert fn.text.lstrip().startswith("/**")
    assert fn.start_line == 1  # chunk now starts at the doc comment


def test_leading_line_comment_block_attached():
    source = (
        "// helper that resets state\n"
        "// used on startup\n"
        "void reset_state(void) { return; }\n"
    )
    chunks = chunk_c_source(source)
    fn = next(c for c in chunks if c.symbol == "reset_state")
    assert "helper that resets state" in fn.text
    assert "used on startup" in fn.text


def test_blank_line_breaks_doc_comment_association():
    # A comment separated by a blank line is NOT attached to the declaration.
    source = (
        "// unrelated banner\n"
        "\n"
        "void unrelated(void) { return; }\n"
    )
    chunks = chunk_c_source(source)
    fn = next(c for c in chunks if c.symbol == "unrelated")
    assert "unrelated banner" not in fn.text


def _big_function(name: str, n_lines: int) -> str:
    body = "\n".join(f"    int v{i} = {i};" for i in range(n_lines))
    return f"void {name}(void) {{\n{body}\n}}\n"


def test_no_split_when_cap_not_given():
    # Default behaviour unchanged: a big function stays a single chunk.
    src = _big_function("huge", 300)
    chunks = chunk_c_source(src)  # no max_chunk_chars
    funcs = [c for c in chunks if c.kind == "function"]
    assert len(funcs) == 1
    assert funcs[0].symbol == "huge"


def test_oversized_chunk_is_split_within_cap():
    cap = 600
    src = _big_function("huge", 300)  # ~6000 chars, well above the cap
    chunks = chunk_c_source(src, max_chunk_chars=cap)

    parts = [c for c in chunks if c.kind == "function"]
    assert len(parts) > 1, "the oversized function must be split into sub-chunks"

    # Every sub-chunk respects the cap (lines here are short, so no overflow).
    for p in parts:
        assert len(p.text) <= cap

    # Continuation markers: symbol#partN, metadata part_index/part_total/chunk_split.
    total = parts[0].metadata["part_total"]
    assert total == len(parts)
    for idx, p in enumerate(parts, start=1):
        assert p.symbol == f"huge#part{idx}"
        assert p.metadata["part_index"] == idx
        assert p.metadata["part_total"] == total
        assert p.metadata["chunk_split"] is True

    # Line ranges are sequential and non-overlapping; no text is lost.
    parts_sorted = sorted(parts, key=lambda c: c.start_line)
    for a, b in zip(parts_sorted, parts_sorted[1:]):
        assert b.start_line == a.end_line + 1
    rejoined = "\n".join(p.text for p in parts_sorted)
    assert rejoined == src.rstrip("\n")  # reconstructs the original chunk text


def test_split_preserves_base_metadata():
    cap = 500
    src = _big_function("huge", 300)
    chunks = chunk_c_source(src, base_metadata={"board": "ASY011", "micro": "STM32H750"}, max_chunk_chars=cap)
    parts = [c for c in chunks if c.kind == "function"]
    assert len(parts) > 1
    for p in parts:
        assert p.metadata["board"] == "ASY011"
        assert p.metadata["micro"] == "STM32H750"

