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
