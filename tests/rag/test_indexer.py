"""Tests for rag.indexer — oversized-chunk split through ingest + build resilience."""

from __future__ import annotations

from rag.constants import ABSENT
from rag.indexer import index_file, index_repo, looks_like_data
from rag.store import ChromaStore
from tests.rag.conftest import FakeEmbedder


def _big_function_file(tmp_path, name, n_lines=300):
    body = "\n".join(f"    int v{i} = {i};" for i in range(n_lines))
    p = tmp_path / f"{name}.c"
    p.write_text(f"void {name}(void) {{\n{body}\n}}\n", encoding="utf-8")
    return str(p)


def test_oversized_function_split_persists_distinct_ids(tmp_path):
    fpath = _big_function_file(tmp_path, "huge", 300)
    store = ChromaStore(index_path=str(tmp_path / "idx"))
    store.reset()
    embedder = FakeEmbedder()

    count = index_file(
        fpath, store, embedder,
        base_metadata={"board": "ASY011", "micro": "STM32H750"},
        max_chunk_chars=600,
    )
    assert count > 1  # the single function became several sub-chunks

    rows = store.list_chunks(where={"board": {"$eq": "ASY011"}})
    # Distinct ids (no collisions) and the split markers are stored.
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) == count
    split_rows = [r for r in rows if r["metadata"].get("chunk_split")]
    assert len(split_rows) == count
    assert all(r["metadata"].get("part_total") == count for r in split_rows)


class _FlakyEmbedder(FakeEmbedder):
    """Fails on any batch whose text contains the BOOM marker (simulates timeout)."""

    def embed_documents(self, texts):
        if any("BOOM" in t for t in texts):
            raise TimeoutError("simulated read timeout from embedder")
        return super().embed_documents(texts)


def test_index_repo_skips_failing_file_and_continues(tmp_path):
    good = tmp_path / "good.c"
    good.write_text("void ok(void) { return; }\n", encoding="utf-8")
    bad = tmp_path / "bad.c"
    bad.write_text("void boom(void) { int BOOM = 1; (void)BOOM; }\n", encoding="utf-8")
    good2 = tmp_path / "good2.c"
    good2.write_text("void ok2(void) { return; }\n", encoding="utf-8")

    store = ChromaStore(index_path=str(tmp_path / "idx"))
    store.reset()

    summary = index_repo(
        repo_path=str(tmp_path),
        store=store,
        embedder=_FlakyEmbedder(),
        base_metadata={"board": "ASY011", "micro": "STM32H750"},
        reset=True,
    )

    # The bad file is skipped; the build is NOT aborted — good files indexed.
    assert summary["files"] == 2
    assert summary["chunks"] >= 2
    assert len(summary["skipped_error"]) == 1
    assert summary["skipped_data"] == []
    skipped = summary["skipped_error"][0]
    assert skipped["file"].endswith("bad.c")
    assert "TimeoutError" in skipped["error"]

    # The two good functions are in the store; the bad one is not.
    symbols = {r["metadata"].get("symbol") for r in store.list_chunks()}
    assert {"ok", "ok2"} <= symbols
    assert "boom" not in symbols


# --- content-based asset/data detection ------------------------------------

def _img_descriptor_with_long_line(tmp_path, name="img6"):
    """Mimics an image-as-C file: ~9 useful lines + one giant pixel-blob line."""
    blob = "0xcf," * 1400  # ~7000 chars on a single line, like a real img map
    p = tmp_path / f"{name}.c"
    p.write_text(
        "#include \"lvgl.h\"\n"
        f"const uint8_t {name}_map[] = {{ {blob} }};\n"
        f"const lv_img_dsc_t {name} = {{ .data = {name}_map, .w = 64, .h = 64 }};\n",
        encoding="utf-8",
    )
    return p


def _byte_array_file_short_lines(tmp_path, name="blob", entries=4000):
    """A dominant byte array with SHORT lines (no >2000-char line)."""
    rows = []
    for i in range(0, entries, 16):
        rows.append("    " + ", ".join("0xab" for _ in range(16)) + ",")
    body = "\n".join(rows)
    p = tmp_path / f"{name}.c"
    p.write_text(f"const uint8_t {name}_map[] = {{\n{body}\n}};\n", encoding="utf-8")
    return p


def _normal_module(tmp_path, name="clock"):
    p = tmp_path / f"{name}.c"
    p.write_text(
        "#include <stdint.h>\n"
        "static uint8_t hours, minutes;\n"
        "void clock_tick(void) {\n"
        "    minutes++;\n"
        "    if (minutes >= 60) { minutes = 0; hours = (hours + 1) % 24; }\n"
        "}\n"
        "uint8_t clock_get_hours(void) { return hours; }\n",
        encoding="utf-8",
    )
    return p


def test_looks_like_data_unit():
    # (1) long line
    long_line = "const uint8_t m[] = {" + "0x1," * 1000 + "};"
    assert "longest line" in (looks_like_data(long_line, 2000) or "")
    # (2) dominant byte array, short lines
    rows = "\n".join("    " + ", ".join("0x1" for _ in range(16)) + "," for _ in range(300))
    blob = f"const uint8_t m[] = {{\n{rows}\n}};\n"
    assert "byte-array" in (looks_like_data(blob, 2000) or "")
    # normal code is NOT flagged
    assert looks_like_data("void f(void){ int x=1; (void)x; }\n", 2000) is None
    # a small legitimate lookup table is NOT flagged (below absolute floor)
    lut = "const uint16_t sin_table[8] = { 0, 90, 180, 270, 360, 270, 180, 90 };\n"
    assert looks_like_data(lut, 2000) is None


def test_image_as_c_file_skipped_as_data(tmp_path):
    _img_descriptor_with_long_line(tmp_path, "img6")
    _normal_module(tmp_path, "clock")
    store = ChromaStore(index_path=str(tmp_path / "idx"))
    store.reset()

    summary = index_repo(str(tmp_path), store, FakeEmbedder(),
                         base_metadata={"board": "ASY011", "micro": "STM32H750"}, reset=True)

    assert len(summary["skipped_data"]) == 1
    assert summary["skipped_data"][0]["file"].endswith("img6.c")
    assert "longest line" in summary["skipped_data"][0]["reason"]
    # the normal module is still indexed
    assert summary["files"] == 1
    symbols = {r["metadata"].get("symbol") for r in store.list_chunks()}
    assert "clock_tick" in symbols


def test_byte_array_file_short_lines_skipped_as_data(tmp_path):
    _byte_array_file_short_lines(tmp_path, "blob", entries=4000)
    store = ChromaStore(index_path=str(tmp_path / "idx"))
    store.reset()
    summary = index_repo(str(tmp_path), store, FakeEmbedder(),
                         base_metadata={"board": "ASY011", "micro": "STM32H750"}, reset=True)
    assert len(summary["skipped_data"]) == 1
    assert "byte-array" in summary["skipped_data"][0]["reason"]
    assert summary["files"] == 0


def test_normal_module_is_indexed(tmp_path):
    _normal_module(tmp_path, "clock")
    store = ChromaStore(index_path=str(tmp_path / "idx"))
    store.reset()
    summary = index_repo(str(tmp_path), store, FakeEmbedder(),
                         base_metadata={"board": "ASY011", "micro": "STM32H750"}, reset=True)
    assert summary["skipped_data"] == []
    assert summary["files"] == 1
    symbols = {r["metadata"].get("symbol") for r in store.list_chunks()}
    assert {"clock_tick", "clock_get_hours"} <= symbols


def test_include_glob_forces_indexing_of_flagged_file(tmp_path):
    # A file the heuristic WOULD flag, force-indexed via --include.
    _byte_array_file_short_lines(tmp_path, "lut", entries=4000)
    store = ChromaStore(index_path=str(tmp_path / "idx"))
    store.reset()
    summary = index_repo(str(tmp_path), store, FakeEmbedder(),
                         base_metadata={"board": "ASY011", "micro": "STM32H750"},
                         reset=True, include=["lut.c"])
    assert summary["skipped_data"] == []   # override won
    assert summary["files"] == 1
    # include_data=True also forces everything
    store.reset()
    summary2 = index_repo(str(tmp_path), store, FakeEmbedder(),
                          base_metadata={"board": "ASY011", "micro": "STM32H750"},
                          reset=True, include_data=True)
    assert summary2["skipped_data"] == []
    assert summary2["files"] == 1


def test_exclude_glob_honoured(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "asset.c").write_text("void a(void){return;}\n", encoding="utf-8")
    _normal_module(tmp_path, "clock")
    store = ChromaStore(index_path=str(tmp_path / "idx"))
    store.reset()
    summary = index_repo(str(tmp_path), store, FakeEmbedder(),
                         base_metadata={"board": "ASY011", "micro": "STM32H750"},
                         reset=True, exclude=["images/"])
    assert len(summary["skipped_excluded"]) == 1
    assert summary["skipped_excluded"][0]["file"].endswith("asset.c")
    assert summary["files"] == 1  # only clock.c
    symbols = {r["metadata"].get("symbol") for r in store.list_chunks()}
    assert "clock_tick" in symbols
    assert "a" not in symbols


def test_summary_reports_data_and_error_separately(tmp_path):
    _img_descriptor_with_long_line(tmp_path, "img6")     # -> skipped_data
    (tmp_path / "bad.c").write_text("void boom(void){int BOOM=1;(void)BOOM;}\n", encoding="utf-8")  # -> skipped_error
    _normal_module(tmp_path, "clock")                    # -> indexed
    store = ChromaStore(index_path=str(tmp_path / "idx"))
    store.reset()
    summary = index_repo(str(tmp_path), store, _FlakyEmbedder(),
                         base_metadata={"board": "ASY011", "micro": "STM32H750"}, reset=True)
    assert len(summary["skipped_data"]) == 1
    assert len(summary["skipped_error"]) == 1
    assert summary["files"] == 1
    assert summary["skipped_data"][0]["file"].endswith("img6.c")
    assert summary["skipped_error"][0]["file"].endswith("bad.c")

