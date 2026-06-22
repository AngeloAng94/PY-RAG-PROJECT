"""Tests for rag.indexer — oversized-chunk split through ingest + build resilience."""

from __future__ import annotations

from rag.constants import ABSENT
from rag.indexer import index_file, index_repo
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
    assert len(summary["skipped"]) == 1
    skipped = summary["skipped"][0]
    assert skipped["file"].endswith("bad.c")
    assert "TimeoutError" in skipped["error"]

    # The two good functions are in the store; the bad one is not.
    symbols = {r["metadata"].get("symbol") for r in store.list_chunks()}
    assert {"ok", "ok2"} <= symbols
    assert "boom" not in symbols
