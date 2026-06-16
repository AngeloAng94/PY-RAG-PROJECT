"""Tests for rag.store.list_chunks + the rag.inspect CLI (read-only, offline)."""

from __future__ import annotations

import rag.inspect as inspect_cli
from rag.indexer import index_file
from rag.store import ChromaStore
from tests.rag.conftest import SAMPLE_C


def _populate(tmp_path, fake_embedder):
    index_path = str(tmp_path / "idx")
    target = tmp_path / "screen_caffe.c"
    target.write_text(SAMPLE_C, encoding="utf-8")
    store = ChromaStore(index_path=index_path)
    store.reset()
    index_file(
        str(target),
        store,
        fake_embedder,
        base_metadata={"board": "ASY011", "micro": "STM32H750", "categoria": "caffe", "scope": "categoria"},
    )
    return index_path, store


def test_list_chunks_returns_metadata_no_text_by_default(tmp_path, fake_embedder):
    _, store = _populate(tmp_path, fake_embedder)
    rows = store.list_chunks()
    assert len(rows) == store.count()
    assert all("id" in r and "metadata" in r for r in rows)
    assert all("document" not in r for r in rows)
    assert any(r["metadata"].get("kind") == "ai_block" for r in rows)


def test_list_chunks_filter_and_text(tmp_path, fake_embedder):
    _, store = _populate(tmp_path, fake_embedder)
    rows = store.list_chunks(
        where={"$and": [{"board": {"$eq": "ASY011"}}, {"kind": {"$eq": "ai_block"}}]},
        include_documents=True,
    )
    assert rows
    assert all(r["metadata"]["kind"] == "ai_block" for r in rows)
    assert all("AI_START" in r["document"] for r in rows)
    # Wrong board returns nothing.
    assert store.list_chunks(where={"board": {"$eq": "NOPE"}}) == []


def test_inspect_cli_json_runs(tmp_path, fake_embedder, monkeypatch, capsys):
    index_path, _ = _populate(tmp_path, fake_embedder)
    monkeypatch.setenv("RAG_INDEX_PATH", index_path)
    rc = inspect_cli.main(["--board", "ASY011", "--json"])
    assert rc == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and payload
    assert all(r["metadata"]["board"] == "ASY011" for r in payload)


def test_inspect_cli_group_by(tmp_path, fake_embedder, monkeypatch, capsys):
    index_path, _ = _populate(tmp_path, fake_embedder)
    monkeypatch.setenv("RAG_INDEX_PATH", index_path)
    rc = inspect_cli.main(["--group-by", "kind"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "chunk counts by 'kind'" in out
    assert "ai_block" in out
