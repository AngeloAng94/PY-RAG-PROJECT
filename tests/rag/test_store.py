"""Tests for rag.store — add + query with composed metadata filters."""

from __future__ import annotations

import pytest

from rag.store import ChromaStore


@pytest.fixture
def store(tmp_path, fake_embedder):
    s = ChromaStore(index_path=str(tmp_path / "idx"), collection_name="test_col")
    s.reset()

    docs = [
        "draw header label coffee screen",
        "handle start button click event",
        "oven temperature control loop",
    ]
    metas = [
        {"board": "ASY011", "micro": "STM32H750", "categoria": "caffe", "layer": "ui"},
        {"board": "ASY011", "micro": "STM32H750", "categoria": "caffe", "layer": "app"},
        {"board": "ASY099", "micro": "STM32F4", "categoria": "forno", "layer": "app"},
    ]
    embs = fake_embedder.embed_documents(docs)
    s.add(ids=["a", "b", "c"], embeddings=embs, documents=docs, metadatas=metas)
    return s


def test_add_and_count(store):
    assert store.count() == 3


def test_query_respects_board_filter(store, fake_embedder):
    # Query about an oven, but constrain to the coffee board: the oven chunk
    # (different board) must NOT come back — board boundary is hard.
    q = fake_embedder.embed_query("temperature control")
    res = store.query(
        embedding=q,
        k=5,
        where={"$and": [{"board": {"$eq": "ASY011"}}, {"micro": {"$eq": "STM32H750"}}]},
    )
    ids = {r["id"] for r in res}
    assert "c" not in ids
    assert ids <= {"a", "b"}


def test_query_composed_or_filter(store, fake_embedder):
    q = fake_embedder.embed_query("anything")
    res = store.query(
        embedding=q,
        k=5,
        where={"$or": [{"layer": {"$eq": "ui"}}, {"layer": {"$eq": "forno-only"}}]},
    )
    ids = {r["id"] for r in res}
    assert ids == {"a"}


def test_reset_clears(store):
    store.reset()
    assert store.count() == 0
