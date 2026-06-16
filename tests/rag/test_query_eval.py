"""Tests for rag.query + rag.eval — layered filter and recall@k harness."""

from __future__ import annotations

import pytest

from rag.eval import EvalCase, recall_at_k
from rag.query import build_where, retrieve_relevant
from rag.store import ChromaStore


def test_build_where_requires_board_and_micro():
    with pytest.raises(ValueError):
        build_where(board="", micro="STM32H750")
    with pytest.raises(ValueError):
        build_where(board="ASY011", micro="")


def test_build_where_composes_optional_dims():
    where = build_where(
        board="ASY011",
        micro="STM32H750",
        scope="categoria",
        categoria="caffe",
    )
    assert "$and" in where
    clause_keys = [list(c.keys())[0] for c in where["$and"]]
    assert "board" in clause_keys and "micro" in clause_keys
    assert "scope" in clause_keys and "categoria" in clause_keys


@pytest.fixture
def populated_store(tmp_path, fake_embedder):
    s = ChromaStore(index_path=str(tmp_path / "idx"), collection_name="q_col")
    s.reset()
    docs = [
        "draw header label coffee brewing screen",
        "handle start button click event coffee",
        "oven temperature control loop forno",
    ]
    metas = [
        {"board": "ASY011", "micro": "STM32H750", "categoria": "caffe"},
        {"board": "ASY011", "micro": "STM32H750", "categoria": "caffe"},
        {"board": "ASY099", "micro": "STM32F4", "categoria": "forno"},
    ]
    s.add(
        ids=["draw1", "evt1", "oven1"],
        embeddings=fake_embedder.embed_documents(docs),
        documents=docs,
        metadatas=metas,
    )
    return s


def test_retrieve_relevant_filters_then_ranks(populated_store, fake_embedder):
    res = retrieve_relevant(
        store=populated_store,
        embedder=fake_embedder,
        query="draw header label brewing",
        board="ASY011",
        micro="STM32H750",
        categoria="caffe",
        k=5,
    )
    ids = [r["id"] for r in res]
    assert "oven1" not in ids  # filtered out by board
    assert ids[0] == "draw1"  # best similarity ranked first


def test_recall_at_k_with_real_ids(populated_store, fake_embedder):
    cases = [
        EvalCase(
            query="draw header label brewing",
            board="ASY011",
            micro="STM32H750",
            categoria="caffe",
            expected_ids=["draw1"],
        ),
        EvalCase(
            query="start button click",
            board="ASY011",
            micro="STM32H750",
            categoria="caffe",
            expected_ids=["evt1"],
        ),
    ]
    report = recall_at_k(populated_store, fake_embedder, k=3, eval_set=cases)
    assert report.total == 2
    assert report.recall == 1.0
