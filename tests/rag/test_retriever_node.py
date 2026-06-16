"""Smoke test for the drop-in rag.retriever_node.retrieve graph node."""

from __future__ import annotations

import pytest

import rag.retriever_node as rn
from rag.indexer import index_file
from rag.store import ChromaStore
from tests.rag.conftest import SAMPLE_C


@pytest.fixture
def indexed_env(tmp_path, fake_embedder, monkeypatch):
    """Build a small on-disk index and point the node's config at it."""
    index_path = str(tmp_path / "idx")
    target_file = tmp_path / "screen_caffe.c"
    target_file.write_text(SAMPLE_C, encoding="utf-8")

    store = ChromaStore(index_path=index_path)
    store.reset()
    index_file(
        str(target_file),
        store,
        fake_embedder,
        base_metadata={
            "board": "ASY011",
            "micro": "STM32H750",
            "categoria": "caffe",
            "scope": "categoria",
        },
    )

    # Configure the node to use our temp index + the deterministic embedder.
    monkeypatch.setenv("RAG_INDEX_PATH", index_path)
    monkeypatch.setattr(rn, "get_embedder", lambda cfg=None: fake_embedder)
    # Reload config inside the node by patching load_config to read env fresh.
    return str(target_file)


def test_retrieve_populates_full_context_and_debug(indexed_env):
    state = {
        "user_request": "draw header label coffee screen",
        "target_view": "caffe",
        "target_file": indexed_env,
        "source_content": SAMPLE_C,
        "board": "ASY011",
        "micro": "STM32H750",
        "categoria": "caffe",
        "scope": "categoria",
        "iterations": 0,
        "max_iterations": 3,
    }

    out = rn.retrieve(state)

    # Contract preserved: full_context populated.
    assert out["full_context"]
    assert "TARGET FILE" in out["full_context"]

    # Additive fields present, existing fields untouched.
    assert "retrieved_chunks" in out
    assert "retrieval_debug" in out
    assert out["max_iterations"] == 3
    assert out["retrieval_debug"]["board"] == "ASY011"
    assert out["retrieval_debug"]["status"] in ("ok", "skipped")


def test_budget_caps_examples(indexed_env, monkeypatch):
    # Tiny budget -> examples block must stay within it.
    monkeypatch.setenv("RAG_MAX_EXAMPLE_CHARS", "120")
    state = {
        "user_request": "draw header label",
        "target_file": indexed_env,
        "source_content": SAMPLE_C,
        "board": "ASY011",
        "micro": "STM32H750",
        "categoria": "caffe",
    }
    out = rn.retrieve(state)
    assert out["retrieval_debug"]["examples_chars"] <= 120 + 200  # whole-chunk slack


def test_missing_board_skips_retrieval(indexed_env):
    state = {
        "user_request": "draw something",
        "target_file": indexed_env,
        "source_content": SAMPLE_C,
        # no board/micro -> must NOT cross board boundaries
    }
    out = rn.retrieve(state)
    assert out["retrieval_debug"]["status"] == "skipped"
    assert out["retrieved_chunks"] == []
    assert "TARGET FILE" in out["full_context"]
