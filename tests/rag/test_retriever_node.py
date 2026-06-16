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


def test_repair_pass_enriches_query_with_compile_error(indexed_env):
    base_state = {
        "user_request": "draw header label coffee screen",
        "target_file": indexed_env,
        "source_content": SAMPLE_C,
        "board": "ASY011",
        "micro": "STM32H750",
        "categoria": "caffe",
    }
    # First pass: no compile error.
    out_first = rn.retrieve(dict(base_state))
    q_first = out_first["retrieval_debug"]["query"]
    assert out_first["retrieval_debug"]["repair_pass"] is False

    # Repair pass: same state PLUS a failed compile result.
    repair_state = dict(base_state)
    repair_state["compile_result"] = {
        "success": False,
        "stdout": "",
        "stderr": "screen_caffe.c:42: error: 'lv_label_set_text' undeclared",
    }
    out_repair = rn.retrieve(repair_state)
    q_repair = out_repair["retrieval_debug"]["query"]

    # The query must differ and carry a concise form of the error.
    assert q_repair != q_first
    assert out_repair["retrieval_debug"]["repair_pass"] is True
    assert "lv_label_set_text" in q_repair
    assert q_first in q_repair  # original intent preserved, error appended
    # Stays short — never blows the embedding input.
    assert len(q_repair) <= len(q_first) + rn._ERROR_HINT_MAX_CHARS + 40


def test_successful_compile_does_not_enrich_query(indexed_env):
    state = {
        "user_request": "draw header label",
        "target_file": indexed_env,
        "source_content": SAMPLE_C,
        "board": "ASY011",
        "micro": "STM32H750",
        "categoria": "caffe",
        "compile_result": {"success": True, "stdout": "ok", "stderr": ""},
    }
    out = rn.retrieve(state)
    assert out["retrieval_debug"]["repair_pass"] is False
    assert out["retrieval_debug"]["query"] == "draw header label"
