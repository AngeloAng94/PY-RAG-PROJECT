"""Tests for rag.config defaults + that the embedder reads the configured timeout."""

from __future__ import annotations

from rag.config import load_config
from rag.embeddings import OpenAICompatibleEmbedder, get_embedder


def test_new_config_defaults(monkeypatch):
    # Ensure the two new keys are unset so we read the documented defaults.
    monkeypatch.delenv("RAG_EMBED_TIMEOUT", raising=False)
    monkeypatch.delenv("RAG_MAX_CHUNK_CHARS", raising=False)
    cfg = load_config()
    assert cfg.embed_timeout == 300          # seconds (was a hard-coded 60)
    assert cfg.max_chunk_chars == 12000


def test_embed_timeout_is_read_from_config(monkeypatch):
    monkeypatch.setenv("RAG_EMBED_TIMEOUT", "450")
    monkeypatch.setenv("RAG_EMBED_PROVIDER", "local")
    cfg = load_config()
    assert cfg.embed_timeout == 450

    embedder = get_embedder(cfg)
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    # The factory must wire the configured timeout into the embedder.
    assert embedder._timeout == 450


def test_embed_timeout_default_wired_into_embedder(monkeypatch):
    monkeypatch.delenv("RAG_EMBED_TIMEOUT", raising=False)
    monkeypatch.setenv("RAG_EMBED_PROVIDER", "local")
    embedder = get_embedder(load_config())
    assert embedder._timeout == 300
