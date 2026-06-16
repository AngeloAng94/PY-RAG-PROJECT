"""
rag/config.py — Centralised, typed access to RAG configuration.

All tunables come from environment variables (``.env``). We deliberately keep
*one* place that reads the environment so the rest of the package never calls
``os.environ`` directly. This makes the configuration surface obvious to a
human reviewer and trivial to override in tests.

Recognised ``.env`` keys (see ``.env.rag.template``):

    RAG_INDEX_PATH          On-disk path for the ChromaDB PersistentClient.
    RAG_EMBED_PROVIDER      Which Embedder implementation to instantiate.
    RAG_EMBED_MODEL         Embedding model name passed to the provider.
    RAG_EMBED_BASE_URL      Base URL of the local/remote embedding runtime.
    RAG_EMBED_API_KEY       Optional bearer token (local runtimes ignore it).
    RAG_TOP_K               Default number of chunks to retrieve.
    RAG_MAX_EXAMPLE_CHARS   Hard budget for retrieved example text in context.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    # python-dotenv is already a backend dependency; loading is best-effort so
    # the package also works when the host injects env vars directly.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


# Sensible defaults. They are intentionally conservative and local-only.
_DEFAULTS = {
    "RAG_INDEX_PATH": "./.rag_index",
    "RAG_EMBED_PROVIDER": "local",
    "RAG_EMBED_MODEL": "nomic-embed-text",
    # OpenAI-compatible shape: many local runtimes (llama.cpp, LM Studio,
    # Ollama's /v1 shim, TEI's OpenAI route, vLLM) expose POST /v1/embeddings.
    "RAG_EMBED_BASE_URL": "http://localhost:11434/v1",
    "RAG_EMBED_API_KEY": "",
    "RAG_TOP_K": "5",
    "RAG_MAX_EXAMPLE_CHARS": "8000",
}


def _get(key: str) -> str:
    return os.environ.get(key, _DEFAULTS[key])


@dataclass(frozen=True)
class RagConfig:
    """Immutable snapshot of the RAG configuration."""

    index_path: str
    embed_provider: str
    embed_model: str
    embed_base_url: str
    embed_api_key: str
    top_k: int
    max_example_chars: int


def load_config() -> RagConfig:
    """Read the environment and return a validated :class:`RagConfig`."""

    return RagConfig(
        index_path=_get("RAG_INDEX_PATH"),
        embed_provider=_get("RAG_EMBED_PROVIDER").strip().lower(),
        embed_model=_get("RAG_EMBED_MODEL"),
        embed_base_url=_get("RAG_EMBED_BASE_URL").rstrip("/"),
        embed_api_key=_get("RAG_EMBED_API_KEY"),
        top_k=int(_get("RAG_TOP_K")),
        max_example_chars=int(_get("RAG_MAX_EXAMPLE_CHARS")),
    )
