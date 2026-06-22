"""
rag/embeddings.py — Abstract, swappable embedding interface.

KEY PRINCIPLES
--------------
1. The embedder is a SEPARATE component from the agent's code-generation LLM.
   The LLM that writes C code is configured elsewhere and is NOT touched here.
   This module only turns text into vectors.

2. The provider is selected at runtime via ``RAG_EMBED_PROVIDER`` (``.env``).
   No provider is hardcoded into the call sites — everything goes through the
   :class:`Embedder` interface and the :func:`get_embedder` factory.

3. THE SAME EMBEDDER MUST BE USED FOR INDEXING AND QUERYING. Mixing models (or
   even the same model at a different dimensionality) silently destroys recall.
   To make accidental mixing detectable we expose ``.model`` and ``.signature``
   so the store/indexer can record which embedder produced a vector.

4. Local-first: the default provider talks to a LOCAL HTTP runtime, so no data
   leaves the machine. A cloud provider is left as an explicit extension stub.

The reference local implementation speaks the OpenAI-compatible
``POST /v1/embeddings`` shape, which is understood by many local runtimes
(llama.cpp server, LM Studio, vLLM, Text-Embeddings-Inference's OpenAI route,
and Ollama's ``/v1`` shim). That keeps it runtime-agnostic without binding us
to a single vendor.
"""

from __future__ import annotations

import abc
from typing import List, Optional, Sequence

import requests

from .config import RagConfig, load_config


class Embedder(abc.ABC):
    """Abstract embedding provider.

    Implementations must guarantee that :meth:`embed_documents` and
    :meth:`embed_query` produce vectors in the *same* space (same model, same
    dimensionality, same normalisation), so a query vector can be compared
    against indexed document vectors.
    """

    @property
    @abc.abstractmethod
    def model(self) -> str:
        """Human-readable model identifier (e.g. ``nomic-embed-text``)."""

    @property
    def signature(self) -> str:
        """Stable identifier stored alongside vectors.

        Used to detect "indexed with embedder A, queried with embedder B"
        mistakes. Format: ``"<provider>:<model>"``.
        """
        return f"{type(self).__name__}:{self.model}"

    @abc.abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of documents (used at INDEX time)."""

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (used at RETRIEVAL time)."""


class OpenAICompatibleEmbedder(Embedder):
    """Reference local embedder over the OpenAI ``/v1/embeddings`` shape.

    This is the DEFAULT provider. Point ``RAG_EMBED_BASE_URL`` at any local
    runtime that exposes ``POST {base_url}/embeddings`` returning
    ``{"data": [{"embedding": [...]}, ...]}``.

    ``api_key`` is optional: local runtimes typically ignore it, but we forward
    it as a Bearer token when present so the very same class also works against
    an authenticated endpoint without code changes.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 300,
    ) -> None:
        # timeout is in SECONDS and is supplied by the factory from
        # RAG_EMBED_TIMEOUT. Local CPU embedding of a large input can take
        # minutes, so this must not be a low hard-coded value.
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or None
        self._timeout = timeout
        self._session = requests.Session()

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        """Low-level call shared by document/query embedding.

        Keeping a single transport method guarantees that documents and
        queries are embedded through the exact same code path and model.
        """
        url = f"{self._base_url}/embeddings"
        resp = self._session.post(
            url,
            json={"model": self._model, "input": inputs},
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        # OpenAI returns results in `data`, preserving input order via `index`.
        data = sorted(payload["data"], key=lambda d: d.get("index", 0))
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class CloudEmbedderStub(Embedder):
    """Extension point for a hosted/cloud embedding provider.

    Intentionally NOT implemented: enabling it is a deliberate, reviewable
    decision because it means embedding text leaves the machine. To add one:

      1. Implement ``embed_documents`` / ``embed_query`` against the vendor SDK
         or REST API (read credentials from ``RagConfig`` / env, never inline).
      2. Register it in :func:`get_embedder` under a new ``RAG_EMBED_PROVIDER``
         value (e.g. ``"openai"``, ``"cohere"``).
      3. Re-index from scratch — vectors from a different model are NOT
         compatible with an index built by another embedder.

    # TODO (human): pick the cloud vendor (if any) and implement here. Keep the
    # local provider as the default so the system stays offline by default.
    """

    def __init__(self, *_args, **_kwargs) -> None:  # pragma: no cover - stub
        raise NotImplementedError(
            "Cloud embedding provider is not configured. Implement "
            "CloudEmbedderStub and register it in get_embedder(), or set "
            "RAG_EMBED_PROVIDER=local to use the on-machine runtime."
        )

    @property
    def model(self) -> str:  # pragma: no cover - stub
        raise NotImplementedError

    def embed_documents(self, texts):  # pragma: no cover - stub
        raise NotImplementedError

    def embed_query(self, text):  # pragma: no cover - stub
        raise NotImplementedError


# Registry of known providers. Add new providers here, never at the call site.
_PROVIDERS = {
    "local": OpenAICompatibleEmbedder,
    "openai_compatible": OpenAICompatibleEmbedder,
    "cloud": CloudEmbedderStub,
}


def get_embedder(config: Optional[RagConfig] = None) -> Embedder:
    """Factory: build the configured :class:`Embedder` from ``.env``.

    The SAME factory is used by the indexer (offline) and the retriever
    (online), which guarantees both sides use the identical provider/model.
    """
    cfg = config or load_config()
    provider_cls = _PROVIDERS.get(cfg.embed_provider)
    if provider_cls is None:
        raise ValueError(
            f"Unknown RAG_EMBED_PROVIDER={cfg.embed_provider!r}. "
            f"Known providers: {sorted(_PROVIDERS)}"
        )
    if provider_cls is OpenAICompatibleEmbedder:
        return OpenAICompatibleEmbedder(
            base_url=cfg.embed_base_url,
            model=cfg.embed_model,
            api_key=cfg.embed_api_key,
            timeout=cfg.embed_timeout,
        )
    # Cloud / future providers construct themselves from config.
    return provider_cls(config=cfg)
