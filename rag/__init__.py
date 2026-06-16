"""
rag/ — Retrieval-Augmented Generation infrastructure for the embedded
code-generation agent.

This package provides the *retrieval engine* that replaces the old, naive
`retrieve` node (which concatenated whole files, uncapped, into
`state["full_context"]`).

Design goals (see the individual modules for details):

* **Modular & reviewable** — every concern lives in its own module with
  docstrings and explicit ``# TODO (human):`` markers where domain
  calibration is required. This is scaffolding, not a black box.
* **Local-first** — by default no data leaves the machine. Embeddings are
  produced by a local HTTP runtime through an abstract, swappable interface.
* **Offline indexing vs. online retrieval are separate** — see
  ``rag.indexer`` (offline ingest) vs. ``rag.query`` / ``rag.retriever_node``
  (online retrieval).
* **Metadata-filter BEFORE similarity** — retrieval always narrows the
  candidate set by metadata (board/micro are *mandatory*) and only then
  ranks by vector similarity. Crossing a board boundary is an error, not a
  suggestion.

Public surface:

    from rag.embeddings import Embedder, get_embedder
    from rag.chunker import chunk_c_source, Chunk
    from rag.store import ChromaStore
    from rag.indexer import index_file, index_repo, infer_layer
    from rag.query import retrieve_relevant
    from rag.retriever_node import retrieve  # drop-in graph node
    from rag.eval import recall_at_k
"""

from .config import RagConfig, load_config  # noqa: F401

__all__ = ["RagConfig", "load_config"]
