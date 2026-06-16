"""
rag/store.py — ChromaDB persistence wrapper.

A thin, explicit wrapper around a ChromaDB ``PersistentClient`` (on-disk). We
keep the surface tiny and obvious:

    add(ids, embeddings, documents, metadatas)
    query(embedding, k, where)
    reset()

The ``where`` filter is passed straight through to Chroma and therefore
supports COMPOSED metadata filters using ``$and`` / ``$or`` / ``$eq`` / ``$in``,
e.g.::

    {"$and": [
        {"board": {"$eq": "ASY011"}},
        {"micro": {"$eq": "STM32H750"}},
        {"$or": [{"layer": {"$eq": "ui"}}, {"layer": {"$eq": "app"}}]},
    ]}

We pass ``embeddings`` explicitly (we never let Chroma compute embeddings) so
the SAME external embedder is the single source of truth for both indexing and
querying.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import chromadb
from chromadb.config import Settings


class ChromaStore:
    """On-disk vector store backed by ``chromadb.PersistentClient``."""

    def __init__(
        self,
        index_path: str,
        collection_name: str = "embedded_code",
    ) -> None:
        self._index_path = index_path
        self._collection_name = collection_name
        # telemetry disabled: this is a local-first, offline-by-default system.
        self._client = chromadb.PersistentClient(
            path=index_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # Cosine distance is the conventional default for text embeddings.
            metadata={"hnsw:space": "cosine"},
        )

    # -- write ---------------------------------------------------------------
    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Dict],
    ) -> None:
        """Upsert chunks. Using ``upsert`` makes re-indexing idempotent."""
        if not ids:
            return
        self._collection.upsert(
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],
            documents=list(documents),
            metadatas=[_sanitize_metadata(m) for m in metadatas],
        )

    # -- read ----------------------------------------------------------------
    def query(
        self,
        embedding: Sequence[float],
        k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Vector search with an optional composed metadata filter.

        Returns a list of dicts: ``{id, document, metadata, distance}`` sorted
        by ascending distance (closest first).
        """
        res = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma returns parallel lists, one row per query embedding. We sent 1.
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: List[Dict] = []
        for i, _id in enumerate(ids):
            out.append(
                {
                    "id": _id,
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return out

    # -- maintenance ---------------------------------------------------------
    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection (used by indexer --reset / tests)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def _sanitize_metadata(meta: Dict) -> Dict:
    """Chroma only accepts scalar metadata values (str/int/float/bool).

    Drop ``None`` values (Chroma rejects them) and coerce anything exotic to a
    string so a stray non-scalar never breaks an ingest run.
    """
    clean: Dict[str, object] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean
