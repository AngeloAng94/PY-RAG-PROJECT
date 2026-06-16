"""
rag/query.py — ONLINE retrieval with a layered metadata filter.

This is the runtime read path. The cardinal rule of the knowledge model:

    FILTER BY METADATA *BEFORE* RANKING BY SIMILARITY.

Code is cross-compiled for one specific MCU and runs only on one board.
Retrieving another board's code is an ERROR, not a weaker suggestion. So
``board`` and ``micro`` are MANDATORY here — we refuse to query without them.

``scope`` works as a layered narrowing dimension:

    comune     -> shared code, applies across categorie/clienti
    categoria  -> product family (caffe, forno, tosaerba, ...)
    cliente    -> customer-specific code

The filter is composed with Chroma's ``$and`` / ``$or`` operators so additional
dimensions can be added without changing the store.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .embeddings import Embedder
from .store import ChromaStore


def build_where(
    *,
    board: str,
    micro: str,
    scope: Optional[str] = None,
    categoria: Optional[str] = None,
    cliente: Optional[str] = None,
) -> Dict:
    """Compose the Chroma ``where`` filter from the session dimensions.

    ``board`` and ``micro`` are required and combined with ``$and``. Optional
    dimensions are added only when provided, so omitting (say) ``cliente`` means
    "don't constrain on customer" rather than "match empty customer".
    """
    if not board or not micro:
        # Hard guard: never run an unconstrained cross-board search.
        raise ValueError(
            "board and micro are mandatory for retrieval — refusing to search "
            "across board boundaries."
        )

    clauses: List[Dict] = [
        {"board": {"$eq": board}},
        {"micro": {"$eq": micro}},
    ]
    if scope:
        clauses.append({"scope": {"$eq": scope}})
    if categoria:
        clauses.append({"categoria": {"$eq": categoria}})
    if cliente:
        clauses.append({"cliente": {"$eq": cliente}})

    # A single clause must not be wrapped in $and (Chroma rejects 1-item $and).
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve_relevant(
    store: ChromaStore,
    embedder: Embedder,
    query: str,
    scope: Optional[str] = None,
    categoria: Optional[str] = None,
    cliente: Optional[str] = None,
    board: Optional[str] = None,
    micro: Optional[str] = None,
    k: int = 5,
) -> List[Dict]:
    """Embed ``query`` and return the top-``k`` chunks within the metadata scope.

    Returns the store's result dicts: ``{id, document, metadata, distance}``.

    Order of operations (NON-NEGOTIABLE):
      1. Build the metadata filter (board/micro mandatory).
      2. Embed the query with the SAME embedder used to build the index.
      3. Let Chroma apply the filter, THEN rank survivors by similarity.
    """
    where = build_where(
        board=board,
        micro=micro,
        scope=scope,
        categoria=categoria,
        cliente=cliente,
    )
    query_embedding = embedder.embed_query(query)
    return store.query(embedding=query_embedding, k=k, where=where)
