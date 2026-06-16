"""
rag/query.py — ONLINE retrieval with a layered metadata filter.

This is the runtime read path. The cardinal rule of the knowledge model:

    FILTER BY METADATA *BEFORE* RANKING BY SIMILARITY.

Code is cross-compiled for one specific MCU and runs only on one board.
Retrieving another board's code is an ERROR, not a weaker suggestion. So
``board`` and ``micro`` are MANDATORY here — we refuse to query without them.

``scope`` is a LAYERED, COMPOSABLE dimension. Customer work needs several
layers AT ONCE — the customer's own code PLUS shared ``comune`` code PLUS the
``categoria`` (product family) code. So ``scope`` is a *list* matched with
Chroma's ``$in`` (any-of), not a single ``$eq`` value:

    comune     -> shared code, applies across categorie/clienti
    categoria  -> product family (caffe, forno, tosaerba, ...)
    cliente    -> customer-specific code

    e.g. scope=["comune", "categoria", "cliente"]  -> all three layers together

``layer`` (hal/bsp/rtos/middleware/app/ui) and ``costruttore`` use the same
any-of ``$in`` logic, so a session can pull, say, the ``ui`` and ``app`` layers
together. ``categoria`` and ``cliente`` stay single-valued (``$eq``): a session
targets exactly one product family / one customer.

The filter is composed with Chroma's ``$and`` / ``$in`` operators so additional
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
    scope: Optional[List[str]] = None,
    categoria: Optional[str] = None,
    cliente: Optional[str] = None,
    costruttore: Optional[List[str]] = None,
    layer: Optional[List[str]] = None,
) -> Dict:
    """Compose the Chroma ``where`` filter from the session dimensions.

    ``board`` and ``micro`` are required and combined with ``$and``. Optional
    dimensions are added only when provided, so omitting one means "don't
    constrain on it" rather than "match empty".

    Composable (any-of, ``$in``) dimensions — pass a list:
        * ``scope``       -> e.g. ["comune", "categoria", "cliente"]
        * ``layer``       -> e.g. ["ui", "app"]
        * ``costruttore`` -> e.g. ["acme-srl"]

    Single-valued (``$eq``) dimensions — pass a string:
        * ``categoria`` (one product family)
        * ``cliente``   (one customer)

    # TODO (human): modelling tension to calibrate on the real repos. When
    # scope spans layers (comune + categoria + cliente) but ``categoria`` is
    # also pinned with $eq, shared ``comune`` chunks that carry no categoria
    # would be excluded by the AND. Decide whether comune chunks should be
    # stored WITHOUT a categoria (so they survive) or whether categoria should
    # itself become an $in including a neutral value. Left explicit on purpose.
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

    # Composable, any-of layers (list -> $in). Empty/None means "all".
    if scope:
        clauses.append({"scope": {"$in": list(scope)}})
    if layer:
        clauses.append({"layer": {"$in": list(layer)}})
    if costruttore:
        clauses.append({"costruttore": {"$in": list(costruttore)}})

    # Single-valued session dimensions (string -> $eq).
    if categoria:
        clauses.append({"categoria": {"$eq": categoria}})
    if cliente:
        clauses.append({"cliente": {"$eq": cliente}})

    # A single clause must not be wrapped in $and (Chroma rejects 1-item $and).
    # (board+micro guarantee >= 2 clauses today; kept defensive.)
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve_relevant(
    store: ChromaStore,
    embedder: Embedder,
    query: str,
    scope: Optional[List[str]] = None,
    categoria: Optional[str] = None,
    cliente: Optional[str] = None,
    board: Optional[str] = None,
    micro: Optional[str] = None,
    k: int = 5,
    costruttore: Optional[List[str]] = None,
    layer: Optional[List[str]] = None,
) -> List[Dict]:
    """Embed ``query`` and return the top-``k`` chunks within the metadata scope.

    Returns the store's result dicts: ``{id, document, metadata, distance}``.

    ``scope`` / ``layer`` / ``costruttore`` are composable lists (any-of), so
    ``scope=["comune", "categoria", "cliente"]`` retrieves all three layers
    together within the mandatory board/micro filter.

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
        costruttore=costruttore,
        layer=layer,
    )
    query_embedding = embedder.embed_query(query)
    return store.query(embedding=query_embedding, k=k, where=where)
