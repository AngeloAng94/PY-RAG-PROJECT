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
together.

``categoria`` and ``cliente`` are LAYERED FALL-THROUGH dimensions: they are
single-valued, but their filter matches "target value OR ABSENT" so that shared
chunks survive a narrower query. A ``comune`` chunk (indexed with
``categoria=ABSENT``) is still returned when ``categoria=forno`` is pinned, while
a different category like ``caffe`` is excluded. See rag/constants.py.

The filter is composed with Chroma's ``$and`` / ``$in`` / ``$or`` operators so
additional dimensions can be added without changing the store.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .constants import ABSENT
from .embeddings import Embedder
from .store import ChromaStore


def _eq_or_absent(field: str, value: str) -> Dict:
    """Match ``field == value`` OR ``field == ABSENT`` (the "shared" sentinel).

    This is the heart of the layered model: pinning ``categoria=forno`` must
    still return shared ``comune`` chunks (stored with ``categoria=ABSENT``)
    while EXCLUDING a different category like ``caffe``. A bare ``$eq`` would
    drop the shared chunks; a ``$ne``/``$nin`` would wrongly admit other
    categories (and Chroma has no ``$exists``). The explicit ``$or`` with the
    sentinel is the precise expression. See rag/constants.py.
    """
    return {"$or": [{field: {"$eq": value}}, {field: {"$eq": ABSENT}}]}


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

    Layered fall-through (``$eq`` value OR ``ABSENT``) dimensions — pass a string:
        * ``categoria`` (one product family; shared ``comune`` chunks survive)
        * ``cliente``   (one customer; ``comune``/``categoria`` chunks survive)

    DECISION (resolves the comune/categoria tension): shared chunks must NOT be
    excluded when ``categoria`` is pinned. ``comune`` chunks are indexed with
    ``categoria=ABSENT`` (and likewise ``cliente=ABSENT`` for non-customer code),
    so the categoria/cliente filters match "target value OR ABSENT". A different
    category is still excluded. board/micro stay mandatory and never fall back.
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

    # Layered fall-through dimensions: target value OR the ABSENT sentinel, so
    # shared/less-specific chunks are not dropped by a narrower filter.
    if categoria:
        clauses.append(_eq_or_absent("categoria", categoria))
    if cliente:
        clauses.append(_eq_or_absent("cliente", cliente))

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
