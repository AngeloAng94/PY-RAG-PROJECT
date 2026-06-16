"""
rag/constants.py — Shared knowledge-model constants.

ABSENT sentinel
---------------
Our layered knowledge model needs "shared" code to survive a narrower filter:
``comune`` chunks (no product family) must still be retrieved when a session
pins ``categoria=forno``; ``categoria``-level chunks (no customer) must survive
a ``cliente`` filter; and so on.

The natural way to model that is "this dimension is not applicable here". But
ChromaDB cannot match a *missing* metadata key precisely:

  * there is no ``$exists`` operator, and
  * ``$ne`` / ``$nin`` match a missing key BUT also match every OTHER value —
    so ``categoria $ne forno`` would wrongly let ``caffe`` through.

So instead of leaving the field unset we store an explicit, normalized sentinel
value ``ABSENT`` for the "not applicable" case. Retrieval then matches
``categoria == X OR categoria == ABSENT``, which returns the target category
plus shared chunks while still EXCLUDING a different category.

FALLBACK_DIMS are the dimensions that follow this layered fall-through.
``board`` / ``micro`` are NOT here on purpose — they are mandatory and must
never fall back (crossing a board boundary is an error).
"""

# Normalized value meaning "this dimension does not apply to this chunk".
# Chosen to be something no real categoria/cliente id would ever equal.
ABSENT = "__none__"

# Dimensions that fall back to ABSENT when not provided at index time, and that
# match "target value OR ABSENT" at query time.
FALLBACK_DIMS = ("categoria", "cliente")
