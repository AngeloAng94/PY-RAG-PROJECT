"""
rag/eval.py — Tiny recall@k harness for the retrieval engine.

Retrieval quality must be measurable, not vibes. This harness runs a list of
labelled cases against the live store/embedder and reports recall@k: for each
case, did the expected chunk id appear in the top-k results?

The EVAL_SET below ships with PLACEHOLDER examples only. Real cases — a query
plus the chunk id(s) a human considers correct under a given board/micro scope
— are filled in by the team against the actual index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .embeddings import Embedder
from .query import retrieve_relevant
from .store import ChromaStore


@dataclass
class EvalCase:
    """One labelled retrieval expectation."""

    query: str
    board: str
    micro: str
    # ids the human considers a correct hit (any one in top-k counts).
    expected_ids: List[str]
    scope: Optional[str] = None
    categoria: Optional[str] = None
    cliente: Optional[str] = None
    note: str = ""


# TODO (human): replace these placeholders with REAL cases captured from the
# actual repositories. Each case needs a query, the board/micro scope, and the
# chunk id(s) that should be retrieved. Build the index first, then read ids
# from the store to populate `expected_ids`.
EVAL_SET: List[EvalCase] = [
    EvalCase(
        query="draw the main brewing screen header label",
        board="ASY011",
        micro="STM32H750",
        categoria="caffe",
        expected_ids=["<PLACEHOLDER_CHUNK_ID_1>"],
        note="placeholder — fill from real index",
    ),
    EvalCase(
        query="handle button click event for start button",
        board="ASY011",
        micro="STM32H750",
        categoria="caffe",
        expected_ids=["<PLACEHOLDER_CHUNK_ID_2>"],
        note="placeholder — fill from real index",
    ),
]


@dataclass
class RecallReport:
    k: int
    total: int
    hits: int
    per_case: List[Dict] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return (self.hits / self.total) if self.total else 0.0


def recall_at_k(
    store: ChromaStore,
    embedder: Embedder,
    k: int,
    eval_set: Optional[List[EvalCase]] = None,
) -> RecallReport:
    """Run recall@k over ``eval_set`` (defaults to :data:`EVAL_SET`).

    A case is a "hit" if ANY of its ``expected_ids`` appears in the top-k
    retrieved chunk ids for its query+scope.
    """
    cases = eval_set if eval_set is not None else EVAL_SET
    report = RecallReport(k=k, total=len(cases), hits=0)

    for case in cases:
        results = retrieve_relevant(
            store=store,
            embedder=embedder,
            query=case.query,
            scope=case.scope,
            categoria=case.categoria,
            cliente=case.cliente,
            board=case.board,
            micro=case.micro,
            k=k,
        )
        retrieved_ids = [r["id"] for r in results]
        hit = any(eid in retrieved_ids for eid in case.expected_ids)
        if hit:
            report.hits += 1
        report.per_case.append(
            {
                "query": case.query,
                "expected_ids": case.expected_ids,
                "retrieved_ids": retrieved_ids,
                "hit": hit,
            }
        )

    return report
