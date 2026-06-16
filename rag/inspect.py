#!/usr/bin/env python3
"""
rag/inspect.py — READ-ONLY, OFFLINE index auditor.

Dumps chunk ids + metadata from a built ChromaDB index so a human can:

  * audit WHAT crossed into each scope (catch a chunk tagged with the wrong
    board/cliente before it ever reaches retrieval), and
  * quickly assemble the real EVAL_SET by copying correct chunk ids.

It NEVER writes, embeds, or contacts a model runtime — it only reads the
on-disk store via ``ChromaStore.list_chunks`` (Chroma ``get``). Safe to run any
time against a populated index.

Examples
--------
    # Everything in the index (table view):
    python rag/inspect.py

    # Only one board, show line ranges + symbols:
    python rag/inspect.py --board ASY011

    # Narrow by scope/cliente and dump as JSON (e.g. to seed EVAL_SET ids):
    python rag/inspect.py --board ASY011 --scope cliente --cliente acme --json

    # Show the chunk text too (verbose):
    python rag/inspect.py --board ASY011 --show-text --limit 20

    # Count how many chunks exist per board (audit cross-scope leakage):
    python rag/inspect.py --group-by board
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

# Allow running as `python rag/inspect.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import load_config  # noqa: E402
from rag.store import ChromaStore  # noqa: E402

# Filterable metadata dimensions exposed as CLI flags. Each maps to an $eq
# clause; together they are combined with $and (filter BEFORE any reading).
_FILTER_DIMS = ("scope", "categoria", "cliente", "costruttore", "board", "micro", "layer", "kind")


def _build_where(args: argparse.Namespace) -> Optional[Dict]:
    clauses: List[Dict] = []
    for dim in _FILTER_DIMS:
        value = getattr(args, dim, None)
        if value:
            clauses.append({dim: {"$eq": value}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _print_table(rows: List[Dict], show_text: bool) -> None:
    if not rows:
        print("(no chunks matched)")
        return
    header = f"{'BOARD':<10} {'MICRO':<12} {'SCOPE':<10} {'CATEGORIA':<10} {'KIND':<10} {'SYMBOL':<22} {'LINES':<11} ID"
    print(header)
    print("-" * len(header))
    for r in rows:
        m = r.get("metadata", {}) or {}
        lines = f"{m.get('start_line', '?')}-{m.get('end_line', '?')}"
        symbol = str(m.get("symbol", ""))[:22]
        print(
            f"{str(m.get('board','')):<10} "
            f"{str(m.get('micro','')):<12} "
            f"{str(m.get('scope','')):<10} "
            f"{str(m.get('categoria','')):<10} "
            f"{str(m.get('kind','')):<10} "
            f"{symbol:<22} "
            f"{lines:<11} "
            f"{r['id']}"
        )
        if show_text:
            text = r.get("document", "") or ""
            indented = "\n".join("    " + ln for ln in text.splitlines())
            print(indented)
            print("    " + "-" * 60)


def _print_group_counts(rows: List[Dict], group_by: str) -> None:
    counter: Counter = Counter()
    for r in rows:
        counter[str((r.get("metadata", {}) or {}).get(group_by, "<unset>"))] += 1
    print(f"chunk counts by '{group_by}':")
    for value, n in counter.most_common():
        print(f"  {value:<24} {n}")
    print(f"  {'TOTAL':<24} {sum(counter.values())}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only, offline dump of chunk ids + metadata from a built RAG index."
    )
    # Optional metadata filters (all $eq, combined with $and).
    parser.add_argument("--board", help="Filter by board, e.g. ASY011.")
    parser.add_argument("--micro", help="Filter by MCU, e.g. STM32H750.")
    parser.add_argument("--scope", help="Filter by scope (comune/categoria/cliente).")
    parser.add_argument("--categoria", help="Filter by product family.")
    parser.add_argument("--cliente", help="Filter by customer id.")
    parser.add_argument("--costruttore", help="Filter by manufacturer id.")
    parser.add_argument("--layer", help="Filter by layer (hal/bsp/rtos/middleware/app/ui).")
    parser.add_argument("--kind", help="Filter by chunk kind (function/struct/ai_block/...).")

    parser.add_argument("--limit", type=int, help="Max chunks to return.")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset.")
    parser.add_argument("--show-text", action="store_true", help="Also print chunk text.")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of a table.")
    parser.add_argument(
        "--group-by",
        choices=_FILTER_DIMS,
        help="Instead of listing rows, print chunk counts grouped by this dimension.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config()

    store = ChromaStore(index_path=cfg.index_path)
    where = _build_where(args)

    # --group-by needs metadata for every matching chunk; never load text there.
    need_text = bool(args.show_text) and not args.group_by
    rows = store.list_chunks(
        where=where,
        limit=args.limit,
        offset=args.offset,
        include_documents=need_text,
    )

    if args.group_by:
        _print_group_counts(rows, args.group_by)
        return 0

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    print(f"index_path={cfg.index_path}  total_in_store={store.count()}  matched={len(rows)}")
    if where:
        print(f"filter={where}")
    print()
    _print_table(rows, show_text=need_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
