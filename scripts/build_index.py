#!/usr/bin/env python3
"""
scripts/build_index.py — CLI to build the RAG index from a repository.

This is the human-operated OFFLINE entry point. It walks a repo of C/H files,
chunks them, derives metadata, embeds with the configured local embedder, and
writes vectors to the on-disk ChromaDB store.

The base metadata passed here describes the WHOLE repo being ingested
(scope/categoria/cliente/costruttore/board/micro). Per-file fields like
``layer`` are inferred by the indexer heuristic.

Examples
--------
    # Build (append/upsert) into the configured index:
    python scripts/build_index.py --repo /path/to/firmware \\
        --board ASY011 --micro STM32H750 \\
        --scope categoria --categoria caffe --cliente acme \\
        --costruttore acme-srl

    # Full rebuild (wipe first):
    python scripts/build_index.py --repo /path/to/firmware \\
        --board ASY011 --micro STM32H750 --reset

board and micro are REQUIRED: every chunk must be attributable to exactly one
board/MCU, because retrieval refuses to cross board boundaries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/build_index.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import load_config  # noqa: E402
from rag.embeddings import get_embedder  # noqa: E402
from rag.indexer import index_repo  # noqa: E402
from rag.store import ChromaStore  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RAG index from a repo.")
    parser.add_argument("--repo", required=True, help="Path to the C/H source tree.")

    # Mandatory knowledge-model dimensions.
    parser.add_argument("--board", required=True, help="Board id, e.g. ASY011.")
    parser.add_argument("--micro", required=True, help="MCU, e.g. STM32H750.")

    # Optional knowledge-model dimensions.
    parser.add_argument(
        "--scope",
        choices=["comune", "categoria", "cliente"],
        help="Knowledge scope of this repo.",
    )
    parser.add_argument("--categoria", help="Product family, e.g. caffe/forno/tosaerba.")
    parser.add_argument("--cliente", help="Customer id.")
    parser.add_argument("--costruttore", help="Manufacturer id.")

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the collection before indexing (full rebuild).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config()

    base_metadata = {
        "scope": args.scope,
        "categoria": args.categoria,
        "cliente": args.cliente,
        "costruttore": args.costruttore,
        "board": args.board,
        "micro": args.micro,
    }

    store = ChromaStore(index_path=cfg.index_path)
    embedder = get_embedder(cfg)

    print(f"[build_index] repo={args.repo}")
    print(f"[build_index] index_path={cfg.index_path}")
    print(f"[build_index] embedder={embedder.signature}")
    print(f"[build_index] base_metadata={base_metadata}")
    if args.reset:
        print("[build_index] reset=True (collection will be wiped first)")

    summary = index_repo(
        repo_path=args.repo,
        store=store,
        embedder=embedder,
        base_metadata=base_metadata,
        reset=args.reset,
    )

    print(
        f"[build_index] done: indexed {summary['files']} files / "
        f"{summary['chunks']} chunks. Total in store: {store.count()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
