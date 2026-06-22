"""
rag/indexer.py — OFFLINE ingest pipeline.

This module is NOT a graph node and is NEVER called from the running agent. It
is the batch process that turns a directory of C/H sources into vectors in the
ChromaDB store. Keeping ingest offline (here) separate from retrieval online
(``rag.query`` / ``rag.retriever_node``) is a hard design constraint.

Flow per file:

    read -> chunk (rag.chunker) -> derive metadata -> embed -> store

Metadata derivation from the filesystem path is an INITIAL HEURISTIC. Real
repositories will need their own mapping; see the ``# TODO (human):`` notes.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Dict, List, Optional

from .chunker import Chunk, chunk_c_source
from .config import load_config
from .constants import ABSENT, FALLBACK_DIMS
from .embeddings import Embedder
from .store import ChromaStore

logger = logging.getLogger(__name__)

# File extensions we treat as C sources/headers.
_C_EXTENSIONS = (".c", ".h")

# Canonical firmware layers used by the knowledge model.
_KNOWN_LAYERS = ("hal", "bsp", "rtos", "middleware", "app", "ui")


def infer_layer(path: str) -> Optional[str]:
    """Guess the firmware ``layer`` from a file path.

    INITIAL HEURISTIC: we look for a known layer name as a path segment, with a
    couple of common aliases. This is deliberately simple and meant to be
    replaced by a repo-specific mapping.

    # TODO (human): metadata-from-path derivation. Calibrate against the real
    # repository layout (some teams nest UI under app/, some keep BSP per
    # board, etc.). When in doubt, prefer returning None over guessing wrong —
    # a wrong layer silently biases retrieval.
    """
    parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
    aliases = {
        "drivers": "hal",
        "hal": "hal",
        "bsp": "bsp",
        "board": "bsp",
        "rtos": "rtos",
        "freertos": "rtos",
        "middleware": "middleware",
        "middlewares": "middleware",
        "app": "app",
        "application": "app",
        "ui": "ui",
        "gui": "ui",
        "lvgl": "ui",
    }
    for part in parts:
        if part in aliases:
            return aliases[part]
    return None


def _chunk_id(file_path: str, chunk: Chunk) -> str:
    """Deterministic, stable id so re-indexing upserts instead of duplicating."""
    raw = f"{file_path}:{chunk.kind}:{chunk.symbol}:{chunk.start_line}-{chunk.end_line}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _derive_metadata(
    file_path: str,
    chunk: Chunk,
    base_metadata: Dict,
) -> Dict:
    """Merge caller-supplied base metadata with per-chunk/per-file derivations.

    Precedence: explicit ``base_metadata`` (from the CLI / session) wins over
    anything inferred from the path, because the operator knows the truth.
    """
    meta: Dict[str, object] = {}

    # Path-inferred fields (lowest precedence).
    inferred_layer = infer_layer(file_path)
    if inferred_layer:
        meta["layer"] = inferred_layer

    # Caller-provided base metadata (highest precedence: board, micro, etc.).
    for key, value in base_metadata.items():
        if value is not None:
            meta[key] = value

    # Layered fall-through dimensions (categoria, cliente): when not provided,
    # store the ABSENT sentinel instead of leaving the key unset. This makes
    # "shared" chunks (comune has no categoria; comune/categoria have no
    # cliente) precisely matchable so they survive a narrower query filter.
    # See rag/constants.py for why a sentinel is required (Chroma can't match a
    # missing key without also matching every other value).
    for dim in FALLBACK_DIMS:
        if not meta.get(dim):
            meta[dim] = ABSENT

    # Per-chunk intrinsic fields (always recorded).
    meta["kind"] = chunk.kind
    if chunk.symbol:
        meta["symbol"] = chunk.symbol
    meta["source_path"] = file_path
    meta["start_line"] = chunk.start_line
    meta["end_line"] = chunk.end_line

    # Carry over chunk-level markers set by the chunker (e.g. the split markers
    # part_index / part_total / chunk_split) without overriding anything above.
    for key, value in chunk.metadata.items():
        meta.setdefault(key, value)

    # Record which embedder produced the vector (mismatch detection).
    meta.setdefault("embedder", "")

    return meta


def index_file(
    file_path: str,
    store: ChromaStore,
    embedder: Embedder,
    base_metadata: Optional[Dict] = None,
    max_chunk_chars: Optional[int] = None,
) -> int:
    """Chunk, embed and store a single C/H file. Returns the chunk count.

    ``max_chunk_chars`` caps a single chunk's length (oversized chunks are split
    at line boundaries by the chunker). Pass ``RAG_MAX_CHUNK_CHARS`` here.
    """
    base_metadata = dict(base_metadata or {})
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        source = fh.read()

    chunks = chunk_c_source(source, max_chunk_chars=max_chunk_chars)
    if not chunks:
        return 0

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict] = []
    for chunk in chunks:
        meta = _derive_metadata(file_path, chunk, base_metadata)
        meta["embedder"] = embedder.signature
        ids.append(_chunk_id(file_path, chunk))
        documents.append(chunk.text)
        metadatas.append(meta)

    # Batch-embed all chunk texts with the SAME embedder used at query time.
    embeddings = embedder.embed_documents(documents)
    store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(chunks)


def index_repo(
    repo_path: str,
    store: ChromaStore,
    embedder: Embedder,
    base_metadata: Optional[Dict] = None,
    reset: bool = False,
    max_chunk_chars: Optional[int] = None,
) -> Dict[str, object]:
    """Walk ``repo_path`` and index every .c/.h file.

    Resilient: if a single file fails to embed/store (e.g. an embedding HTTP
    timeout or any other error), it is logged and SKIPPED, and the walk
    continues — a long build is never lost to one bad file/chunk. The returned
    summary lists the skipped files.

    Returns ``{"files": N, "chunks": M, "skipped": [{"file":..., "error":...}]}``.
    Set ``reset=True`` to wipe the collection first (full rebuild).
    ``max_chunk_chars`` defaults to ``RAG_MAX_CHUNK_CHARS`` from config.
    """
    base_metadata = dict(base_metadata or {})
    if max_chunk_chars is None:
        max_chunk_chars = load_config().max_chunk_chars
    if reset:
        store.reset()

    files_indexed = 0
    chunks_indexed = 0
    skipped: List[Dict[str, str]] = []
    for root, _dirs, files in os.walk(repo_path):
        for name in sorted(files):
            if not name.endswith(_C_EXTENSIONS):
                continue
            file_path = os.path.join(root, name)
            try:
                count = index_file(
                    file_path, store, embedder, base_metadata, max_chunk_chars
                )
            except Exception as exc:  # don't abort the whole build on one file
                error = f"{type(exc).__name__}: {exc}"
                logger.warning("Skipping %s — embedding/index failed: %s", file_path, error)
                skipped.append({"file": file_path, "error": error})
                continue
            if count:
                files_indexed += 1
                chunks_indexed += count

    return {"files": files_indexed, "chunks": chunks_indexed, "skipped": skipped}
