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

import fnmatch
import hashlib
import logging
import os
import re
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

# --- "image-as-C / generated asset" detection (content-based) ----------------
# Firmware repos mix code and data in the same folder (e.g. menu_template/ holds
# both auto_boost_on.c [code] and img6.c [a 2.9 MB pixel blob]), so folder-name
# exclusion is not enough — detection must look at the file CONTENT.
#
# These thresholds for the byte-array heuristic are deliberately CONSERVATIVE so
# a legitimate firmware lookup table (e.g. const uint16_t sin_table[256]) is NOT
# misclassified as an asset. The primary, robust signal is the longest-line
# check (RAG_MAX_DATA_LINE_CHARS); the byte-array check below is a backup for
# assets whose lines happen to be short, and it requires ALL of: a single
# initializer covering most of the file, a large absolute size, many entries,
# and an overwhelmingly numeric/hex body.
_DATA_ARRAY_MIN_BODY_CHARS = 20000   # absolute floor — real LUTs are far smaller
_DATA_ARRAY_DOMINANCE = 0.6          # body must cover most of the file
_DATA_ARRAY_MIN_TOKENS = 256         # many comma-separated entries
_DATA_ARRAY_NUMERIC_RATIO = 0.85     # body is overwhelmingly numeric/hex literals

_ARRAY_INIT_RE = re.compile(r"\]\s*=\s*\{")
_NUM_TOKEN_RE = re.compile(r"^(?:0[xX][0-9a-fA-F]+|[+-]?\d+)[uUlL]*$")


def _match_brace_body(source: str, open_idx: int) -> Optional[str]:
    """Return the text between source[open_idx]=='{' and its matching '}'."""
    depth = 0
    for i in range(open_idx, len(source)):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[open_idx + 1 : i]
    return None  # unbalanced braces


def looks_like_data(source: str, max_data_line_chars: int) -> Optional[str]:
    """Return a human-readable reason if ``source`` is generated asset/data, else None.

    Content-based, so it works regardless of folder name. Two heuristics:

    (1) LONGEST LINE — if any line exceeds ``max_data_line_chars`` the file is a
        generated blob (image-as-C pixel arrays have 6000+ char lines). This is
        the primary, reliable signal and the cheapest to compute.

    (2) BYTE-ARRAY DOMINANCE — a single ``... [] = { ... }`` initializer that is
        large in absolute terms AND covers most of the file AND is overwhelmingly
        numeric/hex. Conservative thresholds avoid flagging real lookup tables.
    """
    # (1) longest-line heuristic.
    longest = 0
    for line in source.splitlines():
        if len(line) > longest:
            longest = len(line)
    if longest > max_data_line_chars:
        return (
            f"longest line {longest} chars > {max_data_line_chars}, "
            f"looks like generated asset data"
        )

    # (2) single large byte-array initializer dominating the file.
    total = len(source)
    if total >= _DATA_ARRAY_MIN_BODY_CHARS:
        best_body = ""
        for m in _ARRAY_INIT_RE.finditer(source):
            body = _match_brace_body(source, m.end() - 1)
            if body and len(body) > len(best_body):
                best_body = body
        if (
            best_body
            and len(best_body) >= _DATA_ARRAY_MIN_BODY_CHARS
            and len(best_body) >= _DATA_ARRAY_DOMINANCE * total
        ):
            tokens = [t.strip() for t in best_body.split(",") if t.strip()]
            if len(tokens) >= _DATA_ARRAY_MIN_TOKENS:
                numeric = sum(1 for t in tokens if _NUM_TOKEN_RE.match(t))
                if numeric / len(tokens) >= _DATA_ARRAY_NUMERIC_RATIO:
                    pct = round(100 * len(best_body) / total)
                    return (
                        f"a single byte-array initializer covers ~{pct}% of the "
                        f"file ({len(tokens)} numeric entries), looks like "
                        f"generated asset data"
                    )
    return None


def _matches_globs(file_path: str, repo_path: str, patterns: Optional[List[str]]) -> Optional[str]:
    """Return the first matching pattern (truthy) or None.

    A pattern matches when it equals a path segment (so ``lvgl/`` or ``lvgl``
    excludes anything under an ``lvgl`` directory), or fnmatch-matches the file's
    basename or its path relative to the repo (so ``img*.c`` or ``images/*`` work).
    """
    if not patterns:
        return None
    rel = os.path.relpath(file_path, repo_path)
    base = os.path.basename(file_path)
    segments = rel.split(os.sep)
    for pat in patterns:
        bare = pat.rstrip("/\\")
        if bare in segments:
            return pat
        if fnmatch.fnmatch(base, pat) or fnmatch.fnmatch(rel, pat):
            return pat
        if fnmatch.fnmatch(rel, bare + "/*"):
            return pat
    return None


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
    max_data_line_chars: Optional[int] = None,
    exclude: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    include_data: bool = False,
) -> Dict[str, object]:
    """Walk ``repo_path`` and index every .c/.h file.

    Before embedding, each file is filtered:

    * ``exclude`` globs (e.g. ``lvgl/``, ``Drivers/``, ``images/``) — explicit
      manual exclusion; matched files are skipped (``skipped_excluded``).
    * CONTENT-BASED data detection (:func:`looks_like_data`) — image-as-C blobs
      and similar generated assets are skipped BEFORE embedding so they can't
      time out the embedder or peg the CPU (``skipped_data``).
    * ``include`` globs / ``include_data=True`` — FORCE-index a file even if the
      data heuristic flags it (override for a legitimate file like a lookup
      table the heuristic wrongly caught).

    Resilient: if a single file fails to embed/store (e.g. an embedding HTTP
    timeout), it is logged and skipped (``skipped_error``) and the walk
    continues — a long build is never lost to one bad file.

    Returns ``{"files", "chunks", "skipped_data", "skipped_error",
    "skipped_excluded"}``. Set ``reset=True`` for a full rebuild.
    ``max_chunk_chars`` / ``max_data_line_chars`` default to config.
    """
    base_metadata = dict(base_metadata or {})
    if max_chunk_chars is None or max_data_line_chars is None:
        cfg = load_config()
        if max_chunk_chars is None:
            max_chunk_chars = cfg.max_chunk_chars
        if max_data_line_chars is None:
            max_data_line_chars = cfg.max_data_line_chars
    if reset:
        store.reset()

    files_indexed = 0
    chunks_indexed = 0
    skipped_data: List[Dict[str, str]] = []
    skipped_error: List[Dict[str, str]] = []
    skipped_excluded: List[Dict[str, str]] = []

    for root, _dirs, files in os.walk(repo_path):
        for name in sorted(files):
            if not name.endswith(_C_EXTENSIONS):
                continue
            file_path = os.path.join(root, name)

            # 1) Explicit manual exclusion wins.
            excl_pattern = _matches_globs(file_path, repo_path, exclude)
            if excl_pattern:
                logger.info("skipped %s: excluded by pattern %r", name, excl_pattern)
                skipped_excluded.append({"file": file_path, "pattern": excl_pattern})
                continue

            # 2) Force-index overrides the data heuristic (global flag or glob).
            forced = include_data or bool(_matches_globs(file_path, repo_path, include))

            # 3) Content-based data detection (unless forced).
            if not forced:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                        source = fh.read()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    logger.warning("Skipping %s — could not read: %s", file_path, error)
                    skipped_error.append({"file": file_path, "error": error})
                    continue
                reason = looks_like_data(source, max_data_line_chars)
                if reason:
                    logger.info("skipped %s: %s", name, reason)
                    skipped_data.append({"file": file_path, "reason": reason})
                    continue

            # 4) Index (resilient to per-file embedding/store failures).
            try:
                count = index_file(
                    file_path, store, embedder, base_metadata, max_chunk_chars
                )
            except Exception as exc:  # don't abort the whole build on one file
                error = f"{type(exc).__name__}: {exc}"
                logger.warning("Skipping %s — embedding/index failed: %s", file_path, error)
                skipped_error.append({"file": file_path, "error": error})
                continue
            if count:
                files_indexed += 1
                chunks_indexed += count

    return {
        "files": files_indexed,
        "chunks": chunks_indexed,
        "skipped_data": skipped_data,
        "skipped_error": skipped_error,
        "skipped_excluded": skipped_excluded,
    }
