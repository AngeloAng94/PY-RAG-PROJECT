"""
rag/chunker.py — Semantic chunking of C source using tree-sitter.

WHY SEMANTIC CHUNKING
---------------------
Fixed-length / sliding-window chunking shreds C code mid-statement and ruins
both embeddings and the snippets shown to the LLM. We instead split on the
syntax tree so every chunk is a self-contained semantic unit.

Chunk kinds we extract (top-level declarations):

    function_definition   -> kind "function"
    struct_specifier      -> kind "struct"
    enum_specifier        -> kind "enum"
    type_definition       -> kind "typedef"
    preproc_def           -> kind "define"
    preproc_function_def  -> kind "define"   (function-like macro)

PLUS — and this is specific to THIS agent — we extract the delimited blocks

    // [AI_START_<tag>]
    ... agent-written code ...
    // [AI_END_<tag>]

as dedicated chunks of kind ``ai_block``. Those regions are *exactly* where the
agent patches code, so they are the highest-value retrieval targets: "show me
how the AI block for <feature> looked in a similar file".

Each :class:`Chunk` carries: ``text``, ``kind``, ``symbol`` (best-effort
function/struct/typedef name), ``start_line``, ``end_line`` and a ``metadata``
dict (filled in later by the indexer with board/micro/layer/etc).

NEVER split by fixed length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import tree_sitter_c
from tree_sitter import Language, Parser

# --- tree-sitter bootstrap (version-tolerant) ------------------------------
# tree-sitter's Python API changed across 0.21 -> 0.25. We try the modern API
# first and fall back to the legacy signatures so this works on either.
try:
    C_LANGUAGE = Language(tree_sitter_c.language())  # tree-sitter >= 0.22
except TypeError:  # pragma: no cover - legacy binding
    C_LANGUAGE = Language(tree_sitter_c.language(), "c")  # tree-sitter < 0.22


def _new_parser() -> Parser:
    try:
        return Parser(C_LANGUAGE)  # tree-sitter >= 0.25
    except TypeError:  # pragma: no cover - legacy binding
        p = Parser()
        try:
            p.language = C_LANGUAGE  # tree-sitter ~0.22-0.24
        except AttributeError:
            p.set_language(C_LANGUAGE)  # tree-sitter < 0.22
        return p


# Map tree-sitter node types -> our coarse "kind" label.
_NODE_KIND = {
    "function_definition": "function",
    "struct_specifier": "struct",
    "enum_specifier": "enum",
    "type_definition": "typedef",
    "preproc_def": "define",
    "preproc_function_def": "define",
}

# AI marker blocks. We match a START line, then everything up to the matching
# END line. The tag (e.g. "DRAW", "EVENTS") is captured for the chunk symbol.
_AI_BLOCK_RE = re.compile(
    r"//\s*\[AI_START_(?P<tag>[A-Za-z0-9_]+)\].*?//\s*\[AI_END_(?P=tag)\]",
    re.DOTALL,
)


@dataclass
class Chunk:
    """A single semantic unit extracted from a C source file."""

    text: str
    kind: str
    symbol: Optional[str]
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    metadata: Dict[str, object] = field(default_factory=dict)


def _node_text(source_bytes: bytes, node) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode(
        "utf-8", errors="replace"
    )


def _extract_symbol(source_bytes: bytes, node) -> Optional[str]:
    """Best-effort name extraction for a declaration node.

    # TODO (human): C declarators are deeply nested (pointers, arrays,
    # function pointers, typedef chains). This heuristic covers the common
    # cases seen in firmware UI code; tune it against the real repositories if
    # symbol-based filtering/ranking is added later.
    """
    kind = _NODE_KIND.get(node.type)

    if node.type == "function_definition":
        decl = node.child_by_field_name("declarator")
        return _declarator_name(source_bytes, decl)

    if node.type in ("struct_specifier", "enum_specifier"):
        name = node.child_by_field_name("name")
        return _node_text(source_bytes, name) if name else None

    if node.type == "type_definition":
        # The typedef's introduced name is the last declarator before ';'.
        names = [
            _node_text(source_bytes, c)
            for c in node.children
            if c.type in ("type_identifier", "identifier")
        ]
        return names[-1] if names else None

    if kind == "define":
        name = node.child_by_field_name("name")
        return _node_text(source_bytes, name) if name else None

    return None


def _declarator_name(source_bytes: bytes, node) -> Optional[str]:
    """Walk down a (possibly nested) declarator to the function identifier."""
    if node is None:
        return None
    if node.type in ("identifier", "field_identifier"):
        return _node_text(source_bytes, node)
    inner = node.child_by_field_name("declarator")
    if inner is not None:
        return _declarator_name(source_bytes, inner)
    # Fall back: first identifier descendant.
    for child in node.children:
        name = _declarator_name(source_bytes, child)
        if name:
            return name
    return None


def _extract_ai_blocks(source: str, base_metadata: Optional[Dict]) -> List[Chunk]:
    """Find every // [AI_START_*] ... // [AI_END_*] region as an ai_block chunk."""
    chunks: List[Chunk] = []
    for m in _AI_BLOCK_RE.finditer(source):
        start_line = source.count("\n", 0, m.start()) + 1
        end_line = source.count("\n", 0, m.end()) + 1
        chunks.append(
            Chunk(
                text=m.group(0),
                kind="ai_block",
                symbol=m.group("tag"),
                start_line=start_line,
                end_line=end_line,
                metadata=dict(base_metadata or {}),
            )
        )
    return chunks


def chunk_c_source(
    source: str,
    base_metadata: Optional[Dict] = None,
) -> List[Chunk]:
    """Parse ``source`` and return a list of semantic :class:`Chunk` objects.

    ``base_metadata`` is shallow-copied into every chunk's ``metadata`` so the
    caller (indexer) can attach board/micro/layer/etc. up front.

    The result contains BOTH top-level declarations and ai_block regions. The
    two can overlap (an AI block may sit inside a function), which is
    intentional — we want the AI block retrievable on its own.

    # TODO (human): chunking strategy tuning. Decisions to calibrate on the
    # real codebase: whether to include leading doc comments with each chunk,
    # how to handle very large functions, and whether to also index file-level
    # context (includes / global defines) as a synthetic chunk.
    """
    parser = _new_parser()
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    chunks: List[Chunk] = []

    # Only walk TOP-LEVEL children of the translation unit. Nested declarations
    # (e.g. a struct inside a function) stay part of their enclosing chunk.
    for node in root.children:
        kind = _NODE_KIND.get(node.type)
        if kind is None:
            # `typedef struct {...} Foo;` parses as a `declaration` wrapping a
            # struct_specifier. Reach in so we still capture the struct.
            if node.type == "declaration":
                for child in node.children:
                    if child.type in _NODE_KIND:
                        node = child
                        kind = _NODE_KIND[child.type]
                        break
            if kind is None:
                continue

        chunks.append(
            Chunk(
                text=_node_text(source_bytes, node),
                kind=kind,
                symbol=_extract_symbol(source_bytes, node),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                metadata=dict(base_metadata or {}),
            )
        )

    # AI marker blocks are independent of the syntax tree (they are comments).
    chunks.extend(_extract_ai_blocks(source, base_metadata))

    # Stable ordering by position helps reproducibility and debugging.
    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks
