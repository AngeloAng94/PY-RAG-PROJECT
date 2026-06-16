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

FILE-LEVEL CONTEXT (kind ``file_context``)
Everything at the top level that is NOT one of the semantic units above is
collected into a SINGLE synthetic ``file_context`` chunk per file:

    * ``#include`` directives,
    * global variable declarations (e.g. ``static lv_obj_t *g_label;``),
    * function prototypes, and
    * global assignments.

Global state variables are exactly the context the LLM needs to patch a file
correctly, so we never drop them. (One grouped chunk keeps the index tidy; flip
``GROUP_FILE_CONTEXT`` to emit them individually if you prefer.)

AI MARKER BLOCKS (kind ``ai_block``)
We also extract the delimited blocks

    // [AI_START_<tag>]
    ... agent-written code ...
    // [AI_END_<tag>]

as dedicated chunks. Those regions are *exactly* where the agent patches code,
so they are the highest-value retrieval targets. Extraction is a line-by-line
START->END scan (not a regex) so repeated tags each yield their own chunk and a
missing ``AI_END`` is dropped gracefully instead of swallowing the rest of the
file.

DOC COMMENTS
A leading doc comment (``/** ... */`` or a ``//`` block) sitting immediately
above a declaration is attached to that declaration's chunk.

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

# Specifier node types that, when given a body, are first-class semantic chunks
# even when wrapped in a `declaration` (e.g. bare `struct point_s {...};`).
_BODIED_SPECIFIERS = ("struct_specifier", "enum_specifier")

# Top-level node types that carry file-level context (not semantic units, but
# exactly the surrounding state/imports the LLM needs).
_FILE_CONTEXT_TYPES = ("preproc_include", "expression_statement")

# When True, all file-context nodes are merged into ONE `file_context` chunk per
# file. When False, each is emitted as its own `file_context` chunk.
GROUP_FILE_CONTEXT = True

# Line-level AI marker matchers. A line-by-line scan (below) uses these so that
# repeated tags and unterminated blocks are handled explicitly.
_AI_START_RE = re.compile(r"//\s*\[AI_START_(?P<tag>[A-Za-z0-9_]+)\]")
_AI_END_RE = re.compile(r"//\s*\[AI_END_(?P<tag>[A-Za-z0-9_]+)\]")


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
    """Extract every // [AI_START_*] ... // [AI_END_*] region as an ai_block.

    Implemented as an explicit line-by-line scan (not a regex) so that:

      * REPEATED TAGS work — ``START_DRAW ... END_DRAW`` twice yields two
        separate chunks (a backreference regex would mis-pair them);
      * NESTING is tolerated — an END closes the nearest still-open START with
        the same tag (LIFO);
      * a MISSING ``AI_END`` is handled gracefully — an unterminated START is
        discarded at EOF instead of swallowing the rest of the file;
      * an ``AI_END`` with no matching open START is simply ignored.
    """
    lines = source.splitlines()
    open_stack: List[tuple] = []  # (tag, start_index) for still-open STARTs
    chunks: List[Chunk] = []

    for idx, line in enumerate(lines):
        start_match = _AI_START_RE.search(line)
        if start_match:
            open_stack.append((start_match.group("tag"), idx))
            continue

        end_match = _AI_END_RE.search(line)
        if end_match:
            tag = end_match.group("tag")
            # Close the nearest still-open START with the same tag (LIFO).
            for j in range(len(open_stack) - 1, -1, -1):
                if open_stack[j][0] == tag:
                    _, start_idx = open_stack.pop(j)
                    text = "\n".join(lines[start_idx : idx + 1])
                    chunks.append(
                        Chunk(
                            text=text,
                            kind="ai_block",
                            symbol=tag,
                            start_line=start_idx + 1,
                            end_line=idx + 1,
                            metadata=dict(base_metadata or {}),
                        )
                    )
                    break
            # END without a matching START -> ignored on purpose.

    # Any STARTs left open at EOF are unterminated; drop them so we never
    # swallow the rest of the file into a runaway chunk.
    return chunks


def _bodied_specifier(decl_node):
    """Return a struct/enum specifier WITH a body inside a `declaration`, else None.

    Distinguishes ``struct point_s {...};`` (a real type definition -> semantic
    chunk) from ``struct point_s p;`` (a variable -> file-level context).
    """
    for child in decl_node.children:
        if child.type in _BODIED_SPECIFIERS:
            for grandchild in child.children:
                if grandchild.type in ("field_declaration_list", "enumerator_list"):
                    return child
    return None


def _leading_comment_nodes(pending_comments: list, node) -> list:
    """Pick the contiguous comment block sitting directly above ``node``.

    ``pending_comments`` is the run of comment siblings seen just before
    ``node``. We attach them only when the last one is on the line immediately
    above ``node`` (no blank line in between) — that is what marks a doc comment
    as belonging to the declaration.
    """
    if not pending_comments:
        return []
    last = pending_comments[-1]
    if node.start_point[0] == last.end_point[0] + 1:
        return pending_comments
    return []


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
    file_context_nodes: List = []  # globals / includes / prototypes / assignments
    pending_comments: List = []  # consecutive comment siblings seen so far

    def _make_semantic_chunk(text_node, kind: str, symbol_node, leading: list) -> None:
        """Build a declaration chunk, optionally prefixed with its doc comment."""
        if leading:
            start_byte = leading[0].start_byte
            start_line = leading[0].start_point[0] + 1
        else:
            start_byte = text_node.start_byte
            start_line = text_node.start_point[0] + 1
        text = source_bytes[start_byte : text_node.end_byte].decode(
            "utf-8", errors="replace"
        )
        chunks.append(
            Chunk(
                text=text,
                kind=kind,
                symbol=_extract_symbol(source_bytes, symbol_node),
                start_line=start_line,
                end_line=text_node.end_point[0] + 1,
                metadata=dict(base_metadata or {}),
            )
        )

    for node in root.children:
        # Track runs of contiguous comment siblings (candidate doc comments).
        if node.type == "comment":
            if (
                pending_comments
                and node.start_point[0] > pending_comments[-1].end_point[0] + 1
            ):
                pending_comments = [node]  # blank line broke the run
            else:
                pending_comments.append(node)
            continue

        leading = _leading_comment_nodes(pending_comments, node)

        if node.type in _NODE_KIND:
            # Direct semantic unit: function/typedef/define (and bare specifiers).
            _make_semantic_chunk(node, _NODE_KIND[node.type], node, leading)
        elif node.type == "declaration":
            spec = _bodied_specifier(node)
            if spec is not None:
                # Bare `struct/enum Foo {...};` — keep the whole declaration text
                # but take kind/symbol from the specifier.
                _make_semantic_chunk(node, _NODE_KIND[spec.type], spec, leading)
            else:
                # Global variable declaration / function prototype -> context.
                file_context_nodes.append(node)
        elif node.type in _FILE_CONTEXT_TYPES:
            # #include directives and global assignments -> context.
            file_context_nodes.append(node)
        # else: preprocessor conditionals, errors, etc. are skipped.

        pending_comments = []  # any non-comment node ends the doc-comment run

    # File-level context: includes + globals + prototypes + global assignments.
    chunks.extend(_build_file_context_chunks(source_bytes, file_context_nodes, base_metadata))

    # AI marker blocks are independent of the syntax tree (they are comments).
    chunks.extend(_extract_ai_blocks(source, base_metadata))

    # Stable ordering by position helps reproducibility and debugging.
    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks


def _build_file_context_chunks(
    source_bytes: bytes,
    nodes: List,
    base_metadata: Optional[Dict],
) -> List[Chunk]:
    """Turn collected top-level context nodes into file_context chunk(s).

    With ``GROUP_FILE_CONTEXT`` (default) all nodes merge into one chunk per
    file; otherwise each node becomes its own chunk. Either way the LLM gets the
    global state it needs to patch the file correctly.
    """
    if not nodes:
        return []

    def _one(node) -> Chunk:
        return Chunk(
            text=_node_text(source_bytes, node),
            kind="file_context",
            symbol=None,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            metadata=dict(base_metadata or {}),
        )

    if not GROUP_FILE_CONTEXT:
        return [_one(n) for n in nodes]

    texts = [_node_text(source_bytes, n) for n in nodes]
    return [
        Chunk(
            text="\n".join(texts),
            kind="file_context",
            symbol=None,
            start_line=nodes[0].start_point[0] + 1,
            end_line=nodes[-1].end_point[0] + 1,
            metadata=dict(base_metadata or {}),
        )
    ]
