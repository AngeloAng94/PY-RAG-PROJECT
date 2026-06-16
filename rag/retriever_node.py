"""
rag/retriever_node.py — DROP-IN replacement for the agent's ``retrieve`` node.

CONTRACT (unchanged — do not break):

    def retrieve(state: AgentState) -> AgentState

It receives the shared ``AgentState`` and returns it with ``full_context``
populated, exactly like the old node. We ONLY change the internals: instead of
concatenating whole files uncapped, we:

    (a) read the TARGET file whole — it is the file being edited, so the agent
        needs its current content verbatim (this part is intentionally NOT
        retrieval; an edited file must be seen in full);
    (b) query the RAG store with the SESSION metadata filter (board/micro
        mandatory) to pull semantically relevant EXAMPLES from other files;
    (c) assemble ``full_context`` while enforcing a CHAR/TOKEN BUDGET
        (``RAG_MAX_EXAMPLE_CHARS``) — no more uncapped loading;
    (d) add the ADDITIVE fields ``retrieved_chunks`` and ``retrieval_debug``
        without removing or renaming any existing state field.

Nothing in ``graph.py`` or the node signature changes.

HOW THE SESSION FILTER IS RESOLVED
----------------------------------
The board/micro/scope/categoria/cliente dimensions describe the current build
session. In the real system the classifier populates these. To keep this node
drop-in WITHOUT touching the classifier, we read them from ``state`` if present
and otherwise fall back to env vars. Wiring to the real classifier is out of
scope (see TODO).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from .config import load_config
from .embeddings import get_embedder
from .query import retrieve_relevant
from .store import ChromaStore

# ---------------------------------------------------------------------------
# AgentState: in the real system this is imported from the agent package. We
# import it if available, otherwise fall back to a structural TypedDict so this
# module is self-contained and testable in isolation. The real import wins.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on the host project layout
    from agent.state import AgentState  # type: ignore
except Exception:  # pragma: no cover - standalone/testing fallback
    from typing import TypedDict

    class AgentState(TypedDict, total=False):
        user_request: str
        target_view: str
        target_file: str
        target_headers: List[str]
        reference_file: str
        source_content: str
        backup_content: str
        reference_content: str
        full_context: str
        modification_plan: Any
        code_snippets: Any
        compile_result: Any
        iterations: int
        max_iterations: int
        notes: Any
        # Additive fields introduced by this node (kept optional):
        retrieved_chunks: List[Dict[str, Any]]
        retrieval_debug: Dict[str, Any]


def _session_dimension(state: AgentState, key: str) -> str:
    """Read a retrieval dimension from state, falling back to env.

    # TODO (human): wire to the real classifier output. For now we accept the
    # value either directly on the state (if the classifier already sets it) or
    # from an env var (useful for manual/offline runs and tests). board/micro
    # MUST resolve to a real value or retrieval will (correctly) refuse to run.
    """
    value = state.get(key)  # type: ignore[arg-type]
    if value:
        return str(value)
    return os.environ.get(f"RAG_SESSION_{key.upper()}", "").strip()


def _read_target_file(state: AgentState) -> str:
    """Return the target file content, preferring already-loaded state.

    The agent often pre-loads the file into ``source_content``. We reuse that
    if present (avoids a redundant disk read) and otherwise read from disk.
    """
    if state.get("source_content"):
        return state["source_content"]  # type: ignore[return-value]
    target_file = state.get("target_file")
    if target_file and os.path.exists(target_file):
        with open(target_file, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return ""


def _assemble_examples(chunks: List[Dict[str, Any]], budget_chars: int) -> str:
    """Concatenate retrieved example chunks until the char budget is exhausted.

    We add whole chunks only (never split a semantic unit). Each example is
    annotated with provenance so the downstream LLM knows where it came from.
    Returns the assembled examples block (may be empty).
    """
    parts: List[str] = []
    used = 0
    for chunk in chunks:
        meta = chunk.get("metadata", {}) or {}
        header = (
            f"// --- example: {meta.get('kind', '?')} "
            f"{meta.get('symbol', '')} "
            f"from {meta.get('source_path', '?')} "
            f"(board={meta.get('board', '?')}, micro={meta.get('micro', '?')}) ---\n"
        )
        body = chunk.get("document", "") or ""
        block = header + body + "\n"
        if used + len(block) > budget_chars:
            # Stop at the budget; do NOT truncate inside a chunk.
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def retrieve(state: AgentState) -> AgentState:
    """Drop-in ``retrieve`` node backed by the RAG engine.

    Keeps the original contract: returns ``state`` with ``full_context`` set.
    Adds ``retrieved_chunks`` and ``retrieval_debug`` additively.
    """
    cfg = load_config()

    # (a) The file under edit — always included in full, verbatim.
    target_content = _read_target_file(state)

    # Resolve the mandatory + optional session dimensions.
    board = _session_dimension(state, "board")
    micro = _session_dimension(state, "micro")
    scope = _session_dimension(state, "scope") or None
    categoria = _session_dimension(state, "categoria") or None
    cliente = _session_dimension(state, "cliente") or None

    # The semantic query: prefer an explicit modification intent, else the
    # raw user request, else the target view name.
    query_text = (
        state.get("user_request")
        or state.get("target_view")
        or ""
    )

    retrieved: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {
        "board": board,
        "micro": micro,
        "scope": scope,
        "categoria": categoria,
        "cliente": cliente,
        "k": cfg.top_k,
        "budget_chars": cfg.max_example_chars,
        "index_path": cfg.index_path,
        "embedder": None,
        "query": query_text,
        "status": "ok",
        "error": None,
    }

    # (b) Query the RAG store — but only if we have a valid board/micro scope.
    # Without them we must NOT cross board boundaries, so we skip retrieval and
    # degrade gracefully to "target file only" rather than returning wrong code.
    if board and micro and query_text:
        try:
            store = ChromaStore(index_path=cfg.index_path)
            embedder = get_embedder(cfg)
            debug["embedder"] = embedder.signature
            retrieved = retrieve_relevant(
                store=store,
                embedder=embedder,
                query=query_text,
                scope=scope,
                categoria=categoria,
                cliente=cliente,
                board=board,
                micro=micro,
                k=cfg.top_k,
            )
        except Exception as exc:  # never let retrieval crash the graph
            debug["status"] = "error"
            debug["error"] = f"{type(exc).__name__}: {exc}"
    else:
        debug["status"] = "skipped"
        debug["error"] = (
            "missing board/micro/query — retrieval skipped to avoid a "
            "cross-board search; using target file only."
        )

    # (c) Assemble full_context with the char budget enforced on EXAMPLES only.
    # The target file is the editing surface and is always present in full.
    examples_block = _assemble_examples(retrieved, cfg.max_example_chars)
    debug["examples_chars"] = len(examples_block)
    debug["retrieved_count"] = len(retrieved)

    sections: List[str] = []
    if target_content:
        sections.append("// ===== TARGET FILE (being edited) =====\n" + target_content)
    if examples_block:
        sections.append("// ===== RELEVANT EXAMPLES (retrieved) =====\n" + examples_block)
    full_context = "\n\n".join(sections)

    # (d) Write back: existing contract field + additive debug fields.
    state["full_context"] = full_context
    state["retrieved_chunks"] = retrieved
    state["retrieval_debug"] = debug
    return state
