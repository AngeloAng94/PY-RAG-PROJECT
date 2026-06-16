# RAG Infrastructure Scaffolding — PRD

## Problem
Replace ONLY the internals of an existing LangGraph agent's `retrieve` node
(graph: classify→retrieve→plan→generate→patch→compile→repair→restore) with a
real semantic retrieval engine. Contract unchanged: `retrieve(state: AgentState) -> AgentState`,
`full_context` populated (now budgeted, plus additive debug fields).

## Knowledge model
Every chunk carries: scope (comune/categoria/cliente), categoria, cliente,
costruttore, board (e.g. ASY011), micro (e.g. STM32H750), layer
(hal/bsp/rtos/middleware/app/ui). Metadata filter BEFORE similarity.
board+micro mandatory — never cross board boundaries.

## Delivered (2026-06-16)
- `rag/config.py` — typed RAG_* env reader.
- `rag/embeddings.py` — abstract Embedder, local OpenAI-compatible `/v1/embeddings`
  default, cloud stub; provider via RAG_EMBED_PROVIDER; same embedder index+query.
- `rag/chunker.py` — tree-sitter semantic C chunking + `// [AI_START_*]…[AI_END_*]`
  ai_block extraction. Never fixed-length.
- `rag/store.py` — ChromaDB PersistentClient wrapper; add/query/reset; $and/$or filters.
- `rag/indexer.py` — offline ingest (index_file/index_repo/infer_layer).
- `rag/query.py` — online retrieve_relevant with layered mandatory board/micro filter.
- `rag/retriever_node.py` — drop-in node; target file whole + budgeted examples
  (RAG_MAX_EXAMPLE_CHARS) + additive retrieved_chunks/retrieval_debug.
- `rag/eval.py` — recall_at_k harness with placeholder EVAL_SET.
- `scripts/build_index.py` — CLI with base metadata.
- `.env.rag.template`, `tests/rag/*` (15 tests, offline FakeEmbedder), `rag/README.md`.

Verification: `pytest tests/rag -q` → 15 passed (offline, no model server needed).

## Out of scope (intentionally not implemented)
Classifier wiring, real index population, real EVAL_SET content, action-perimeter logic.

## Backlog / TODO (human) — domain calibration
- P1: chunking strategy tuning (doc comments, huge functions, file-level context chunk).
- P1: metadata-from-path derivation (infer_layer) for real repo layouts.
- P2: implement CloudEmbedderStub if a hosted provider is ever approved.
- P2: populate EVAL_SET from real repositories; track recall@k over time.
