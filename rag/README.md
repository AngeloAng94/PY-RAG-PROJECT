# RAG infrastructure (`rag/`)

Retrieval engine that replaces the agent's old `retrieve` node. The graph
(`classify → retrieve → plan → generate → patch → compile → repair → restore`)
and every node signature stay **unchanged**: `retrieve(state: AgentState) -> AgentState`.

## Modules

| Module | Responsibility |
|---|---|
| `rag/config.py` | Single typed reader for all `RAG_*` env vars. |
| `rag/embeddings.py` | Abstract `Embedder` + swappable providers. Default = local OpenAI-compatible `/v1/embeddings`. Cloud is an explicit stub. |
| `rag/chunker.py` | Semantic C chunking via tree-sitter (functions/structs/enums/typedefs/defines) **plus** `// [AI_START_*]…[AI_END_*]` blocks as `ai_block` chunks. Never fixed-length. |
| `rag/store.py` | ChromaDB `PersistentClient` wrapper. `add / query / reset`, composed `$and`/`$or` metadata filters. |
| `rag/indexer.py` | **Offline** ingest: walk repo → chunk → derive metadata → embed → store. Not a graph node. |
| `rag/query.py` | **Online** retrieval. Builds the layered metadata filter (board/micro **mandatory**) then ranks by similarity. |
| `rag/retriever_node.py` | Drop-in `retrieve` node. Target file in full + budgeted retrieved examples → `full_context`, plus additive `retrieved_chunks` / `retrieval_debug`. |
| `rag/eval.py` | `recall_at_k` harness over a placeholder `EVAL_SET`. |
| `scripts/build_index.py` | CLI to build the index with base metadata. |

## Knowledge model

Every chunk carries: `scope` (comune/categoria/cliente), `categoria`, `cliente`,
`costruttore`, `board` (e.g. `ASY011`), `micro` (e.g. `STM32H750`), `layer`
(hal/bsp/rtos/middleware/app/ui).

**Metadata filter happens BEFORE similarity ranking. `board`+`micro` are
mandatory — retrieving another board's code is an error, not a suggestion.**

## Quick start

```bash
cp .env.rag.template .env        # adjust RAG_* values, point at your embed runtime
python scripts/build_index.py --repo /path/to/firmware \
    --board ASY011 --micro STM32H750 --scope categoria --categoria caffe --reset
pytest tests/rag -q              # tests run offline with a deterministic FakeEmbedder
```

## What is intentionally NOT implemented (out of scope)

- Wiring to the real classifier (session dimensions are read from state/env for now).
- Index population with real code.
- Real `EVAL_SET` content.
- Any agent-level "action perimeter" logic.

Search the code for `# TODO (human):` for every spot that needs domain calibration
(chunking strategy + metadata-from-path derivation).
