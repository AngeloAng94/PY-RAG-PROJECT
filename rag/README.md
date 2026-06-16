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
| `rag/inspect.py` | **Read-only, offline** CLI to dump chunk ids + metadata from a built index (optional board/scope/cliente/kind filters, `--group-by`, `--json`, `--show-text`). For auditing scope leakage and seeding the real `EVAL_SET`. |
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

## Populating the index — the shared-code (ABSENT) rule

> **RULE: shared code MUST be tagged explicitly, never left blank.**
> Code that is not specific to a product family has `categoria = "__none__"`
> (the `ABSENT` sentinel from `rag/constants.py`); code that is not specific to a
> customer has `cliente = "__none__"`. **Never store these as an empty/unset
> value.**

**Why.** Retrieval uses a layered fall-through: a query pinned to
`categoria=forno` matches `categoria == "forno" OR categoria == ABSENT`, so
shared (`comune`) chunks survive while a different category (`caffe`) is
excluded. ChromaDB cannot match a *missing* key safely (no `$exists`; `$ne`/`$nin`
also admit every other value), so absence is modelled as an explicit sentinel.
If shared code were left blank it would be **silently excluded** from every
narrowed query — exactly the bug we avoid.

**You get this for free.** Both entry points default any unset
`categoria`/`cliente` to `ABSENT` automatically, so a human who simply omits the
flag gets correct behaviour:

- `scripts/build_index.py` — omit `--categoria` and/or `--cliente`; the CLI tags
  them as `ABSENT` and prints a note showing exactly what was stored.
- `rag/indexer.py` (`_derive_metadata`) — enforces the same default for any
  programmatic ingest (handles both `None` and empty string).

```bash
# Shared / comune code: omit --categoria and --cliente (both -> ABSENT)
python scripts/build_index.py --repo /path/to/common \
    --board ASY011 --micro STM32H750 --scope comune

# Product-family code: pin --categoria, omit --cliente (cliente -> ABSENT)
python scripts/build_index.py --repo /path/to/coffee \
    --board ASY011 --micro STM32H750 --scope categoria --categoria caffe

# Customer-specific code: pin both
python scripts/build_index.py --repo /path/to/acme \
    --board ASY011 --micro STM32H750 --scope cliente --categoria caffe --cliente acme
```

Verify with the read-only inspector: `python rag/inspect.py --group-by categoria`
should show a `__none__` bucket for your shared chunks (and never blanks).

## What is intentionally NOT implemented (out of scope)

- Wiring to the real classifier (session dimensions are read from state/env for now).
- Index population with real code.
- Real `EVAL_SET` content.
- Any agent-level "action perimeter" logic.

Search the code for `# TODO (human):` for every spot that needs domain calibration
(chunking strategy + metadata-from-path derivation).
