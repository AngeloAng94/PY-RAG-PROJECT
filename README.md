# RAG infrastructure for the embedded code-generation agent

Semantic retrieval engine that replaces the agent's old `retrieve` node (which
concatenated whole files, uncapped) with metadata-filtered, budget-bounded,
board-scoped retrieval over a local ChromaDB index. **Self-contained, local-first,
pure Python** — no FastAPI/Mongo/web stack.

> The LangGraph pipeline and every node signature stay unchanged:
> `classify → retrieve → plan → generate → patch → compile → repair → restore`,
> with `retrieve(state: AgentState) -> AgentState`.

## Repository layout

```
rag/                    the retrieval engine (see rag/README.md)
scripts/build_index.py  offline CLI to build the index from a repo
tests/rag/              offline test suite (deterministic FakeEmbedder)
conftest.py             makes the package importable from a fresh clone
requirements-rag.txt    self-contained dependencies
.env.rag.template       documented config template (copy to .env)
INTEGRATION.md          how to wire this to the real agent
AUDIT_TECNICO_PY.md     technical audit (IT)
```

## Quick start (from a fresh clone, project root)

```bash
pip install -r requirements-rag.txt
python -m pytest tests/rag -q          # expect: 54 passed (offline, no runtime/network)
cp .env.rag.template .env              # then point RAG_EMBED_BASE_URL at your local embedder
python scripts/build_index.py --repo /path/to/firmware \
    --board ASY011 --micro STM32H750 --scope categoria --categoria caffe --reset
```

**Full documentation:** [`rag/README.md`](rag/README.md) (architecture, knowledge
model, indexing rules, asset auto-skip, repair loop) and
[`INTEGRATION.md`](INTEGRATION.md) (wiring to the agent).

## Highlights

- **Filter-before-similarity**, `board`+`micro` mandatory (never cross a board boundary).
- **Layered scope** (`comune`/`categoria`/`cliente`) with shared-code fall-through (`ABSENT`).
- **Robust indexing**: configurable embed timeout, oversized-chunk splitting, per-file
  resilience, **content-based auto-skip of image-as-C / generated data files**, and
  **live progress feedback** (`found N files…` pre-scan + `[i/N] indexing/skipped …` per file).
- **Local-first**: by default nothing leaves the machine; same embedder for index and query.

- **Drop-in node** exposed as both `retrieve` and `retrieve_context` (identity alias)
  so it binds to the graph regardless of the registered node name — same signature, same logic.

Author: Angelo Anglani.
