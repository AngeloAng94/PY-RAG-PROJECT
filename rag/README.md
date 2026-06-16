# RAG infrastructure (`rag/`)

Semantic retrieval engine that replaces the agent's old `retrieve` node (which
concatenated whole files, uncapped). The LangGraph pipeline and every node
signature stay **unchanged**:

```
classify → retrieve → plan → generate → patch → compile → repair → restore
                 ▲ this node's internals were replaced; contract identical:
                   retrieve(state: AgentState) -> AgentState
```

**Status:** scaffolding complete & reviewed · **36 tests passing** (offline) ·
local-first (no data leaves the machine by default).
For wiring this to the real agent, see [`/INTEGRATION.md`](../INTEGRATION.md).

---

## 1. What it does (in one picture)

```
retrieve(state):
  (a) read TARGET .c file WHOLE        ── always included verbatim (it's being edited)
  (b) build metadata filter            ── board/micro MANDATORY, then scope/categoria/…
  (c) embed query, search ChromaDB     ── FILTER first, THEN rank by similarity
  (d) assemble full_context            ── target file + examples, capped by a char BUDGET
  (e) add retrieved_chunks + retrieval_debug   (additive; no existing field removed)
```

Two paths are kept strictly separate:

- **OFFLINE ingest** — `scripts/build_index.py` → `rag/indexer.py` → `rag/chunker.py`.
- **ONLINE retrieval** — `rag/retriever_node.py` → `rag/query.py` → `rag/store.py`.

The **same embedder** (`rag/embeddings.py`) is used for both — mixing models
silently destroys recall.

---

## 2. Modules

| Module | Responsibility |
|---|---|
| `rag/config.py` | Single typed reader for all `RAG_*` env vars (`load_config`). |
| `rag/constants.py` | The `ABSENT="__none__"` sentinel + `FALLBACK_DIMS` (the layered "shared code" rule). |
| `rag/embeddings.py` | Abstract `Embedder` + swappable providers. Default = **local** OpenAI-compatible `/v1/embeddings`; cloud is an explicit stub. Same embedder for index & query. |
| `rag/chunker.py` | Semantic C chunking via tree-sitter (function/struct/enum/typedef/define), **`// [AI_START_*]…[AI_END_*]` blocks** as `ai_block`, **`file_context`** (includes/globals/prototypes), and leading doc-comments. Never fixed-length. |
| `rag/store.py` | ChromaDB `PersistentClient` wrapper: `add / query / reset / count / list_chunks`; composed `$and`/`$or`/`$in` filters. |
| `rag/indexer.py` | **Offline** ingest: walk repo → chunk → derive metadata → embed → store. Defaults unset `categoria`/`cliente` to `ABSENT`. Not a graph node. |
| `rag/query.py` | **Online** retrieval. Builds the layered metadata filter (board/micro **mandatory**), then ranks by similarity. |
| `rag/retriever_node.py` | Drop-in `retrieve` node. Target file in full + budgeted examples → `full_context`; **enriches the query with the compile error on the repair loop**; adds `retrieved_chunks`/`retrieval_debug`. |
| `rag/eval.py` | `recall_at_k` harness over a placeholder `EVAL_SET`. |
| `rag/inspect.py` | **Read-only, offline** CLI to dump chunk ids + metadata (`--group-by`, `--json`, `--show-text`, metadata filters). For auditing scope leakage and seeding the real `EVAL_SET`. |
| `scripts/build_index.py` | CLI to build the index with base metadata. |

---

## 3. Knowledge model & how retrieval filters

Every chunk carries metadata on these dimensions:

| Dimension | Example | Filter behaviour at query time |
|---|---|---|
| `board` | `ASY011` | **mandatory** · exact (`$eq`) · never crossed |
| `micro` | `STM32H750` | **mandatory** · exact (`$eq`) · never crossed |
| `scope` | `comune` / `categoria` / `cliente` | **composable list** (`$in`), e.g. `["comune","categoria","cliente"]` |
| `layer` | `hal/bsp/rtos/middleware/app/ui` | composable list (`$in`), e.g. `["ui","app"]` |
| `costruttore` | `acme-srl` | composable list (`$in`) |
| `categoria` | `caffe` / `forno` / `tosaerba` | single value **OR `ABSENT`** (shared survives) |
| `cliente` | `acme` | single value **OR `ABSENT`** (shared survives) |

Two rules make this correct:

1. **Filter BEFORE similarity.** Metadata narrows the candidate set first; only
   survivors are ranked by vector distance. **`board`+`micro` are mandatory** —
   retrieving another board's code is an error, not a suggestion. Without them
   the node skips retrieval and returns the target file only.

2. **Layered fall-through (shared code survives a narrower filter).** Pinning
   `categoria=forno` matches `categoria == "forno" OR categoria == ABSENT`, so
   shared (`comune`) chunks come back too, while a *different* category (`caffe`)
   is excluded. Same logic for `cliente`. See §5.

---

## 4. Quick start

```bash
pip install -r requirements-rag.txt   # self-contained deps (chromadb, tree-sitter, requests, …)
cp .env.rag.template .env              # set RAG_* and point RAG_EMBED_BASE_URL at your local embed runtime
python scripts/build_index.py --repo /path/to/firmware \
    --board ASY011 --micro STM32H750 --scope categoria --categoria caffe --reset
pytest tests/rag -q                    # 36 tests, fully offline (deterministic FakeEmbedder, no runtime/network)
```

`.env` keys (see `.env.rag.template`): `RAG_INDEX_PATH`, `RAG_EMBED_PROVIDER`,
`RAG_EMBED_MODEL`, `RAG_EMBED_BASE_URL`, `RAG_EMBED_API_KEY`, `RAG_TOP_K`,
`RAG_MAX_EXAMPLE_CHARS`.

### Portability

The component is **self-contained and machine-independent** — copy `rag/`,
`scripts/`, `tests/` (and `requirements-rag.txt`) anywhere and run from the
project root.

- **Pure Python 3.9+**, depends only on `chromadb`, `tree-sitter`,
  `tree-sitter-c`, `requests` (+ optional `python-dotenv`). No FastAPI / Mongo /
  framework coupling — those belong to a separate scaffold and are not used.
- **No hardcoded paths or environment assumptions.** Everything tunable comes
  from `RAG_*` env vars with sensible defaults; the index path is relative
  (`./.rag_index` by default). Scripts resolve the repo root from their own
  location, so they work from any directory.
- **No network needed to run the tests** (deterministic `FakeEmbedder`). Only
  real indexing/retrieval needs your local embedding runtime (configurable URL).
- Verified to run identically outside the development environment (tests pass
  from a fresh copy in another directory).

---

## 5. Populating the index — the shared-code (`ABSENT`) rule

> **RULE: shared code MUST be tagged explicitly, never left blank.**
> Code not specific to a product family is stored with `categoria = "__none__"`
> (the `ABSENT` sentinel); code not specific to a customer with
> `cliente = "__none__"`. **Never store these as empty/unset.**

**Why.** ChromaDB cannot match a *missing* key safely (there is no `$exists`,
and `$ne`/`$nin` also admit every other value). So "not applicable" is modelled
as an explicit sentinel. If shared code were left blank it would be **silently
excluded** from every narrowed query — exactly the bug we avoid.

**You get this for free.** Both entry points default any unset
`categoria`/`cliente` to `ABSENT`:

- `scripts/build_index.py` — omit `--categoria`/`--cliente`; the CLI tags them
  `ABSENT` and prints a note showing exactly what was stored.
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

Audit what landed where (read-only):

```bash
python rag/inspect.py --group-by board        # no cross-board leakage
python rag/inspect.py --group-by categoria    # expect a __none__ bucket for shared code, never blanks
```

> **Same embedder for index and query.** Whatever `RAG_EMBED_PROVIDER` /
> `RAG_EMBED_MODEL` you index with MUST be used at query time. Changing the
> model means re-indexing from scratch.

---

## 6. Repair loop (compile error → better query)

The graph re-enters `retrieve` after a failed compile (`… → repair → retrieve`).
On that pass the node appends a **short, capped** form of the compile error to
the query, so retrieval surfaces *fix-relevant* examples instead of repeating the
first pass's results:

```
query = "<original request>\nfix compile error: <concise error hint>"   # capped at 300 chars
```

It only triggers when a previous compile actually failed; a first pass or a
successful compile leaves the query unchanged. `retrieval_debug` exposes
`repair_pass` (bool) and `error_hint` for observability.

---

## 7. Out of scope (intentionally NOT implemented)

- Wiring to the real classifier (session dimensions read from state/env for now).
- Index population with real code.
- Real `EVAL_SET` content.
- Any agent-level "action perimeter" logic.

Two wiring decisions are documented (not coded) in
[`/INTEGRATION.md`](../INTEGRATION.md): reconciling the `retrieve` vs
`retrieve_context` node name, and whether to include `target_headers` (`.h`
files) verbatim in `full_context`.

Search the code for **`# TODO (human):`** for every spot that needs domain
calibration (chunking strategy tuning, metadata-from-path derivation, symbol
extraction, cloud embedder, `EVAL_SET`). These can only be tuned usefully
against a real index + a real `EVAL_SET`.
