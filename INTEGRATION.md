# INTEGRATION — wiring the RAG engine to the real agent

The `rag/` package is reviewed and stable. This document lists the **exact human
steps** to connect it to the existing LangChain + LangGraph agent. Nothing here
changes agent behaviour automatically — each item is a deliberate decision for
the integrating engineer.

The graph itself is unchanged:
`classify → retrieve → plan → generate → patch → compile → repair → restore`.

---

## 1. Reconcile the node name: `retrieve` vs `retrieve_context`

`rag/retriever_node.py` exposes the node as **`retrieve(state) -> state`**, but
the existing graph registers the node under the key **`retrieve_context`**.
Behaviour is identical; only the registered name differs. Pick ONE:

- **Option A — register under the graph's key (no rename):**
  ```python
  from rag.retriever_node import retrieve
  builder.add_node("retrieve_context", retrieve)   # keep existing edges/keys
  ```
- **Option B — rename the function** to match the graph, if you prefer the name
  in `graph.py` to be authoritative.

Do **not** change the node signature or the graph edges — only the binding.

---

## 2. Decide on `target_headers` (the target's `.h` files)

The old node loaded the target's header files; this version loads only the
target `.c` whole (always, verbatim) plus the budgeted retrieved examples.

Decision for wiring: should `state["target_headers"]` (the API surface the edit
must respect) also be injected verbatim into `full_context`?

- If **yes**, read those files and prepend them as their own section *before*
  the retrieved examples, and count them against (or exempt them from) the
  `RAG_MAX_EXAMPLE_CHARS` budget — your call. Headers are usually small and
  high-value, so exempting them (like the target `.c`) is reasonable.
- If **no**, leave as-is; the headers' symbols still surface via the index if
  the `.h` files were indexed by `build_index.py`.

This is intentionally left unimplemented (a domain decision), see the
`WIRING NOTES (human)` block in `rag/retriever_node.py`.

---

## 3. Populate board / micro / scope from the real classifier

Retrieval **refuses to run without `board` and `micro`** (it must never cross a
board boundary) and degrades to "target file only" when they are missing. The
node currently reads the session dimensions from `state` first, then falls back
to env vars (`RAG_SESSION_*`) — a stop-gap for manual/offline runs.

For production, have the **`classify` node populate these on `AgentState`** so
the fallback is never needed:

| Dimension     | Type            | Required | Notes |
|---------------|-----------------|----------|-------|
| `board`       | `str`           | **yes**  | e.g. `ASY011`; hard guard, no fallback |
| `micro`       | `str`           | **yes**  | e.g. `STM32H750` |
| `scope`       | `list[str]`     | no       | composable layers, e.g. `["comune","categoria","cliente"]`; omit = all |
| `categoria`   | `str`           | no       | one product family; matches value **or** shared (`ABSENT`) |
| `cliente`     | `str`           | no       | one customer; matches value **or** shared (`ABSENT`) |

The node already accepts `scope` as a real list on the state or a
comma-separated string. Remove the `RAG_SESSION_*` env fallback once the
classifier is authoritative (optional). See the `# TODO (human):` marker in
`rag/retriever_node.py` (`_session_dimension`).

---

## 4. Build the index on a real repo (offline, once per ingest)

Retrieval returns nothing until the index is populated. Run the offline CLI per
source set, passing the correct base metadata. **Shared code: omit
`--categoria`/`--cliente`** — they are tagged `ABSENT` automatically so shared
chunks survive narrower filters (never leave them blank).

```bash
cp .env.rag.template .env        # set RAG_* and point RAG_EMBED_BASE_URL at your local runtime

# shared / comune code (categoria & cliente default to ABSENT)
python scripts/build_index.py --repo /path/to/common \
    --board ASY011 --micro STM32H750 --scope comune --reset

# product-family code
python scripts/build_index.py --repo /path/to/coffee \
    --board ASY011 --micro STM32H750 --scope categoria --categoria caffe

# customer-specific code
python scripts/build_index.py --repo /path/to/acme \
    --board ASY011 --micro STM32H750 --scope cliente --categoria caffe --cliente acme
```

Audit what landed where (read-only):
```bash
python rag/inspect.py --group-by board        # check no cross-board leakage
python rag/inspect.py --group-by categoria    # expect a __none__ bucket for shared code
```

**Same embedder for index and query**: whatever `RAG_EMBED_PROVIDER` /
`RAG_EMBED_MODEL` you index with MUST be used at query time. Changing the
embedding model means re-indexing from scratch.

---

## 5. Measure before tuning (next phase, needs real data)

Further tuning of error-hint length, chunking strategy, and `infer_layer` can
only be done usefully against a **real index + a real `EVAL_SET`**:

1. Populate `EVAL_SET` in `rag/eval.py` with real cases (query + expected chunk
   ids; copy ids from `python rag/inspect.py --json`).
2. Run `recall_at_k(store, embedder, k)` and establish a baseline.
3. Only then tune the `# TODO (human):` spots, measuring recall after each change.

---

## Out of scope (by design — do not implement blindly)

- Wiring to the real classifier logic (only the data hand-off is described above).
- Index population with real code.
- Real `EVAL_SET` content.
- Any agent-level "action perimeter" logic.
