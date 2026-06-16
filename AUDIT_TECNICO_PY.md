# AUDIT TECNICO — RAG Infrastructure (embedded-code-generation agent)
**Data**: 2026-06-16
**Versione codebase**: git `main` @ `bb771dd` (10 commit totali; nessun tag SemVer presente → versione formale N/D)
**Autore**: Angelo Anglani
**Stato**: scaffolding completo e revisionato · 36 test verdi (offline) · iterazioni concluse (hold)

> Ambito dell'audit: pacchetto `rag/` + `scripts/` + `tests/rag/` (lo scaffolding di retrieval consegnato).
> Le cartelle `backend/` e `frontend/` sono lo **scaffold pre-esistente** dell'ambiente (template FastAPI/React non collegato a `rag/`) e sono trattate separatamente dove pertinente.

---

## 1. PANORAMICA ARCHITETTURA ATTUALE

### 1.1 Schema a blocchi

```
                        ┌──────────────────────────────────────────────┐
                        │            AGENTE LangGraph (esistente)        │
                        │  classify → retrieve → plan → generate →       │
                        │            patch → compile → repair → restore  │
                        └───────────────────┬──────────────────────────┘
                                            │ (drop-in, contratto invariato)
                                            ▼
                            rag/retriever_node.py :: retrieve(state)->state
                            (repair loop: arricchisce la query con l'errore di compile)
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              ▼                             ▼                             ▼
     file target (intero)          rag/query.py                  budget char (.env)
                                  retrieve_relevant()            RAG_MAX_EXAMPLE_CHARS
                                          │
                          build_where (filtro metadati PRIMA)
                          board/micro OBBLIGATORI + scope/layer/
                          costruttore ($in) + categoria/cliente
                          (X OR ABSENT, fall-through)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                                ▼
        rag/embeddings.py                                   rag/store.py
        Embedder (ABC)                                      ChromaStore
        OpenAICompatibleEmbedder (local default)            PersistentClient (disco)
        CloudEmbedderStub (estensione)                      collection "embedded_code"
                  │   (STESSO embedder index+query)                 ▲
                  │                                                 │
   ════════════════ PERCORSO OFFLINE (ingest, separato) ═══════════│════════
                  │                                                 │
        scripts/build_index.py ──► rag/indexer.py ──► rag/chunker.py (tree-sitter)
        (CLI, metadati base)       index_repo/index_file   chunk semantico C + ai_block

        rag/inspect.py  (audit read-only dell'indice)
        rag/eval.py     (recall@k, EVAL_SET placeholder)
```

### 1.2 Stack tecnologico

| Componente | Tecnologia | Versione |
|---|---|---|
| Linguaggio | Python | 3.11.15 |
| Vector store | ChromaDB (`PersistentClient`, on-disk) | 1.5.9 |
| Parsing C (chunking semantico) | tree-sitter | 0.25.2 |
| Grammatica C | tree-sitter-c | 0.24.2 |
| Embeddings (default locale) | HTTP OpenAI-compatible `/v1/embeddings` via `requests` | requests 2.34.2 |
| Config / env | python-dotenv | >=1.0.1 (da `backend/requirements.txt`) |
| Testing | pytest | 8.x (>=8.0.0) |
| Embedding runtime | Esterno/locale (Ollama, llama.cpp, LM Studio, vLLM, TEI) | N/D (non incluso nel repo) |
| CI/CD | Nessuno | N/D |

### 1.3 Tipo di architettura

Libreria/pacchetto modulare Python (non un servizio web). Architettura a due percorsi
nettamente separati:
- **Offline (ingest)**: walk repo → chunking semantico → derivazione metadati → embedding → persistenza ChromaDB.
- **Online (retrieval)**: nodo drop-in che filtra per metadati **prima** della similarità e assembla il contesto entro un budget.

Principio guida "local-first": di default nessun dato lascia la macchina (embedder locale, telemetria Chroma disabilitata). Lo **stesso** embedder è usato per indicizzazione e query. Vincolo di dominio forte: `board`/`micro` obbligatori — mai attraversare il confine di board.

**Portabilità**: componente auto-contenuto e indipendente dalla macchina. Pure Python 3.9+, dipende solo da `chromadb`, `tree-sitter`, `tree-sitter-c`, `requests` (+ `python-dotenv` opzionale), dichiarate in `requirements-rag.txt`. Nessun percorso assoluto né accoppiamento a FastAPI/Mongo/ambiente; gli script ricavano la root dalla propria posizione e l'index path è relativo. Verificato eseguendo la suite (36 PASS) da una copia in una directory diversa da quella di sviluppo.

---

## 2. STRUTTURA DEL CODICE

### 2.1 Directory tree (file rilevanti)

```
/app
├── rag/                          # pacchetto di retrieval (oggetto dell'audit)
│   ├── __init__.py               # superficie pubblica + docstring d'insieme
│   ├── config.py                 # lettura tipizzata delle env RAG_*
│   ├── constants.py              # sentinel ABSENT + FALLBACK_DIMS
│   ├── embeddings.py             # Embedder (ABC) + provider locale/cloud + factory
│   ├── chunker.py                # chunking semantico C (tree-sitter) + ai_block + file_context
│   ├── store.py                  # wrapper ChromaDB PersistentClient
│   ├── indexer.py                # ingest OFFLINE (index_file/index_repo/infer_layer)
│   ├── query.py                  # retrieval ONLINE (build_where, retrieve_relevant)
│   ├── retriever_node.py         # nodo drop-in retrieve(state)->state
│   ├── eval.py                   # harness recall@k + EVAL_SET placeholder
│   ├── inspect.py                # CLI audit indice (read-only, offline)
│   └── README.md                 # documentazione del pacchetto
├── scripts/
│   └── build_index.py            # CLI per costruire l'indice da un repo
├── tests/
│   └── rag/
│       ├── conftest.py           # FakeEmbedder deterministico + sorgente C di esempio
│       ├── test_chunker.py       # 10 test
│       ├── test_store.py         # 4 test
│       ├── test_query_eval.py    # 13 test
│       ├── test_retriever_node.py# 5 test (incl. repair loop)
│       └── test_inspect.py       # 4 test
├── AUDIT_TECNICO_PY.md           # questo documento
├── INTEGRATION.md                # checklist di cablaggio all'agente reale
├── requirements-rag.txt          # dipendenze AUTO-CONTENUTE del componente (portabilità)
├── .env.rag.template             # template chiavi RAG_*
├── backend/  frontend/           # scaffold pre-esistente (NON collegato a rag/)
└── memory/   test_reports/       # artefatti di piattaforma
```

### 2.2 Analisi file principali

| File | Righe | Responsabilità |
|---|---:|---|
| `rag/__init__.py` | 37 | Documentazione d'insieme; esporta `RagConfig`, `load_config`. |
| `rag/config.py` | 77 | `RagConfig` (dataclass frozen) + `load_config()`; unico punto di lettura env, default conservativi. |
| `rag/constants.py` | 34 | Sentinel `ABSENT="__none__"`; `FALLBACK_DIMS=("categoria","cliente")`; razionale modello. |
| `rag/embeddings.py` | 198 | `Embedder` (ABC), `OpenAICompatibleEmbedder` (default locale), `CloudEmbedderStub`, factory `get_embedder`; `.signature` per rilevare mismatch. |
| `rag/chunker.py` | 392 | Chunking semantico via tree-sitter (function/struct/enum/typedef/define), `file_context` raggruppato, `ai_block` con scan riga-per-riga, doc-comment in testa. |
| `rag/store.py` | 168 | `ChromaStore`: `add/query/reset/count/list_chunks`; filtri `$and`/`$or`/`$in`; sanitizzazione metadati. |
| `rag/indexer.py` | 179 | Ingest offline: `index_file`, `index_repo`, `infer_layer`, derivazione metadati + fallback `ABSENT`. |
| `rag/query.py` | 160 | `build_where` (filtro a strati) + `retrieve_relevant`; board/micro obbligatori; ordine filtro→similarità. |
| `rag/retriever_node.py` | 314 | Nodo drop-in `retrieve`; file target intero + esempi a budget; **arricchimento query nel repair loop** (`_compile_error_hint`, cap 300 char); campi additivi `retrieved_chunks`/`retrieval_debug` (incl. `repair_pass`, `error_hint`); import tollerante di `AgentState`; note di wiring documentate. |
| `rag/eval.py` | 113 | `EvalCase`, `recall_at_k`, `RecallReport`; `EVAL_SET` con 2 placeholder. |
| `rag/inspect.py` | 163 | CLI read-only: dump id+metadati, filtri, `--group-by`, `--json`, `--show-text`. |
| `scripts/build_index.py` | 124 | CLI ingest: board/micro (obbligatori) + scope/categoria/cliente/costruttore + `--reset`; **normalizza a `ABSENT` le dimensioni categoria/cliente omesse** e stampa una nota esplicita. |
| **Totale codice `rag/` + `scripts/`** | **1959** | — |
| **Totale `tests/rag/`** | **849** | 36 funzioni di test |
| **Totale complessivo** | **2808** | (incl. file `__init__`) |

> Documenti di supporto nella root: `INTEGRATION.md` (checklist di cablaggio all'agente reale), `README.md` del pacchetto (`rag/README.md`), `.env.rag.template`.

---

## 3. DATABASE & MODELLO DATI

Persistenza tramite **ChromaDB** (vector store su disco, `PersistentClient`). Nessun RDBMS o MongoDB usato dal pacchetto `rag/` (lo scaffold `backend/` include driver Mongo ma è separato e non collegato).

### 3.1 Elenco collections / tabelle

| Nome | Documenti/Righe stimate | Campi principali (metadati) |
|---|---|---|
| `embedded_code` (collection Chroma, spazio cosine) | N/D (dipende dall'indice costruito; vuota di default — nessun dato reale incluso) | `scope`, `categoria`, `cliente`, `costruttore`, `board`, `micro`, `layer`, `kind`, `symbol`, `source_path`, `start_line`, `end_line`, `embedder` |

Per ogni elemento Chroma sono inoltre memorizzati: `id` (SHA1 deterministico di path+kind+symbol+range), `embedding` (vettore prodotto dall'embedder configurato), `document` (testo del chunk).

### 3.2 Modelli / Entità principali

- **`Chunk`** (`chunker.py`, dataclass) — unità semantica: `text, kind, symbol, start_line, end_line, metadata`.
- **`RagConfig`** (`config.py`, dataclass frozen) — snapshot immutabile della configurazione `RAG_*`.
- **`Embedder`** (`embeddings.py`, ABC) — interfaccia astratta; impl. `OpenAICompatibleEmbedder`, stub `CloudEmbedderStub`.
- **`ChromaStore`** (`store.py`) — wrapper di persistenza/ricerca.
- **`EvalCase`** / **`RecallReport`** (`eval.py`, dataclass) — caso di valutazione e report recall@k.
- **`AgentState`** (`retriever_node.py`, `TypedDict`) — importato dall'agente reale se disponibile, altrimenti fallback strutturale locale (campi del contratto + additivi `retrieved_chunks`/`retrieval_debug`).

### 3.3 Enum / Tipi custom

Nel codice non sono presenti `enum.Enum` Python; i tipi categoriali sono **costanti stringa**:

- **Chunk kinds** (etichette `kind`): `function`, `struct`, `enum`, `typedef`, `define`, `file_context`, `ai_block`.
- **Layer canonici** (`indexer._KNOWN_LAYERS`): `hal`, `bsp`, `rtos`, `middleware`, `app`, `ui`.
- **Scope** (dimensione a strati): `comune`, `categoria`, `cliente`.
- **`ABSENT = "__none__"`** (`constants.py`) — sentinel "dimensione non applicabile".
- **`FALLBACK_DIMS = ("categoria", "cliente")`** — dimensioni con fall-through verso `ABSENT`.
- **`PyObjectId` / `BaseDocument`**: N/D (non utilizzati — nessun modello MongoDB nel pacchetto).

---

## 4. SICUREZZA

| Meccanismo | Stato |
|---|---|
| Local-first / nessun dato fuori macchina di default | Implementato (embedder locale default; telemetria Chroma disabilitata) |
| Segreti non hardcoded (chiavi solo da env) | Implementato (`RAG_EMBED_API_KEY` opzionale via env; `.env.rag.template` con valore vuoto) |
| Bearer token opzionale verso runtime embedding | Implementato (header `Authorization` solo se chiave presente) |
| Confine di board/micro (vincolo di sicurezza di dominio) | Implementato (board/micro obbligatori; `build_where` solleva `ValueError`; il nodo degrada a "solo file target") |
| Validazione/sanitizzazione metadati verso Chroma | Implementato (`_sanitize_metadata`: scarta `None`, forza scalari) |
| Gestione robusta degli errori nel nodo (no crash del grafo) | Implementato (try/except → `retrieval_debug.status="error"`) |
| Autenticazione applicativa (JWT/OAuth) nel pacchetto `rag/` | N/D (fuori ambito; non pertinente a una libreria di retrieval) |
| Validazione percorso file target / path traversal | Parziale (lettura diretta del file target; nessuna sandbox — accettabile per tool interno offline) |
| Timeout chiamate di rete embedding | Implementato (default 60s in `OpenAICompatibleEmbedder`) |
| Rate limiting / retry verso runtime embedding | Mancante (nessun backoff/retry) |
| Pinning/lock delle dipendenze | Parziale (`requirements.txt` con vincoli misti `==`/`>=`; nessun lockfile dedicato per `rag/`) |

---

## 5. TESTING & QUALITÀ

### 5.1 Test automatici

Suite eseguita offline con `FakeEmbedder` deterministico (nessun runtime/rete). Esito: **36 passed** (1 warning di deprecation proveniente da OpenTelemetry/Chroma, non dal codice del progetto).

| Area | Test | Stato |
|---|---:|---|
| Chunking (`test_chunker.py`) | 10 | PASS — kinds semantici, simboli funzione, `ai_block` (tag/ripetuti/END mancante), `file_context` (include+globali), doc-comment, range righe |
| Store (`test_store.py`) | 4 | PASS — add/count, filtro board, filtro composto `$or`, reset |
| Query + Eval (`test_query_eval.py`) | 13 | PASS — board/micro obbligatori, `$in` su scope/layer/costruttore, composizione strati, fall-through comune (`X OR ABSENT`) anche end-to-end via indexer, recall@k |
| Retriever node (`test_retriever_node.py`) | 5 | PASS — popolamento `full_context`+debug, cap budget, skip senza board/micro, **arricchimento query nel repair loop**, nessun arricchimento su compile riuscito |
| Inspect CLI (`test_inspect.py`) | 4 | PASS — `list_chunks` metadati/filtro/testo, `--json`, `--group-by` |
| **Totale** | **36** | **PASS** |

Copertura di codice (coverage %) misurata: N/D (non configurata `pytest-cov`).

### 5.2 CI/CD

Nessuna pipeline CI/CD presente (assenza di `.github/`, `.gitlab-ci.yml`, ecc.). Esecuzione test solo manuale (`pytest tests/rag -q`). Versionamento via auto-commit di piattaforma.

---

## 6. FUNZIONALITÀ IMPLEMENTATE — STORICO COMPLETO

| Step | Versione | Funzionalità |
|---|---|---|
| 1 | commit iniziali | Pacchetto `rag/` completo: `config`, `embeddings` (ABC + locale OpenAI-compatible + stub cloud), `chunker` (tree-sitter + `ai_block`), `store` (ChromaDB PersistentClient), `indexer` (offline), `query` (filtro board/micro + similarità), `retriever_node` (drop-in, budget), `eval` (recall@k), `scripts/build_index.py`, `.env.rag.template`, test (15 PASS) |
| 2 | `5265518`/`11ef39c` | `rag/inspect.py` — CLI audit read-only/offline dell'indice (`list_chunks`, filtri, `--group-by`, `--json`, `--show-text`); +4 test (19 PASS) |
| 3 | `f98799c` | Fix `chunker`: (a) `file_context` per include/globali/prototipi/assegnazioni globali; (b) scan riga-per-riga degli `ai_block` (tag ripetuti, END mancante gestito); (c) doc-comment in testa; +6 test (25 PASS) |
| 4 | `fadf69a` | `query.scope` diventa lista componibile con `$in`; aggiunti filtri `layer` e `costruttore` (`$in`); board/micro restano obbligatori; +6 test (31 PASS) |
| 5 | `9d8398c` | Fall-through `comune`: `constants.py` (sentinel `ABSENT`), indexer scrive `categoria`/`cliente=ABSENT` quando assenti, `query` filtra `X OR ABSENT`; +3 test (34 PASS) |
| 6 | `914e678`/`ece23d3` | `build_index.py` normalizza a `ABSENT` le dimensioni categoria/cliente omesse e stampa una nota; sezione "Populating the index" nel README con la regola del codice condiviso |
| 7 | `e514a53` | Repair loop: `retriever_node` arricchisce la query con una forma concisa dell'errore di compile (`_compile_error_hint`, cap 300 char); debug `repair_pass`/`error_hint`; note di wiring documentate; +2 test (36 PASS) |
| 8 | `bb771dd` | `INTEGRATION.md` (checklist di cablaggio all'agente reale); chiusura iterazioni di scaffolding (hold) |

*Nota: la mappatura step→commit è ricostruita dalla cronologia (`Initial commit` + 9 auto-commit) ed è indicativa.*

---

## 7. DEBITO TECNICO RESIDUO

| ID | Area | Problema | Priorità |
|---|---|---|---|
| DT-01 | Integrazione agente | `AgentState`/dimensioni di sessione (board/micro/scope) non ancora cablate al classificatore reale; lette da `state`/env con fallback (`# TODO (human)`). Fuori ambito ma bloccante per la produzione. **Passi di cablaggio ora documentati in `INTEGRATION.md`** (nome nodo `retrieve` vs `retrieve_context`, `target_headers`, popolamento dimensioni, `build_index.py` su repo reale). | P1 |
| DT-02 | Qualità retrieval | `EVAL_SET` contiene solo 2 placeholder; recall@k non misurabile su casi reali finché non popolato. | P1 |
| DT-03 | Metadati da path | `infer_layer` è euristica iniziale; da calibrare sul layout reale dei repo (`# TODO (human)`). | P1 |
| DT-04 | Chunking | Tuning strategia chunking (commenti doc, funzioni molto grandi, contesto a livello file) da calibrare (`# TODO (human)`). | P2 |
| DT-05 | Estrazione simboli | `_extract_symbol` copre i casi comuni; dichiaratori C complessi (puntatori a funzione, array, catene typedef) non pienamente gestiti (`# TODO (human)`). | P2 |
| DT-06 | Embeddings cloud | `CloudEmbedderStub` non implementato (solleva `NotImplementedError`); richiede re-indicizzazione se attivato. | P2 |
| DT-07 | Resilienza rete | Nessun retry/backoff né batching configurabile verso il runtime embedding; un timeout interrompe l'ingest del file. | P2 |
| DT-08 | Modello dati comune/categoria | Soluzione sentinel `ABSENT` adottata perché Chroma non sa filtrare "chiave assente" (no `$exists`; `$ne`/`$nin` ammettono altri valori). Decisione corretta ma da validare sui dati reali per coerenza di ingest. | P2 |
| DT-09 | Coverage | Nessuna misura di code coverage configurata. | P3 |
| DT-10 | CI/CD | Assenza di pipeline automatica (lint/test su push). | P3 |
| DT-11 | Dipendenze | Manifest dedicato e auto-contenuto `requirements-rag.txt` (versioni pinnate) **presente** → componente installabile/portabile altrove. Residuo minore: `backend/requirements.txt` resta condiviso con lo scaffold (vincoli misti `==`/`>=`); nessun lockfile/`pyproject.toml` formale. | P3 |
| DT-12 | Warning deprecation | DeprecationWarning OpenTelemetry (dipendenza transitiva di Chroma) nei test; non bloccante. | P3 |

---

## 8. DA COMPLETARE / ROADMAP

**P1 (necessario per produzione)**
- Cablaggio del nodo `retrieve` al classificatore reale per popolare board/micro/scope/categoria/cliente da `AgentState`.
- Popolamento di `EVAL_SET` con casi reali (query + id chunk attesi) e baseline recall@k.
- Indicizzazione dei repository reali con `scripts/build_index.py` (metadati base corretti) e validazione `infer_layer`.

**P2 (qualità/robustezza)**
- Tuning della strategia di chunking e dell'estrazione simboli sui sorgenti reali.
- Implementazione (eventuale) di un provider embedding cloud dietro `CloudEmbedderStub`.
- Retry/backoff e batching configurabile verso il runtime di embedding.
- Validazione del modello comune/categoria/cliente (sentinel `ABSENT`) sui dati reali.

**P3 (igiene di progetto)**
- Configurare `pytest-cov` e una soglia minima di coverage.
- Aggiungere pipeline CI (lint `flake8`/`black`/`isort` già tra le dipendenze + test).
- Separare le dipendenze di `rag/` (extra/gruppo dedicato) ed eventuale lockfile.

**Esplicitamente FUORI AMBITO (da requisiti)**: cablaggio classificatore reale, popolamento indice con codice reale, contenuto dell'`EVAL_SET`, logica di "action perimeter" a livello agente.

---

## 9. NOTE FINALI

**Qualità del codice — buona.** Il pacchetto è modulare, ben documentato (docstring esaustive su ogni modulo) e coerente con i requisiti dichiarati. I confini sono netti: offline (ingest) vs online (retrieval), interfaccia `Embedder` astratta, store isolato dietro un wrapper sottile. Lo stesso embedder è garantito tra indice e query (con `.signature` per rilevare mismatch). I 9 marcatori `# TODO (human):` segnalano in modo trasparente i punti di calibrazione di dominio, evitando "scatole nere" — in linea con l'obiettivo di una scaffolding rivedibile da umani.

**Punti di forza**
- Contratto del nodo `retrieve(state)->state` rispettato e non invasivo (campi additivi, nessuna modifica al grafo/firme).
- Vincolo di dominio "mai attraversare il board" applicato come hard guard con `ValueError` e degradazione sicura.
- Budget di contesto applicato (mai caricamento illimitato); il file target resta sempre intero ma gli esempi sono limitati da `RAG_MAX_EXAMPLE_CHARS`.
- Chunking semantico solido (tree-sitter) con gestione robusta degli `ai_block` (tag ripetuti, END mancante senza "swallow" del file).
- Modello a strati componibile (`$in` su scope/layer/costruttore) e fall-through `comune` corretto e testato anche end-to-end.
- Repair loop consapevole dell'errore: la query viene arricchita con una forma concisa (cap 300 char) dell'errore di compile, così il retrieval propone esempi pertinenti alla correzione invece di ripetere il primo passaggio.
- Portabilità verificata: componente auto-contenuto (`requirements-rag.txt`), pure Python, senza percorsi assoluti né accoppiamento all'ambiente; suite eseguita con successo anche da una directory esterna a quella di sviluppo.
- Suite di test offline deterministica (36 PASS) eseguibile senza runtime/rete grazie al `FakeEmbedder`.

**Rischi tecnici**
- Dipendenza da un runtime di embedding esterno non incluso: senza di esso il percorso online reale non è esercitabile (mitigato nei test dal fake, ma non in produzione).
- Accoppiamento al comportamento di filtro di ChromaDB (assenza di `$exists`): la scelta del sentinel `ABSENT` è corretta ma va mantenuta coerente lato ingest, altrimenti i chunk condivisi verrebbero esclusi.
- Mancanza di CI e coverage: regressioni possibili non intercettate automaticamente.
- Versionamento formale assente (solo hash git); tracciabilità delle release affidata agli auto-commit di piattaforma.

**Raccomandazioni (prioritarie)**
1. Popolare `EVAL_SET` e stabilire una baseline recall@k prima di calibrare chunking/metadati.
2. Cablare le dimensioni di sessione dal classificatore reale e indicizzare un repo pilota per validare `infer_layer` e il modello a strati.
3. Introdurre CI minima (lint + pytest) e `pytest-cov`, e valutare retry/backoff verso il runtime embedding.

Nel complesso: scaffolding maturo, revisionato e in stato di **hold** (iterazioni concluse), pronto per la revisione umana e l'estensione; il debito residuo è prevalentemente di **calibrazione di dominio** (P1/P2 attesi) più alcune voci di **igiene di progetto** (P3), senza criticità bloccanti nel codice consegnato. Il prossimo passo utile (`INTEGRATION.md`) richiede un indice reale e un `EVAL_SET` reale su cui misurare.

---
*Fine audit tecnico. Ultimo aggiornamento: 2026-06-16 (git `main` @ `bb771dd`).*
