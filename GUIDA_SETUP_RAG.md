# Guida di setup — RAG engine

Istruzioni passo-passo per costruire l'ambiente, eseguire i test e creare un
indice. Pensata per chi parte da zero. Se un passo non torna, è segnalato in
fondo nella sezione "Errori comuni".

---

## 0. Cosa c'è nella repo (e cosa NON c'è ancora) — leggere prima

La repo contiene il **motore del RAG**: la parte che costruisce la "biblioteca"
di codice e la interroga. Con questa repo puoi fare due cose:

1. **Eseguire i test** (verifica che il motore funzioni) — non serve nulla di esterno.
2. **Costruire un indice e fare ricerche** — serve un modello di embedding locale (vedi §2).

Quello che **non** è incluso è l'aggancio al resto dell'agente (il pezzo che,
a partire da una richiesta, genera davvero il codice C end-to-end). Quel
collegamento è un passo separato, descritto in `INTEGRATION.md`, e richiede
anche la LLM che scrive il codice (es. Gemini). In altre parole: **oggi questa
repo serve a provare il motore di ricerca, non l'intero agente.**

---

## 1. I due modelli, da non confondere (importante)

Nel sistema completo ci sono **due modelli di AI diversi**, con compiti diversi:

| | Cosa fa | Serve per questa repo? |
|---|---|---|
| **Embedder** (modello di embedding) | Trasforma il codice in "vettori" per la ricerca. Modello piccolo e specializzato. | **Sì**, per costruire l'indice e cercare. |
| **LLM** (modello che scrive il codice) | Genera il codice C dalla richiesta. Modello grande (es. Gemini). | **No**, non in questa fase. Appartiene all'agente completo. |

Quindi, rispondendo alla domanda "servono LLM da eseguire sul PC?":
- Per **provare il motore RAG** (questa repo): ti serve solo l'**embedder** locale (piccolo). Niente LLM da chat.
- La **LLM** che scrive il codice entrerà in gioco solo quando si collegherà il motore all'agente completo (fase successiva).

---

## 2. Di cosa hai bisogno (i tool)

1. **Python 3.9+** e un virtual environment — *già fatto da voi.*
2. **Le dipendenze** da `requirements-rag.txt` — *già fatto da voi.*
3. **`pytest`** per i test (potrebbe non essere incluso nei requirements): se manca, `pip install pytest`.
4. **Un runtime di embedding locale** — solo per costruire indici veri / fare ricerche. Due opzioni semplici (scegline una):
   - **Ollama** (a riga di comando) — leggero, si gestisce da terminale.
   - **LM Studio** (con interfaccia grafica) — più visuale, comodo se preferisci una GUI.

Nessun altro servizio è necessario. Niente cloud, niente chiavi API, niente LLM da chat per questa fase.

---

## 3. Setup passo-passo

### Passo 1 — Ambiente Python (già fatto, qui per completezza)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements-rag.txt
pip install pytest        # se non già presente
```

### Passo 2 — Eseguire i test (NON serve nessun runtime esterno)

I test girano **completamente offline**, con un modello finto: servono solo a
verificare che il motore sia a posto. Eseguili **dalla radice del progetto**
(la cartella che contiene `rag/`, `scripts/`, `tests/`):

```bash
# dalla RADICE del progetto:
python -m pytest tests/rag -q
```

Nota: usa `python -m pytest` (non solo `pytest`) — così Python trova il
pacchetto `rag` senza bisogno di installazioni aggiuntive. Devi vedere `52 passed`.

> Se ottieni `ModuleNotFoundError: No module named 'rag'`, vedi "Errori comuni".

### Passo 3 — Installare un modello di embedding locale

Questo serve **solo** dal Passo 4 in poi (indice vero / ricerche). Per i test del
Passo 2 non serve.

**Opzione A — Ollama (riga di comando):**
1. Installa Ollama dal sito ufficiale.
2. Scarica un modello di embedding:
   ```bash
   ollama pull nomic-embed-text
   ```
3. Ollama espone già l'API compatibile a `http://localhost:11434/v1`. Lascialo in esecuzione.

**Opzione B — LM Studio (interfaccia grafica):**
1. Installa LM Studio.
2. Cerca e scarica un modello di **embedding** (es. "nomic-embed-text").
3. Avvia il server locale di LM Studio (tab "Local Server"): espone l'API a `http://localhost:1234/v1`.

### Passo 4 — Creare il file di configurazione `.env`

Il file `.env.rag.template` è già presente nella radice del progetto (lo trovi
dopo aver clonato la repo). Copialo in `.env`:

```bash
# dalla radice del progetto (dove c'è .env.rag.template):
cp .env.rag.template .env
```

Poi apri `.env` e controlla **una sola riga**, l'indirizzo del runtime di embedding:
- Se usi **Ollama**: `RAG_EMBED_BASE_URL=http://localhost:11434/v1`
- Se usi **LM Studio**: `RAG_EMBED_BASE_URL=http://localhost:1234/v1`

Gli altri valori vanno bene così come sono (li trovi spiegati uno per uno in §4).

### Passo 5 — Costruire un indice

Con il runtime di embedding in esecuzione e il `.env` pronto, costruisci l'indice
puntando alla cartella del codice da indicizzare. **Esegui dalla radice del progetto:**

```bash
python scripts/build_index.py --repo /percorso/al/codice \
    --board ASY011 --micro STM32H750 --scope categoria --categoria caffe --reset
```

Cosa significano i parametri:
- `--repo` : la cartella con il codice C da indicizzare.
- `--board` / `--micro` : la scheda e il microcontrollore di quel codice (obbligatori).
- `--scope` : il livello (`comune`, `categoria` o `cliente`).
- `--categoria` / `--cliente` : da specificare secondo il livello (vedi nota sotto).
- `--reset` : azzera l'indice prima di ricostruirlo (usalo solo la prima volta o per ripartire da zero).

**Cosa indicizzare:** puntate `--repo` alla cartella del **vostro** codice, non alla
radice di un progetto con dentro librerie e asset. Le librerie di terzi (LVGL,
ecc.) e il codice generato non vanno indicizzati: escludeteli con `--exclude`
(es. `--exclude lvgl/ --exclude Drivers/`). Le immagini e i font convertiti in C
(array di pixel) vengono invece **riconosciuti e saltati automaticamente** dal
sistema, in base al contenuto: a fine build trovate il conteggio dei file saltati
come dati. Se mai un file venisse saltato per sbaglio, potete forzarne l'inclusione
con `--include`.

**Regola importante sul codice condiviso:** se stai indicizzando codice *comune*
a tutti, **ometti** `--categoria` e `--cliente` (vengono etichettati
automaticamente come "condiviso"). Non lasciarli vuoti a mano.

```bash
# codice comune (categoria e cliente diventano "condiviso" da soli)
python scripts/build_index.py --repo /percorso/comune --board ASY011 --micro STM32H750 --scope comune --reset

# codice di categoria (es. macchine da caffè)
python scripts/build_index.py --repo /percorso/caffe --board ASY011 --micro STM32H750 --scope categoria --categoria caffe

# codice specifico di un cliente
python scripts/build_index.py --repo /percorso/clienteX --board ASY011 --micro STM32H750 --scope cliente --categoria caffe --cliente clienteX
```

### Passo 6 — Verificare cosa è stato indicizzato

Strumento di sola lettura per controllare l'indice (utile per accorgersi di errori
di etichettatura):

```bash
python rag/inspect.py --group-by board        # quante parti per scheda
python rag/inspect.py --group-by categoria     # deve comparire un gruppo "__none__" per il codice condiviso
```

---

## 4. Le chiavi del file `.env`, spiegate

| Chiave | A cosa serve | Valore consigliato |
|---|---|---|
| `RAG_INDEX_PATH` | Cartella dove salvare l'indice | `./.rag_index` |
| `RAG_EMBED_PROVIDER` | Tipo di embedder: `local` (sul PC) o `cloud` | `local` |
| `RAG_EMBED_MODEL` | Nome del modello di embedding | `nomic-embed-text` |
| `RAG_EMBED_BASE_URL` | Indirizzo del runtime locale | `http://localhost:11434/v1` (Ollama) o `http://localhost:1234/v1` (LM Studio) |
| `RAG_EMBED_API_KEY` | Chiave API (non serve in locale) | un valore qualsiasi, es. `not-needed-for-local` |
| `RAG_TOP_K` | Quanti esempi recuperare per ricerca | `5` |
| `RAG_MAX_EXAMPLE_CHARS` | Tetto di caratteri degli esempi nel contesto | `8000` |
| `RAG_EMBED_TIMEOUT` | Secondi di attesa per ogni richiesta all'embedder | `300` |
| `RAG_MAX_CHUNK_CHARS` | Dimensione massima di un singolo chunk (oltre la quale viene spezzato) | `12000` |
| `RAG_MAX_DATA_LINE_CHARS` | Soglia per riconoscere un file-dato: oltre questa lunghezza di riga, il file è trattato come asset e saltato | `2000` |

Regola d'oro: **lo stesso `RAG_EMBED_MODEL` usato per costruire l'indice deve
essere usato per le ricerche.** Se lo cambi, ricostruisci l'indice da zero.

---

## 5. Errori comuni e soluzioni

**"Non trovo `.env.rag.template`"**
Assicurati di aver clonato l'**ultima** versione della repo: il file è nella
radice. Se hai un clone vecchio, aggiorna con `git pull` o riclona.

**`ModuleNotFoundError: No module named 'rag'` quando lancio i test**
Stai eseguendo da una cartella sbagliata, o `pytest` non trova il pacchetto.
Soluzioni, in ordine:
1. Esegui dalla **radice del progetto** (la cartella con dentro `rag/`).
2. Usa `python -m pytest tests/rag -q` (non solo `pytest`).
3. Se ancora non va: `set PYTHONPATH=.` (Windows) o `export PYTHONPATH=.` (Linux/Mac) prima del comando.

**`pytest: command not found`**
`pytest` non è installato: `pip install pytest`.

**Errore di connessione / "connection refused" quando costruisco l'indice**
Il runtime di embedding non è in esecuzione o l'indirizzo è sbagliato.
1. Verifica che Ollama o LM Studio siano **avviati**.
2. Controlla che `RAG_EMBED_BASE_URL` nel `.env` corrisponda al runtime giusto
   (porta 11434 per Ollama, 1234 per LM Studio).
3. Verifica che il modello di embedding sia stato scaricato (`ollama pull nomic-embed-text`).

**Il build parte ma non trova codice / indice vuoto**
Controlla che `--repo` punti a una cartella che contiene davvero file `.c`/`.h`.

**Il build si interrompe dopo qualche minuto con "Read timed out"**
L'embedder non ha risposto entro il tempo limite su una singola richiesta (di
solito un chunk di codice grande, o un momento di rallentamento del runtime).
1. Verifica di avere `RAG_EMBED_TIMEOUT=300` nel `.env` (il tempo di attesa per
   ogni richiesta; su CPU 60 secondi possono non bastare).
2. Il build ora è resiliente: se un singolo file fallisce, lo salta e continua,
   e a fine build stampa l'elenco dei file saltati. Controlla quel riepilogo per
   capire se è rimasto fuori qualcosa.
3. Non serve una GPU o un PC più potente per *completare* l'operazione: il limite
   è di tempo per richiesta, non di potenza di calcolo.

---

## Nota sullo stato della repo

La repo è già stata sistemata per l'uso: il file `.env.rag.template` è presente
nella radice, e `.env` e la cartella dell'indice (`.rag_index/`) sono esclusi dai
commit (non finiscono mai nella repo). Clonando l'ultima versione hai tutto il
necessario.
