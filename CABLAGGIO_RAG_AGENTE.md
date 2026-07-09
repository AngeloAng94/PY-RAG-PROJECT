# Come si collegano il RAG e l'agente — spiegazione e istruzioni

**A chi è rivolto.** Al team firmware, in risposta alla domanda "il motore gira:
vuol dire che l'agente sa già dove andare a parare?". Risposta breve: **non
ancora — e il collegamento è a una riga di distanza.** Questo documento spiega
la differenza, cosa manca, e come si fa materialmente l'innesto. Tutte le
istruzioni sono basate sul codice reale di entrambi i lati (il `graph.py`
dell'agente e il `retriever_node.py` del RAG), non su ipotesi.

---

## 1. Cosa vuol dire "il motore gira" (e cosa non vuol dire)

Con la prima indicizzazione riuscita (67 file, 1.420 frammenti, ~11 minuti,
zero errori) abbiamo dimostrato che **la meccanica funziona end-to-end su
codice vero**: il codice viene letto, spezzato in frammenti sensati, trasformato
in vettori, archiviato con le etichette giuste (scheda, micro, categoria,
cliente). E il lato ricerca è testato: a una domanda, il sistema recupera i
frammenti più pertinenti rispettando i filtri di prodotto.

Con una metafora: **la biblioteca è costruita, i primi libri sono sugli
scaffali, etichettati, e il bibliotecario sa andare a prenderli.**

Cosa NON abbiamo ancora dimostrato:

1. **L'agente non è collegato.** Oggi l'agente, quando genera codice, usa
   ancora il suo retriever originale — quello che legge il file target e UN
   file di riferimento fisso. Il RAG gira accanto, non dentro.
2. **Non abbiamo misurato se il bibliotecario porta il libro giusto.** I test
   dimostrano che il recupero funziona meccanicamente; la qualità su domande
   reali ("come si gestisce il timer del forno" → tornano davvero le funzioni
   del timer?) si misura con un insieme di prove (EVAL_SET) ancora da costruire.
3. **L'indice è parziale.** L'agente potrà "sapere" solo ciò che è sugli
   scaffali: il popolamento (estensione a menu_template e alle altre cartelle
   con logica applicativa) è in corso.

E un confine da tenere chiaro: anche a collegamento fatto, **il RAG non decide
niente — consegna materiale.** Capire la richiesta, pianificare e scrivere il
codice resta il mestiere dell'agente (classifier, planner, generator). Il RAG
gli mette davanti gli esempi giusti.

---

## 2. Il punto d'innesto nel codice dell'agente

Nel `graph.py` dell'agente, il grafo è costruito così (codice reale):

```python
from nodes.retriever import retrieve_context
...
g = StateGraph(AgentState)
g.add_node("classify", classify_intent)
g.add_node("retrieve", retrieve_context)   # <-- il punto d'innesto
g.add_node("plan", plan_modification)
...
g.add_edge("classify", "retrieve")
g.add_edge("retrieve", "plan")
```

Il nodo `retrieve` è registrato passando la funzione `retrieve_context`
importata da `nodes/retriever.py`. **Il RAG espone una funzione con lo stesso
identico nome, la stessa firma (`AgentState -> AgentState`) e lo stesso
contratto** — legge e scrive gli stessi campi di stato (`source_content`,
`backup_content`, `reference_content`, `full_context`, `target_file`,
`target_headers`). L'abbiamo verificato campo per campo sul codice della
distribuzione, ed è il motivo per cui esiste l'alias `retrieve_context` nel
modulo del RAG.

## 3. Il cablaggio: la modifica è UNA riga

```python
# graph.py — PRIMA
from nodes.retriever import retrieve_context

# graph.py — DOPO
from rag.retriever_node import retrieve_context
```

Tutto il resto del file resta identico: la registrazione del nodo
(`g.add_node("retrieve", retrieve_context)`), gli archi, i loop di repair.
Nessun altro nodo va toccato.

**Prerequisiti perché quella riga funzioni** (checklist):

1. Il pacchetto `rag/` del progetto PY-RAG-PROJECT deve essere importabile dal
   progetto agente: la via più semplice è copiare la cartella `rag/` dentro
   `src/ai_agent/` (accanto a `nodes/`), oppure aggiungere il percorso della
   repo RAG al `PYTHONPATH`.
2. Dipendenze installate: `pip install -r requirements-rag.txt`.
3. `.env` configurato (copiando `.env.rag.template`) **nella cartella da cui
   gira l'agente**, con `RAG_INDEX_PATH` che punta all'indice già costruito.
4. L'embedder locale attivo (lo stesso `llama-server`/runtime usato per
   l'indicizzazione: serve anche in lettura, per trasformare la domanda in
   vettore).
5. L'indice popolato (quello che state costruendo ora).

## 4. Il punto da decidere insieme: chi dice al RAG cosa filtrare

Il RAG filtra per `board / micro / scope / categoria / cliente`. A runtime,
queste dimensioni descrivono la sessione di lavoro ("sto lavorando sul forno,
scheda ASY011"). Il retriever del RAG le cerca in quest'ordine, **con due
meccanismi già pronti**:

1. **Dallo stato dell'agente** — se lo stato contiene chiavi come `board`,
   `micro`, `categoria`, le usa. Si possono iniettare nello stato iniziale in
   `main.py` (il dizionario `initial_state`), aggiungendo ad esempio:
   ```python
   initial_state = {
       "user_request": user_request,
       "board": "ASY011",
       "micro": "STM32H750",
       "categoria": "forno",
       ...
   }
   ```
2. **Da variabili d'ambiente** — in mancanza dei valori nello stato, legge
   `RAG_SESSION_BOARD`, `RAG_SESSION_MICRO`, `RAG_SESSION_CATEGORIA`, ecc. dal
   `.env`. **Con questa via il cablaggio non richiede nemmeno di toccare
   `main.py`**: si mettono le dimensioni nel `.env` della sessione e basta.

**Raccomandazione:** partire con la via 2 (`.env`) — zero modifiche oltre alla
riga di import; il progetto su cui si lavora è comunque noto a chi lancia
l'agente. In prospettiva, quando il classificatore dell'agente verrà esteso per
dedurre il prodotto dalla richiesta, sarà lui a scrivere queste chiavi nello
stato (via 1) e il `.env` resterà solo come default.

## 5. Cosa cambia nel comportamento (e cosa no)

| | Prima (retriever originale) | Dopo (RAG) |
|---|---|---|
| `full_context` | file target + **un** file di riferimento fisso, per intero | file target + **i frammenti più pertinenti** recuperati dall'indice, filtrati per prodotto, dentro un budget di caratteri |
| Il resto della pipeline | consuma `full_context` | identico — non si accorge del cambio |
| Copertura | il solo esempio cablato a mano | tutto ciò che è nell'indice (e cresce col popolamento) |

Nota sul flusso: nel grafo attuale il nodo `retrieve` gira **una volta per
sessione** (il loop di repair torna a `plan`, non a `retrieve`). Il retriever
del RAG è già predisposto anche per richiami ripetuti, ma con questo grafo non
serve.

## 6. Prova consigliata e rollback

**Prova A/B**, quando l'indice sarà popolato a sufficienza: prendete una
richiesta già usata nei test (es. il caso del pulsante PLAY) e lanciatela una
volta col retriever originale e una col RAG. Confrontate il `full_context`
prodotto e l'esito della generazione. È il modo più concreto per vedere la
differenza.

**Rollback:** ripristinare la riga di import. Un secondo, zero rischio — il
retriever originale resta nel progetto, intatto.

## 7. Quando farlo — la sequenza consigliata

Il cablaggio tecnico si può *provare* anche subito (l'indice parziale basta a
verificare che l'aggancio giri). Ma il passaggio **in uso reale** conviene
farlo dopo due cose: (1) il popolamento completo dell'indice, (2) una prima
misura di qualità del recupero (l'EVAL_SET). Collegare all'agente un indice
mezzo vuoto darebbe risposte deludenti — e giudicheremmo male uno strumento che
in realtà ha solo la dispensa vuota.

---

*Angelo Anglani · Cablaggio RAG–Agente v1.0 · Luglio 2026*
