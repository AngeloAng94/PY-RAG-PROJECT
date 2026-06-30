# Guida all'etichettatura — come popolare l'indice del RAG

Questa guida spiega **come caricare il codice nell'indice del RAG** etichettandolo
correttamente. È la parte di curatela: il valore del RAG dipende quasi tutto da
*cosa* ci mettete dentro e da *come* lo etichettate.

Va letta dopo `GUIDA_SETUP_RAG.md` (che spiega come costruire l'ambiente).

---

## 0. Il concetto in una frase

**C'è un solo indice (una sola "biblioteca"), che cresce.** Ogni prodotto NON ha
un indice a parte: il codice di tutti i prodotti sta nello stesso indice,
distinto da un'**etichetta**. Aggiungere un prodotto = caricare il suo codice con
l'etichetta giusta. Non si ricostruisce mai nulla da zero.

È come una biblioteca unica dove i libri hanno un'etichetta di genere: stanno
tutti sullo stesso scaffale, ma l'etichetta permette di trovare solo quelli che
servono.

---

## 1. Le etichette (i parametri del comando)

Quando caricate del codice con `build_index.py`, lo etichettate con questi campi:

| Etichetta | Cosa indica | Esempio | Obbligatoria? |
|---|---|---|---|
| `--board` | La scheda hardware | `ASY011` | Sì |
| `--micro` | Il microcontrollore | `STM32H750` | Sì |
| `--scope` | Il livello: `comune` / `categoria` / `cliente` | `categoria` | No (default: nessuno) |
| `--categoria` | La famiglia di prodotto | `forno`, `caffe`, `tosaerba` | No (vedi §3) |
| `--cliente` | Il cliente specifico | `acme` | No (vedi §3) |
| `--costruttore` | Il produttore | `acme-srl` | No |

`--board` e `--micro` sono **sempre obbligatori**: il RAG non recupera mai codice
di una scheda diversa, quindi deve sapere di che scheda è il codice che caricate.

---

## 2. I tre livelli (`scope`): comune, categoria, cliente

Il codice si divide in tre livelli. Caricate ogni livello con il suo `--scope`:

**`comune`** — codice condiviso da TUTTI i prodotti (circa il 60%). Es. driver di
base, utility, gestione di periferiche standard.
```bash
python scripts/build_index.py --repo "...\CodiceComune" \
    --board ASY011 --micro STM32H750 --scope comune
```
Per il codice comune, **omettete `--categoria` e `--cliente`** (vengono
etichettati automaticamente come "condiviso" — vedi §3).

**`categoria`** — codice specifico di una famiglia di prodotto (forno, caffè…).
```bash
python scripts/build_index.py --repo "...\Forno\Syen" \
    --board ASY011 --micro STM32H750 --scope categoria --categoria forno
```

**`cliente`** — personalizzazioni di un singolo cliente.
```bash
python scripts/build_index.py --repo "...\ClienteAcme" \
    --board ASY011 --micro STM32H750 --scope cliente --categoria forno --cliente acme
```

Quando l'agente lavora su un forno del cliente Acme, il RAG comporrà i livelli:
recupera il codice **comune + forno + Acme** insieme, senza mai mescolare altri
clienti o altre categorie.

---

## 3. La regola del codice condiviso (importante)

> **Il codice condiviso si etichetta con il valore neutro, NON si lascia vuoto.**

Il codice comune non appartiene a nessuna categoria né cliente. Internamente
viene marcato con un valore speciale (`__none__`). **Non dovete farlo a mano:**
basta **omettere** `--categoria` e/o `--cliente`, e il sistema li imposta da solo
al valore neutro, stampando una nota tipo:
```
[build_index] note: ['cliente'] not provided -> tagged as shared
```

Perché conta: se per sbaglio etichettaste del codice comune con una categoria a
caso, quel codice verrebbe recuperato solo per quella categoria, perdendo il suo
valore di "condiviso da tutti". Quindi: **codice comune → omettete categoria e
cliente**, non inventate valori.

---

## 4. LA REGOLA D'ORO: `--reset` cancella tutto

> **Usate `--reset` SOLO la primissima volta (o per ricostruire da zero).
> Mai dopo, altrimenti cancellate tutto l'indice esistente.**

`--reset` **azzera l'intera biblioteca** prima di ricaricare. Va bene la prima
volta, quando partite da un indice vuoto. Ma se lo usate al secondo caricamento,
cancellate quello che avevate messo prima.

**Sbagliato** (il forno sparisce quando aggiungete il caffè):
```bash
build_index --repo Forno --categoria forno --reset      # carica il forno
build_index --repo Caffe --categoria caffe --reset      # ⚠️ CANCELLA il forno, lascia solo il caffè
```

**Giusto** (l'indice cresce):
```bash
build_index --repo Forno --categoria forno --reset      # prima volta: parte pulito
build_index --repo Caffe --categoria caffe              # aggiunge il caffè, il forno resta
build_index --repo Comune --scope comune                # aggiunge il comune, tutto resta
```

Regola pratica: **`--reset` lo digitate una volta sola, all'inizio di tutto.**
Da lì in poi, mai più (salvo voler ripartire da capo di proposito).

---

## 5. Cosa indicizzare (e cosa no)

**Indicizzate:** il VOSTRO codice applicativo — la logica, i driver scritti da
voi, la gestione della GUI. Tipicamente la cartella del vostro codice (es. la
cartella `Syen/` di un progetto).

**NON indicizzate:**
- **Librerie di terzi** (LVGL, LWIP, FATFS, middleware vari): non sono codice
  vostro, non hanno valore come "esempi vostri", e per giunta non sono di vostra
  proprietà. Escludetele con `--exclude` (es. `--exclude lvgl/ --exclude Drivers/`).
- **Codice generato automaticamente** (es. da STM32CubeMX): è boilerplate, poco
  utile.
- **Immagini e font convertiti in C** (gli array di pixel): vengono **scartati
  automaticamente** dal sistema (riconosce i file-dato dal contenuto). Se per
  qualche motivo volete forzarne uno, c'è `--include`.

Puntare `--repo` direttamente alla cartella del vostro codice (non alla radice
del progetto con tutte le librerie dentro) è il modo più semplice e pulito: meno
roba inutile, più veloce, e niente problemi di proprietà del codice.

---

## 6. Verificare cosa è stato caricato

Strumento di sola lettura per controllare l'indice:

```bash
python rag/inspect.py --group-by categoria    # quante parti per categoria
python rag/inspect.py --group-by board         # niente codice di schede sbagliate
python rag/inspect.py --board ASY011 --categoria forno   # cosa c'è per quel prodotto
```

Controllate sempre dopo un caricamento importante: vi accorgete subito se
qualcosa è finito con l'etichetta sbagliata.

---

## 7. Esempio completo di popolamento

```bash
# 1) Primo caricamento: il codice comune. --reset perché partiamo da zero.
python scripts/build_index.py --repo "...\Comune" \
    --board ASY011 --micro STM32H750 --scope comune --reset

# 2) La categoria forno (codice Syen del forno). NIENTE --reset.
python scripts/build_index.py --repo "...\TFT70_TFTMB_STAND\Syen" \
    --board ASY011 --micro STM32H750 --scope categoria --categoria forno \
    --exclude lvgl/ --exclude Drivers/ --exclude Middlewares/

# 3) Domani, un macinacaffè. STESSO indice, NIENTE --reset.
python scripts/build_index.py --repo "...\Caffe\Syen" \
    --board ASY011 --micro STM32H750 --scope categoria --categoria caffe

# 4) Una personalizzazione cliente. NIENTE --reset.
python scripts/build_index.py --repo "...\ClienteAcme" \
    --board ASY011 --micro STM32H750 --scope cliente --categoria forno --cliente acme

# Verifica finale
python rag/inspect.py --group-by categoria
```

Un solo indice, che cresce a ogni caricamento. L'agente poi filtra per prodotto
al momento della ricerca.

---

## Riepilogo delle regole

1. **Un solo indice**, che cresce: una categoria per prodotto, non un RAG per prodotto.
2. **`--board` e `--micro` sempre**; il RAG non attraversa mai schede diverse.
3. **Codice comune** → omettete `--categoria`/`--cliente` (diventano "condiviso").
4. **`--reset` una volta sola**, all'inizio. Mai più, o cancellate tutto.
5. **Indicizzate il vostro codice**, escludete librerie e codice generato (`--exclude`).
6. **Le immagini-in-C** vengono scartate da sole; verificate con `inspect.py`.
