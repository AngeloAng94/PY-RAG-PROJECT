"""Tests for rag.query + rag.eval — layered filter and recall@k harness."""

from __future__ import annotations

import pytest

from rag.constants import ABSENT
from rag.eval import EvalCase, recall_at_k
from rag.indexer import index_file
from rag.query import build_where, retrieve_relevant
from rag.store import ChromaStore


def test_build_where_requires_board_and_micro():
    with pytest.raises(ValueError):
        build_where(board="", micro="STM32H750")
    with pytest.raises(ValueError):
        build_where(board="ASY011", micro="")


def test_build_where_composes_optional_dims():
    where = build_where(
        board="ASY011",
        micro="STM32H750",
        scope=["comune", "categoria", "cliente"],
        categoria="caffe",
    )
    assert "$and" in where
    clauses = where["$and"]
    clause_keys = [list(c.keys())[0] for c in clauses]
    assert "board" in clause_keys and "micro" in clause_keys
    # scope is a composable $in (any-of) list.
    scope_clause = next(c for c in clauses if "scope" in c)
    assert scope_clause["scope"] == {"$in": ["comune", "categoria", "cliente"]}
    # categoria is now a layered fall-through: target value OR ABSENT sentinel.
    or_clauses = [c for c in clauses if "$or" in c]
    assert any(
        c["$or"] == [{"categoria": {"$eq": "caffe"}}, {"categoria": {"$eq": ABSENT}}]
        for c in or_clauses
    )


def test_build_where_layer_and_costruttore_use_in():
    where = build_where(
        board="ASY011",
        micro="STM32H750",
        layer=["ui", "app"],
        costruttore=["acme-srl"],
    )
    clauses = where["$and"]
    layer_clause = next(c for c in clauses if "layer" in c)
    costruttore_clause = next(c for c in clauses if "costruttore" in c)
    assert layer_clause["layer"] == {"$in": ["ui", "app"]}
    assert costruttore_clause["costruttore"] == {"$in": ["acme-srl"]}


def test_build_where_omitting_scope_means_all():
    where = build_where(board="ASY011", micro="STM32H750")
    clause_keys = [list(c.keys())[0] for c in where["$and"]]
    # Only the mandatory board/micro constraints remain — no scope clause.
    assert clause_keys == ["board", "micro"]


@pytest.fixture
def populated_store(tmp_path, fake_embedder):
    s = ChromaStore(index_path=str(tmp_path / "idx"), collection_name="q_col")
    s.reset()
    docs = [
        "draw header label coffee brewing screen",
        "handle start button click event coffee",
        "oven temperature control loop forno",
    ]
    metas = [
        {"board": "ASY011", "micro": "STM32H750", "categoria": "caffe"},
        {"board": "ASY011", "micro": "STM32H750", "categoria": "caffe"},
        {"board": "ASY099", "micro": "STM32F4", "categoria": "forno"},
    ]
    s.add(
        ids=["draw1", "evt1", "oven1"],
        embeddings=fake_embedder.embed_documents(docs),
        documents=docs,
        metadatas=metas,
    )
    return s


def test_retrieve_relevant_filters_then_ranks(populated_store, fake_embedder):
    res = retrieve_relevant(
        store=populated_store,
        embedder=fake_embedder,
        query="draw header label brewing",
        board="ASY011",
        micro="STM32H750",
        categoria="caffe",
        k=5,
    )
    ids = [r["id"] for r in res]
    assert "oven1" not in ids  # filtered out by board
    assert ids[0] == "draw1"  # best similarity ranked first


def test_recall_at_k_with_real_ids(populated_store, fake_embedder):
    cases = [
        EvalCase(
            query="draw header label brewing",
            board="ASY011",
            micro="STM32H750",
            categoria="caffe",
            expected_ids=["draw1"],
        ),
        EvalCase(
            query="start button click",
            board="ASY011",
            micro="STM32H750",
            categoria="caffe",
            expected_ids=["evt1"],
        ),
    ]
    report = recall_at_k(populated_store, fake_embedder, k=3, eval_set=cases)
    assert report.total == 2
    assert report.recall == 1.0


@pytest.fixture
def layered_store(tmp_path, fake_embedder):
    """Same board/micro, chunks spread across the three knowledge layers."""
    s = ChromaStore(index_path=str(tmp_path / "idx"), collection_name="layer_col")
    s.reset()
    docs = [
        "shared utility ring buffer comune",          # comune
        "coffee brewing screen layout categoria",     # categoria
        "customer acme custom splash cliente",        # cliente
        "other board unrelated code",                 # different board
    ]
    metas = [
        {"board": "ASY011", "micro": "STM32H750", "scope": "comune"},
        {"board": "ASY011", "micro": "STM32H750", "scope": "categoria"},
        {"board": "ASY011", "micro": "STM32H750", "scope": "cliente"},
        {"board": "ASY099", "micro": "STM32F4", "scope": "cliente"},
    ]
    s.add(
        ids=["comune1", "categoria1", "cliente1", "other1"],
        embeddings=fake_embedder.embed_documents(docs),
        documents=docs,
        metadatas=metas,
    )
    return s


def test_scope_list_composes_multiple_layers(layered_store, fake_embedder):
    # (a) a multi-element scope list returns chunks from EACH listed layer.
    res = retrieve_relevant(
        store=layered_store,
        embedder=fake_embedder,
        query="anything",
        scope=["comune", "categoria", "cliente"],
        board="ASY011",
        micro="STM32H750",
        k=10,
    )
    ids = {r["id"] for r in res}
    assert ids == {"comune1", "categoria1", "cliente1"}  # all three layers
    assert "other1" not in ids  # (b) different board still excluded


def test_scope_list_subset_excludes_unlisted_layer(layered_store, fake_embedder):
    res = retrieve_relevant(
        store=layered_store,
        embedder=fake_embedder,
        query="anything",
        scope=["comune", "cliente"],  # categoria intentionally omitted
        board="ASY011",
        micro="STM32H750",
        k=10,
    )
    ids = {r["id"] for r in res}
    assert ids == {"comune1", "cliente1"}
    assert "categoria1" not in ids


def test_board_micro_still_enforced_with_scope(layered_store, fake_embedder):
    # (b) board/micro remain mandatory even when scope is provided.
    with pytest.raises(ValueError):
        retrieve_relevant(
            store=layered_store,
            embedder=fake_embedder,
            query="anything",
            scope=["comune", "cliente"],
            board="ASY011",
            micro="",  # missing micro -> refuse
            k=10,
        )


def test_omitting_scope_returns_all_layers(layered_store, fake_embedder):
    # (c) no scope -> "all scopes" within the mandatory board/micro filter.
    res = retrieve_relevant(
        store=layered_store,
        embedder=fake_embedder,
        query="anything",
        board="ASY011",
        micro="STM32H750",
        k=10,
    )
    ids = {r["id"] for r in res}
    assert ids == {"comune1", "categoria1", "cliente1"}
    assert "other1" not in ids  # still board-bounded


# --- comune fall-through (categoria == X OR ABSENT) -------------------------

@pytest.fixture
def categoria_store(tmp_path, fake_embedder):
    """Chunks mirroring how the indexer stores them: comune carries ABSENT."""
    s = ChromaStore(index_path=str(tmp_path / "idx"), collection_name="cat_col")
    s.reset()
    docs = [
        "shared utility comune ring buffer",   # comune: categoria ABSENT
        "oven temperature control forno",      # categoria forno
        "coffee brewing screen caffe",         # categoria caffe
    ]
    metas = [
        {"board": "ASY011", "micro": "STM32H750", "scope": "comune", "categoria": ABSENT},
        {"board": "ASY011", "micro": "STM32H750", "scope": "categoria", "categoria": "forno"},
        {"board": "ASY011", "micro": "STM32H750", "scope": "categoria", "categoria": "caffe"},
    ]
    s.add(
        ids=["comune1", "forno1", "caffe1"],
        embeddings=fake_embedder.embed_documents(docs),
        documents=docs,
        metadatas=metas,
    )
    return s


def test_comune_survives_categoria_filter(categoria_store, fake_embedder):
    # comune chunk (categoria ABSENT) returned ALONGSIDE forno; caffe excluded.
    res = retrieve_relevant(
        store=categoria_store,
        embedder=fake_embedder,
        query="anything",
        scope=["comune", "categoria"],
        categoria="forno",
        board="ASY011",
        micro="STM32H750",
        k=10,
    )
    ids = {r["id"] for r in res}
    assert "comune1" in ids  # shared code survives the categoria filter
    assert "forno1" in ids   # target category present
    assert "caffe1" not in ids  # a different category is still excluded


def test_comune_survives_categoria_filter_end_to_end(tmp_path, fake_embedder):
    """Prove the indexer actually writes categoria=ABSENT for comune files."""
    index_path = str(tmp_path / "idx")
    store = ChromaStore(index_path=index_path)
    store.reset()

    comune_c = tmp_path / "ringbuffer.c"
    comune_c.write_text("void rb_init(void) { return; }\n", encoding="utf-8")
    forno_c = tmp_path / "oven.c"
    forno_c.write_text("void oven_loop(void) { return; }\n", encoding="utf-8")

    # comune ingest: scope=comune, NO categoria provided -> indexer fills ABSENT.
    index_file(str(comune_c), store, fake_embedder,
               base_metadata={"board": "ASY011", "micro": "STM32H750", "scope": "comune"})
    # categoria ingest: categoria=forno.
    index_file(str(forno_c), store, fake_embedder,
               base_metadata={"board": "ASY011", "micro": "STM32H750", "scope": "categoria", "categoria": "forno"})

    # Sanity: the comune chunk really carries categoria=ABSENT.
    rows = store.list_chunks(where={"categoria": {"$eq": ABSENT}})
    assert any("rb_init" == r["metadata"].get("symbol") for r in rows)

    res = retrieve_relevant(
        store=store,
        embedder=fake_embedder,
        query="oven loop init",
        scope=["comune", "categoria"],
        categoria="forno",
        board="ASY011",
        micro="STM32H750",
        k=10,
    )
    symbols = {r["metadata"].get("symbol") for r in res}
    assert "rb_init" in symbols   # comune survives
    assert "oven_loop" in symbols  # forno present


def test_cliente_filter_keeps_comune_and_categoria(categoria_store, fake_embedder):
    # Pinning a customer must still admit non-customer (ABSENT cliente) chunks.
    # categoria_store chunks have no cliente key, so the indexer-equivalent here
    # is ABSENT; emulate by adding one cliente-specific chunk.
    categoria_store.add(
        ids=["acme1"],
        embeddings=fake_embedder.embed_documents(["acme custom splash"]),
        documents=["acme custom splash"],
        metadatas=[{"board": "ASY011", "micro": "STM32H750", "scope": "cliente",
                    "categoria": ABSENT, "cliente": "acme"}],
    )
    # Give the comune/forno chunks an explicit cliente=ABSENT for this check.
    categoria_store.add(
        ids=["comune1", "forno1", "caffe1"],
        embeddings=fake_embedder.embed_documents(["a", "b", "c"]),
        documents=["a", "b", "c"],
        metadatas=[
            {"board": "ASY011", "micro": "STM32H750", "scope": "comune", "categoria": ABSENT, "cliente": ABSENT},
            {"board": "ASY011", "micro": "STM32H750", "scope": "categoria", "categoria": "forno", "cliente": ABSENT},
            {"board": "ASY011", "micro": "STM32H750", "scope": "categoria", "categoria": "caffe", "cliente": ABSENT},
        ],
    )
    res = retrieve_relevant(
        store=categoria_store,
        embedder=fake_embedder,
        query="anything",
        scope=["comune", "categoria", "cliente"],
        categoria="forno",
        cliente="acme",
        board="ASY011",
        micro="STM32H750",
        k=10,
    )
    ids = {r["id"] for r in res}
    assert {"acme1", "comune1", "forno1"} <= ids  # customer + shared survive
    assert "caffe1" not in ids  # different category still excluded
