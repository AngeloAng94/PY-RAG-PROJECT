"""
Shared pytest fixtures for the rag/ test suite.

The most important fixture is :class:`FakeEmbedder`: a deterministic, offline
stand-in for a real embedding runtime. It lets every test exercise the full
index/query path WITHOUT a network or a model server, while still honouring the
critical invariant that the SAME embedder is used for indexing and querying.
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence

import pytest

from rag.embeddings import Embedder


class FakeEmbedder(Embedder):
    """Deterministic hash-based embedder for tests (no network).

    It maps text -> a fixed-dimension vector derived from a hash of salient
    tokens. Identical text always yields the identical vector, and texts that
    share tokens land closer together, which is enough to test retrieval order
    deterministically. It is NOT a real semantic model.
    """

    def __init__(self, dim: int = 64, model: str = "fake-test-embedder") -> None:
        self._dim = dim
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def _vec(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        for token in text.lower().split():
            h = hashlib.sha1(token.encode("utf-8")).digest()
            idx = h[0] % self._dim
            vec[idx] += 1.0
        # L2-normalise so cosine distance is well-behaved.
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


SAMPLE_C = """\
#include <stdio.h>

#define MAX_ITEMS 16
#define SQUARE(x) ((x) * (x))

typedef enum {
    STATE_IDLE,
    STATE_RUN
} machine_state_t;

typedef struct {
    int id;
    char name[32];
} item_t;

struct point_s {
    int x;
    int y;
};

enum color_e {
    COLOR_RED,
    COLOR_GREEN
};

static void draw_header(lv_obj_t *parent) {
    lv_obj_t *label = lv_label_create(parent);
    lv_label_set_text(label, "Coffee");
    // [AI_START_DRAW]
    lv_obj_set_style_text_color(label, lv_color_hex(0xFFFFFF), 0);
    // [AI_END_DRAW]
}

void on_start_clicked(lv_event_t *e) {
    // [AI_START_EVENTS]
    machine_state_t st = STATE_RUN;
    (void) st;
    // [AI_END_EVENTS]
}
"""
