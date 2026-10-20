"""Unit tests for embed: fake model, cache roundtrip. No network."""

from types import SimpleNamespace

import pytest

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.transformers.embed.embed import BATCH_SIZE, EmbedTransformer, chunks_of


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr("rag_etl.config.CONFIG", {"CACHE_DIR": str(tmp_path / "cache")}, raising=False)
    (tmp_path / "cache").mkdir()
    return Cache("embed")


def fake_llm(monkeypatch, calls: list):
    def embed_texts(texts):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("rag_etl.transformers.embed.embed.get_model", lambda task: SimpleNamespace(embeddings=embed_texts))


def test_one_vector_per_chunk_in_order(cache, monkeypatch):
    calls: list = []
    fake_llm(monkeypatch, calls)
    chunks = [Resource(title=f"chunk {i}", text=f"text {i}") for i in range(3)]

    EmbedTransformer(cache).transform(chunks)

    assert all(chunk.embedding == [1.0, 0.0] for chunk in chunks)


def test_second_run_reads_from_cache(cache, monkeypatch):
    calls: list = []
    fake_llm(monkeypatch, calls)
    chunk = Resource(title="chunk", text="the same text")

    EmbedTransformer(cache).transform([chunk])
    EmbedTransformer(cache).transform([chunk])

    # The second run embeds nothing: same content, cached vector
    assert len(calls) == 1
    assert chunk.embedding == [1.0, 0.0]


def test_changed_chunks_embed_again(cache, monkeypatch):
    calls: list = []
    fake_llm(monkeypatch, calls)
    EmbedTransformer(cache).transform([Resource(title="chunk", text="before")])
    EmbedTransformer(cache).transform([Resource(title="chunk", text="after")])

    assert len(calls) == 2


def test_calls_are_batched(cache, monkeypatch):
    calls: list = []
    fake_llm(monkeypatch, calls)
    chunks = [Resource(title=f"chunk {i}", text=f"text {i}") for i in range(BATCH_SIZE + 1)]

    EmbedTransformer(cache).transform(chunks)

    assert [len(call) for call in calls] == [BATCH_SIZE, 1]


def test_non_chunks_are_left_alone(cache, monkeypatch):
    calls: list = []
    fake_llm(monkeypatch, calls)
    root = Resource(title="root")
    root.add_child(Resource(title="container", children=[Resource(title="page")]))

    EmbedTransformer(cache).transform([root])

    assert calls == []
    assert root.embedding is None


def test_chunks_of_finds_text_nodes():
    root = Resource(title="root")
    chunk = root.add_child(Resource(title="chunk", text="hello"))
    root.add_child(Resource(title="container", children=[Resource(title="page")]))

    assert chunks_of([root]) == [chunk]
