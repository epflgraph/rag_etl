"""Unit tests for embed: fake client, cache roundtrip. No network."""

from types import SimpleNamespace

import pytest

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.transformers.embed.embed import BATCH_SIZE, chunks_of, embed


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr("rag_etl.config.CONFIG", {"CACHE_DIR": str(tmp_path / "cache")}, raising=False)
    (tmp_path / "cache").mkdir()
    return Cache("embed")


def fake_llm(calls: list):
    def embed_texts(texts, *, model):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    return SimpleNamespace(embeddings=embed_texts)


def test_one_vector_per_chunk_in_order(cache):
    calls: list = []
    chunks = [Resource(title=f"chunk {i}", text=f"text {i}") for i in range(3)]

    pairs = embed(chunks, fake_llm(calls), "fake", cache)

    assert [pair[0] for pair in pairs] == chunks
    assert all(pair[1] == [1.0, 0.0] for pair in pairs)


def test_second_run_reads_from_cache(cache):
    calls: list = []
    chunks = [Resource(title="chunk", text="the same text")]

    embed(chunks, fake_llm(calls), "fake", cache)
    embed(chunks, fake_llm(calls), "fake", cache)

    # The second run embeds nothing: same content, cached vector
    assert len(calls) == 1


def test_changed_chunks_embed_again(cache):
    calls: list = []
    embed([Resource(title="chunk", text="before")], fake_llm(calls), "fake", cache)
    embed([Resource(title="chunk", text="after")], fake_llm(calls), "fake", cache)

    assert len(calls) == 2


def test_calls_are_batched(cache):
    calls: list = []
    chunks = [Resource(title=f"chunk {i}", text=f"text {i}") for i in range(BATCH_SIZE + 1)]

    embed(chunks, fake_llm(calls), "fake", cache)

    assert [len(call) for call in calls] == [BATCH_SIZE, 1]


def test_chunks_of_finds_text_nodes():
    root = Resource(title="root")
    chunk = root.add_child(Resource(title="chunk", text="hello"))
    root.add_child(Resource(title="container", children=[Resource(title="page")]))

    assert chunks_of([root]) == [chunk]
