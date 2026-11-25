"""Unit tests for embed: fake model, cache roundtrip. No network."""

from types import SimpleNamespace

import pytest

from rag_etl.core import Resource
from rag_etl.transformers.embed import BATCH_SIZE, REQUEST_CHAR_BUDGET, EmbedTransformer


@pytest.fixture
def embedder(tmp_path):
    return EmbedTransformer(project_dir=tmp_path)


def fake_llm(monkeypatch, calls: list):
    def embed_texts(texts):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("rag_etl.transformers.embed.get_model", lambda task: SimpleNamespace(embeddings=embed_texts))


def test_one_vector_per_chunk_in_order(embedder, monkeypatch):
    calls: list = []
    fake_llm(monkeypatch, calls)
    chunks = [Resource(title=f"chunk {i}", text=f"text {i}") for i in range(3)]

    embedder.transform(chunks)

    assert all(chunk.embedding == [1.0, 0.0] for chunk in chunks)


def test_second_run_reads_from_cache(embedder, monkeypatch):
    calls: list = []
    fake_llm(monkeypatch, calls)
    chunk = Resource(title="chunk", text="the same text")

    embedder.transform([chunk])
    embedder.transform([chunk])

    # The second run embeds nothing: same content, cached vector
    assert len(calls) == 1
    assert chunk.embedding == [1.0, 0.0]


def test_changed_chunks_embed_again(embedder, monkeypatch):
    calls: list = []
    fake_llm(monkeypatch, calls)
    embedder.transform([Resource(title="chunk", text="before")])
    embedder.transform([Resource(title="chunk", text="after")])

    assert len(calls) == 2


def test_calls_are_batched(embedder, monkeypatch):
    calls: list = []
    fake_llm(monkeypatch, calls)
    chunks = [Resource(title=f"chunk {i}", text=f"text {i}") for i in range(BATCH_SIZE + 1)]

    embedder.transform(chunks)

    assert [len(call) for call in calls] == [BATCH_SIZE, 1]


def test_batches_close_at_the_char_budget(embedder, monkeypatch):
    calls: list = []
    fake_llm(monkeypatch, calls)
    chunks = [Resource(title=f"chunk {i}", text="x" * 20_000) for i in range(10)]

    embedder.transform(chunks)

    assert [len(call) for call in calls] == [4, 4, 2]
    assert all(sum(len(text) for text in call) <= REQUEST_CHAR_BUDGET for call in calls)


def test_single_oversized_chunk_raises(embedder, monkeypatch):
    calls: list = []
    fake_llm(monkeypatch, calls)
    chunk = Resource(title="the thesis", text="x" * (REQUEST_CHAR_BUDGET + 1))

    with pytest.raises(ValueError, match="the thesis"):
        embedder.transform([chunk])

    assert calls == []


def test_non_chunks_are_left_alone(embedder, monkeypatch):
    calls: list = []
    fake_llm(monkeypatch, calls)
    root = Resource(title="root")
    container = root.add_child(title="container")
    container.add_child(title="page")

    embedder.transform([root])

    assert calls == []
    assert root.embedding is None
