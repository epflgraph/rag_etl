"""Unit tests for the unified RCP client: stubbed OpenAI, no network."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from ragtools.llm import AnswerTooLong, RCPClient, get_model


class Answer(BaseModel):
    text: str


class FakeGeneration:
    def update(self, **kwargs):
        pass


@contextmanager
def fake_observation(**kwargs):
    yield FakeGeneration()


class FakeCompletions:
    def __init__(self, content: str = " answer "):
        self.content = content

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content, reasoning=None, reasoning_content=None))],
            usage=None,
        )


class FakeOpenAI:
    def __init__(self, content: str = " answer "):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))
        self.embeddings = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0]) for _ in kwargs["input"]]))


class FakeStream:
    """A fake streaming answer: one content event per chunk, closeable mid-stream."""

    def __init__(self, chunks: list[str]):
        self.closed = False
        self.events = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=chunk))]) for chunk in chunks
        ]

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


class FakeStreamingOpenAI:
    def __init__(self, chunks: list[str]):
        self.stream = FakeStream(chunks)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: self.stream))


@pytest.fixture(autouse=True)
def fresh_underlying_client():
    """Each test patches OpenAI with a fresh fake: the cached client must not leak."""
    from ragtools.llm.rcp import openai_client

    openai_client.cache_clear()
    yield


@pytest.fixture
def stubbed(monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setattr("ragtools.llm.rcp.OpenAI", lambda **kwargs: fake)
    monkeypatch.setattr("ragtools.llm.rcp.get_client", lambda: SimpleNamespace(start_as_current_observation=fake_observation))
    return fake


def test_unknown_task_fails_loud():
    with pytest.raises(ValueError):
        get_model("does-not-exist")


def test_text_strips_and_returns(stubbed):
    client = RCPClient(model="fake-model", base_url="https://rcp", api_key="k")
    assert client.text("hello") == "answer"


def test_structured_parses_the_schema(stubbed):
    stubbed.chat.completions.content = '{"text": "hello"}'
    client = RCPClient(model="fake-model", base_url="https://rcp", api_key="k")
    assert client.structured("hi", Answer) == Answer(text="hello")


def test_embeddings_one_vector_per_text(stubbed):
    client = RCPClient(model="fake-embedding", base_url="https://rcp", api_key="k")
    assert client.embeddings(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]


def test_one_underlying_client_across_calls(monkeypatch):
    constructions = []
    fake = FakeOpenAI()
    monkeypatch.setattr("ragtools.llm.rcp.OpenAI", lambda **kwargs: constructions.append(1) or fake)
    monkeypatch.setattr("ragtools.llm.rcp.get_client", lambda: SimpleNamespace(start_as_current_observation=fake_observation))

    client = RCPClient(model="fake-model", base_url="https://rcp", api_key="k")
    client.text("a")
    client.text("b")

    assert len(constructions) == 1


def test_client_without_endpoint_fails_loud_at_use(monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setattr("ragtools.llm.rcp.OpenAI", lambda **kwargs: fake)
    monkeypatch.setattr("ragtools.llm.rcp.get_client", lambda: SimpleNamespace(start_as_current_observation=fake_observation))

    client = RCPClient(model="fake-model", base_url="", api_key="")
    with pytest.raises(ValueError):
        client.text("hello")


def test_vision_streams_and_joins_chunks(tmp_path, monkeypatch):
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    fake = FakeStreamingOpenAI([" Hel", "lo "])
    monkeypatch.setattr("ragtools.llm.rcp.OpenAI", lambda **kwargs: fake)
    monkeypatch.setattr("ragtools.llm.rcp.get_client", lambda: SimpleNamespace(start_as_current_observation=fake_observation))

    client = RCPClient(model="fake-model", base_url="https://rcp", api_key="k")
    assert client.vision("read", [image], max_chars=100) == "Hello"
    assert not fake.stream.closed


def test_vision_cuts_the_stream_past_max_chars(tmp_path, monkeypatch):
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    fake = FakeStreamingOpenAI(["x" * 60, "y" * 60])
    monkeypatch.setattr("ragtools.llm.rcp.OpenAI", lambda **kwargs: fake)
    monkeypatch.setattr("ragtools.llm.rcp.get_client", lambda: SimpleNamespace(start_as_current_observation=fake_observation))

    client = RCPClient(model="fake-model", base_url="https://rcp", api_key="k")
    with pytest.raises(AnswerTooLong):
        client.vision("read", [image], max_chars=100)
    assert fake.stream.closed
