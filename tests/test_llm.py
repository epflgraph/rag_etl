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
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
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
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.stream


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


def test_a_prompt_alone_returns_the_answers_text(stubbed):
    client = RCPClient(model="fake-model", base_url="https://rcp", api_key="k")

    assert client.generate("hello", system="be brief") == "answer"

    [request] = stubbed.chat.completions.requests
    assert request["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]
    assert "response_format" not in request


def test_a_schema_returns_the_answer_parsed(stubbed):
    stubbed.chat.completions.content = '{"text": "hello"}'
    client = RCPClient(model="fake-model", base_url="https://rcp", api_key="k")

    assert client.generate("hi", schema=Answer) == Answer(text="hello")
    assert stubbed.chat.completions.requests[0]["response_format"]["json_schema"]["name"] == "Answer"


def test_images_and_a_schema_combine(tmp_path, stubbed):
    """The options are independent: an image beside the prompt, the answer parsed."""
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    stubbed.chat.completions.content = '{"text": "a table"}'
    client = RCPClient(model="fake-model", base_url="https://rcp", api_key="k")

    assert client.generate("what is it?", images=[image], schema=Answer, detail="high") == Answer(text="a table")

    [request] = stubbed.chat.completions.requests
    text, picture = request["messages"][0]["content"]
    assert text == {"type": "text", "text": "what is it?"}
    assert picture["image_url"]["url"].startswith("data:image/png;base64,") and picture["image_url"]["detail"] == "high"
    assert "response_format" in request


def test_endpoint_and_key_come_from_the_environment(stubbed, monkeypatch):
    seen = {}
    monkeypatch.setattr("ragtools.llm.rcp.OpenAI", lambda **kwargs: seen.update(kwargs) or stubbed)
    monkeypatch.setenv("RCP_BASE_URL", "https://from-env")
    monkeypatch.setenv("RCP_API_KEY", "env-key")

    RCPClient(model="fake-model").generate("hello")

    assert seen["base_url"] == "https://from-env" and seen["api_key"] == "env-key"


def test_a_missing_variable_fails_loud_at_use_by_its_name(stubbed, monkeypatch):
    monkeypatch.delenv("RCP_BASE_URL", raising=False)
    monkeypatch.setenv("RCP_API_KEY", "env-key")

    client = RCPClient(model="fake-model")  # building it reads nothing
    with pytest.raises(KeyError, match="RCP_BASE_URL"):
        client.generate("hello")


def test_embeddings_one_vector_per_text(stubbed):
    client = RCPClient(model="fake-embedding", base_url="https://rcp", api_key="k")
    assert client.embeddings(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]


def test_one_underlying_client_across_calls(monkeypatch):
    constructions = []
    fake = FakeOpenAI()
    monkeypatch.setattr("ragtools.llm.rcp.OpenAI", lambda **kwargs: constructions.append(1) or fake)
    monkeypatch.setattr("ragtools.llm.rcp.get_client", lambda: SimpleNamespace(start_as_current_observation=fake_observation))

    client = RCPClient(model="fake-model", base_url="https://rcp", api_key="k")
    client.generate("a")
    client.generate("b")

    assert len(constructions) == 1


def streaming(monkeypatch, chunks: list[str]) -> FakeStreamingOpenAI:
    fake = FakeStreamingOpenAI(chunks)
    monkeypatch.setattr("ragtools.llm.rcp.OpenAI", lambda **kwargs: fake)
    monkeypatch.setattr("ragtools.llm.rcp.get_client", lambda: SimpleNamespace(start_as_current_observation=fake_observation))
    return fake


def test_a_length_limit_streams_and_joins_the_chunks(tmp_path, monkeypatch):
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    fake = streaming(monkeypatch, [" Hel", "lo "])

    client = RCPClient(model="fake-model", base_url="https://rcp", api_key="k")
    assert client.generate("read", images=[image], max_chars=100) == "Hello"
    assert fake.requests[0]["stream"] is True and not fake.stream.closed


def test_the_stream_is_cut_past_the_length_limit(monkeypatch):
    fake = streaming(monkeypatch, ["x" * 60, "y" * 60])

    client = RCPClient(model="fake-model", base_url="https://rcp", api_key="k")
    with pytest.raises(AnswerTooLong):
        client.generate("read", max_chars=100)
    assert fake.stream.closed


def test_a_length_limit_and_a_schema_combine(monkeypatch):
    """A streamed answer is asked in the schema and parsed too."""
    fake = streaming(monkeypatch, ['{"text": ', '"hello"}'])

    client = RCPClient(model="fake-model", base_url="https://rcp", api_key="k")
    assert client.generate("hi", schema=Answer, max_chars=100) == Answer(text="hello")
    assert "response_format" in fake.requests[0]
