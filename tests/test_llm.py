"""Unit tests for the unified RCP client: stubbed OpenAI, no network."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from ragtools.llm import RCPClient


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


@pytest.fixture
def stubbed(monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setattr("ragtools.llm.rcp.OpenAI", lambda **kwargs: fake)
    monkeypatch.setattr("ragtools.llm.rcp.get_client", lambda: SimpleNamespace(start_as_current_observation=fake_observation))
    return fake


def test_unknown_model_fails_loud_before_any_request(stubbed):
    client = RCPClient(base_url="https://rcp", api_key="k")
    with pytest.raises(ValueError):
        client.text("hello", model="does-not-exist")


def test_text_strips_and_returns(stubbed):
    client = RCPClient(base_url="https://rcp", api_key="k")
    assert client.text("hello", model="Qwen/Qwen3.6-35B-A3B", enable_thinking=False) == "answer"


def test_structured_parses_the_schema(stubbed):
    stubbed.chat.completions.content = '{"text": "hello"}'
    client = RCPClient(base_url="https://rcp", api_key="k")
    assert client.structured("hi", Answer, model="Qwen/Qwen3.6-35B-A3B", enable_thinking=False) == Answer(text="hello")


def test_embeddings_one_vector_per_text(stubbed):
    client = RCPClient(base_url="https://rcp", api_key="k")
    assert client.embeddings(["a", "b"], model="fake-embedding") == [[1.0, 2.0], [1.0, 2.0]]
