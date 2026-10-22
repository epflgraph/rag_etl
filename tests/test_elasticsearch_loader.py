"""Unit tests for the Elasticsearch loader: fake client, no network."""

from datetime import UTC, datetime

import pytest

from elasticsearch import NotFoundError

from rag_etl.core import Resource
from rag_etl.loaders.elasticsearch import ElasticsearchLoader

TIMESTAMP = datetime(2026, 10, 19, 15, 30, 45, tzinfo=UTC)


class FakeIndices:
    def __init__(self, aliases: dict[str, set[str]]):
        self.aliases = aliases
        self.calls: list[tuple] = []

    def create(self, index, settings=None, mappings=None):
        self.calls.append(("create", index, settings, mappings))

    def refresh(self, index):
        self.calls.append(("refresh", index))

    def get_alias(self, name):
        self.calls.append(("get_alias", name))
        if name not in self.aliases:
            raise NotFoundError("missing", None, None)
        return {index: {"aliases": {name: {}}} for index in self.aliases[name]}

    def update_aliases(self, actions):
        self.calls.append(("update_aliases", actions))


class FakeClient:
    def __init__(self, aliases: dict[str, set[str]] | None = None):
        self.indices = FakeIndices(aliases or {})


def loader(client: FakeClient, language: str | None = "en") -> ElasticsearchLoader:
    return ElasticsearchLoader(project="demo", timestamp=TIMESTAMP, language=language, client=client)


def bulk_into(monkeypatch, sink: list) -> None:
    def record(client, actions):
        sink.extend(actions)

    monkeypatch.setattr("rag_etl.loaders.elasticsearch.bulk", record)


def tree() -> tuple[list[Resource], tuple[Resource, Resource, Resource]]:
    pdf = Resource(title="Exercise set 3", url="https://moodle/3", type="practice", subtype="homework")
    a = pdf.add_child(
        Resource(title="Exercise 2a", type="practice", subtype="homework", number="2", subnumber="a", text="compute")
    )
    b = pdf.add_child(
        Resource(
            title="Exercise 2b", type="practice", subtype="homework", number="2", subnumber="b",
            is_solution=True, text="solution",
        )
    )
    video = Resource(title="Lecture 5")
    intro = video.add_child(Resource(title="intro", type="theory", subtype="lecture", text="hello"))
    for chunk in (a, b, intro):
        chunk.embedding = [1.0]
    return [pdf, video], (a, b, intro)


def test_documents_are_built_from_chunks(monkeypatch):
    roots, (a, _, _) = tree()
    sink: list = []
    bulk_into(monkeypatch, sink)

    loader(FakeClient()).load(roots)

    [compute, solution, hello] = sink
    assert compute["_source"]["content"] == "compute"
    assert compute["_source"]["embedding"] == [1.0]
    assert compute["_id"].endswith(a.content_hash)
    assert compute["_index"] == "rag_demo_20261019_153045"

    assert compute["_source"]["url"] == "https://moodle/3"
    assert compute["_source"]["lineage"] == ["Exercise set 3", "Exercise 2a"]
    assert compute["_source"]["type"] == "practice"
    assert compute["_source"]["number"] == "2"
    assert compute["_source"]["subnumber"] == "a"
    assert compute["_source"]["is_solution"] is False
    assert "from" not in compute["_source"]
    assert "week" not in compute["_source"]

    assert solution["_source"]["is_solution"] is True

    assert hello["_source"]["url"] == "<redacted>"
    assert hello["_source"]["lineage"] == ["Lecture 5", "intro"]


def test_first_run_creates_refreshes_and_adds_the_alias(monkeypatch):
    roots, _ = tree()
    sink: list = []
    bulk_into(monkeypatch, sink)
    client = FakeClient()

    loader(client).load(roots)

    kinds = [call[0] for call in client.indices.calls]
    assert kinds == ["create", "refresh", "get_alias", "update_aliases"]

    [create] = [call for call in client.indices.calls if call[0] == "create"]
    assert create[1] == "rag_demo_20261019_153045"
    assert create[3]["properties"]["content"]["analyzer"] == "base_en"

    [update] = [call for call in client.indices.calls if call[0] == "update_aliases"]
    assert update[1] == [{"add": {"index": "rag_demo_20261019_153045", "alias": "rag_demo"}}]


def test_same_day_runs_get_distinct_indices(monkeypatch):
    roots, _ = tree()
    sink: list = []
    bulk_into(monkeypatch, sink)
    client = FakeClient()

    loader(client).load(roots)
    later = ElasticsearchLoader(
        project="demo", timestamp=TIMESTAMP.replace(second=46), language="en", client=client
    )
    later.load(roots)

    creates = [call[1] for call in client.indices.calls if call[0] == "create"]
    assert creates == ["rag_demo_20261019_153045", "rag_demo_20261019_153046"]


def test_alias_shift_removes_the_previous_index(monkeypatch):
    roots, _ = tree()
    sink: list = []
    bulk_into(monkeypatch, sink)
    client = FakeClient(aliases={"rag_demo": {"rag_demo_20261018_091500"}})

    loader(client).load(roots)

    [update] = [call for call in client.indices.calls if call[0] == "update_aliases"]
    assert update[1] == [
        {"remove": {"index": "rag_demo_20261018_091500", "alias": "rag_demo"}},
        {"add": {"index": "rag_demo_20261019_153045", "alias": "rag_demo"}},
    ]


def test_french_project_gets_the_french_analyzer(monkeypatch):
    roots, _ = tree()
    sink: list = []
    bulk_into(monkeypatch, sink)
    client = FakeClient()

    loader(client, language="fr").load(roots)

    [create] = [call for call in client.indices.calls if call[0] == "create"]
    assert create[3]["properties"]["content"]["analyzer"] == "base_fr"


def test_duplicate_content_is_kept_with_distinct_ids(monkeypatch):
    root = Resource(title="root")
    root.add_child(Resource(title="first", text="same", embedding=[1.0]))
    root.add_child(Resource(title="second", text="same", embedding=[1.0]))
    sink: list = []
    bulk_into(monkeypatch, sink)

    loader(FakeClient()).load([root])

    assert len(sink) == 2
    assert sink[0]["_id"] != sink[1]["_id"]
    assert sink[0]["_source"]["lineage"] == ["root", "first"]
    assert sink[1]["_source"]["lineage"] == ["root", "second"]


def test_unembedded_chunk_fails_loud(monkeypatch):
    root = Resource(title="root")
    root.add_child(Resource(title="chunk", text="body"))
    sink: list = []
    bulk_into(monkeypatch, sink)

    with pytest.raises(ValueError):
        loader(FakeClient()).load([root])
