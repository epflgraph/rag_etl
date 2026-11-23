"""Unit tests for the fixed-sequence orchestrator."""

import json

from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_etl.core import Resource
from rag_etl.extractors.base import Extractor, SourceUnavailable
from rag_etl.pipeline import extract, run_pipeline, transform
from rag_etl.projects import Project


class FakeExtractor(Extractor):
    def extract(self) -> Resource:
        return Resource(title=f"tree of {type(self).__name__}", path=self.dir)


class BrokenExtractor(Extractor):
    def extract(self) -> Resource:
        raise SourceUnavailable("the source is down")


def make_project(**kwargs) -> Project:
    defaults = {
        "id": "fake",
        "sources": (FakeExtractor(dir=Path("/tmp/fake")),),
    }
    return Project(**{**defaults, **kwargs})


def test_extract_builds_one_tree_per_source(tmp_path):
    roots = extract(make_project(sources=(FakeExtractor(dir=tmp_path), FakeExtractor(dir=tmp_path))), tmp_path)
    assert [root.title for root in roots] == ["tree of FakeExtractor", "tree of FakeExtractor"]


def test_extract_resolves_relative_folders_under_the_project(tmp_path):
    project = make_project(sources=(FakeExtractor(dir=Path("moodle")), FakeExtractor(dir=Path("/abs"))))

    roots = extract(project, tmp_path / "FAKE")

    # Relative folders land under the project's folder; absolute ones are kept
    assert [root.path for root in roots] == [tmp_path / "FAKE" / "moodle", Path("/abs")]
    # The project itself is untouched: the folders are resolved for this run only
    assert project.sources[0].dir == Path("moodle")


def test_extract_fails_loud(tmp_path):
    with pytest.raises(SourceUnavailable):
        extract(make_project(sources=(FakeExtractor(dir=tmp_path), BrokenExtractor(dir=tmp_path))), tmp_path)


STEPS = {
    "Normalize": "normalize",
    "Images": "images",
    "Pdf": "pdf",
    "Video": "video",
    "Judge": "judge",
    "Cut": "cut",
    "Embed": "embed",
}


def stub_steps(monkeypatch) -> tuple[dict, list]:
    """Every transformer replaced by a stub recording how it was built and when it ran."""
    built: dict = {}
    ran: list = []

    def stub(name: str):
        class Stub:
            def __init__(self, **kwargs):
                built[name] = kwargs

            def transform(self, roots):
                ran.append(name)
                return roots

        Stub.name = name
        return Stub

    for prefix, name in STEPS.items():
        monkeypatch.setattr(f"rag_etl.pipeline.{prefix}Transformer", stub(name))
    return built, ran


def test_transform_runs_the_sequence(tmp_path, monkeypatch):
    _, ran = stub_steps(monkeypatch)

    transform([Resource(title="root")], make_project(), tmp_path)

    # Images before pdf and video: it reads only the sources' markdown
    assert ran == list(STEPS.values())


def test_transform_bypasses_only_the_refreshed_cache(tmp_path, monkeypatch):
    built, _ = stub_steps(monkeypatch)

    transform([Resource(title="root")], make_project(), tmp_path, refresh="pdf")

    # every step works in the project's folder; only the named one refreshes
    assert all(kwargs["project_dir"] == tmp_path for kwargs in built.values())
    assert built["pdf"]["refresh"] is True
    assert all(built[name]["refresh"] is False for name in ("images", "video", "cut", "embed"))


def test_run_pipeline_writes_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_PATH", str(tmp_path / "base"))
    monkeypatch.setattr("rag_etl.pipeline.extract", lambda project, project_dir: [Resource(title="root")])
    monkeypatch.setattr("rag_etl.pipeline.transform", lambda roots, project, project_dir, refresh=None: roots)
    monkeypatch.setattr(
        "rag_etl.pipeline.ElasticsearchLoader",
        lambda **kwargs: SimpleNamespace(load=lambda roots: None),
    )

    run_pipeline(make_project())

    [manifest_path] = (tmp_path / "base" / "fake" / "output").glob("*.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["header"]["project"] == "fake"
    assert manifest["header"]["index"].startswith("rag_fake_")
    assert manifest["summary"]["chunks"] == 0


def test_run_pipeline_marks_dry_manifests(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_PATH", str(tmp_path / "base"))
    monkeypatch.setattr("rag_etl.pipeline.extract", lambda project, project_dir: [Resource(title="root")])
    monkeypatch.setattr("rag_etl.pipeline.transform", lambda roots, project, project_dir, refresh=None: roots)

    run_pipeline(make_project(), dry=True)

    [manifest_path] = (tmp_path / "base" / "fake" / "output").glob("*_manifest_dry.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["header"]["dry"] is True
