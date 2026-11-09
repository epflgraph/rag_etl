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


def test_extract_builds_one_tree_per_source(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_PATH", str(tmp_path))
    roots = extract(make_project(sources=(FakeExtractor(dir=tmp_path), FakeExtractor(dir=tmp_path))))
    assert [root.title for root in roots] == ["tree of FakeExtractor", "tree of FakeExtractor"]


def test_extract_resolves_relative_folders_under_the_project(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_PATH", str(tmp_path))
    project = make_project(dir=Path("FAKE"), sources=(FakeExtractor(dir=Path("moodle")), FakeExtractor(dir=Path("/abs"))))

    roots = extract(project)

    # Relative folders land under $BASE_PATH/<project dir>; absolute ones are kept
    assert [root.path for root in roots] == [tmp_path / "FAKE" / "moodle", Path("/abs")]
    # The project itself is untouched: the folders are resolved for this run only
    assert project.sources[0].dir == Path("moodle")


def test_extract_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_PATH", str(tmp_path))
    with pytest.raises(SourceUnavailable):
        extract(make_project(sources=(FakeExtractor(dir=tmp_path), BrokenExtractor(dir=tmp_path))))


def test_transform_runs_the_sequence(tmp_path, monkeypatch):
    ran = []

    class FakeStep:
        name = "fake"

        def __init__(self, *args, **kwargs):
            pass

        def transform(self, roots):
            ran.append(type(self).__name__)
            return roots

    monkeypatch.setattr("rag_etl.pipeline.NormalizeTransformer", FakeStep)
    monkeypatch.setattr("rag_etl.pipeline.MaterializeTransformer", FakeStep)
    monkeypatch.setattr("rag_etl.pipeline.JudgeTransformer", FakeStep)
    monkeypatch.setattr("rag_etl.pipeline.CutTransformer", FakeStep)
    monkeypatch.setattr("rag_etl.pipeline.EmbedTransformer", FakeStep)

    roots = [Resource(title="root")]
    transform(roots, make_project())

    assert ran == ["FakeStep"] * 5


def test_transform_bypasses_only_the_refreshed_cache(tmp_path, monkeypatch):
    caches = []

    class RecordingCache:
        def __init__(self, project, transform, refresh=False):
            caches.append((transform, refresh))

    def stub(name: str):
        class Stub:
            def __init__(self, **kwargs):
                pass

            def transform(self, roots):
                return roots

        Stub.name = name
        return Stub

    monkeypatch.setattr("rag_etl.pipeline.Cache", RecordingCache)
    monkeypatch.setattr("rag_etl.pipeline.NormalizeTransformer", stub("normalize"))
    monkeypatch.setattr("rag_etl.pipeline.MaterializeTransformer", stub("materialize"))
    monkeypatch.setattr("rag_etl.pipeline.JudgeTransformer", stub("judge"))
    monkeypatch.setattr("rag_etl.pipeline.CutTransformer", stub("cut"))
    monkeypatch.setattr("rag_etl.pipeline.EmbedTransformer", stub("embed"))

    transform([Resource(title="root")], make_project(), refresh="materialize")

    assert ("normalize", False) in caches
    assert ("materialize", True) in caches


def test_run_pipeline_writes_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("BASE_PATH", str(tmp_path / "base"))
    monkeypatch.setattr("rag_etl.pipeline.extract", lambda project: [Resource(title="root")])
    monkeypatch.setattr("rag_etl.pipeline.transform", lambda roots, project, refresh=None: roots)
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
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("BASE_PATH", str(tmp_path / "base"))
    monkeypatch.setattr("rag_etl.pipeline.extract", lambda project: [Resource(title="root")])
    monkeypatch.setattr("rag_etl.pipeline.transform", lambda roots, project, refresh=None: roots)

    run_pipeline(make_project(), dry=True)

    [manifest_path] = (tmp_path / "base" / "fake" / "output").glob("*_manifest_dry.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["header"]["dry"] is True
