"""Unit tests for the fixed-sequence orchestrator."""

import json

from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_etl.core import Resource
from rag_etl.extractors.base_extractor import Extractor, SourceUnavailable
from rag_etl.pipeline import extract, run_pipeline, transform
from rag_etl.projects.spec import ProjectSpec


class FakeExtractor(Extractor):
    def extract(self) -> Resource:
        return Resource(title=f"tree of {type(self).__name__}")


class BrokenExtractor(Extractor):
    def extract(self) -> Resource:
        raise SourceUnavailable("the source is down")


def make_spec(**kwargs) -> ProjectSpec:
    defaults = {
        "id": "fake",
        "sources": (FakeExtractor(dir=Path("/tmp/fake")),),
    }
    return ProjectSpec(**{**defaults, **kwargs})


def test_extract_builds_one_tree_per_source(tmp_path):
    roots = extract((FakeExtractor(dir=tmp_path), FakeExtractor(dir=tmp_path)))
    assert [root.title for root in roots] == ["tree of FakeExtractor", "tree of FakeExtractor"]


def test_extract_fails_loud(tmp_path):
    with pytest.raises(SourceUnavailable):
        extract((FakeExtractor(dir=tmp_path), BrokenExtractor(dir=tmp_path)))


def test_transform_runs_the_sequence(tmp_path, monkeypatch):
    ran = []

    class FakeStep:
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
    transform(roots, make_spec())

    assert ran == ["FakeStep"] * 5


def test_transform_bypasses_only_the_refreshed_cache(tmp_path, monkeypatch):
    caches = []

    class RecordingCache:
        def __init__(self, transform, refresh=False):
            caches.append((transform, refresh))

    def stub(**kwargs):
        return SimpleNamespace(transform=lambda roots: roots)

    monkeypatch.setattr("rag_etl.pipeline.Cache", RecordingCache)
    monkeypatch.setattr("rag_etl.pipeline.NormalizeTransformer", stub)
    monkeypatch.setattr("rag_etl.pipeline.MaterializeTransformer", stub)
    monkeypatch.setattr("rag_etl.pipeline.JudgeTransformer", stub)
    monkeypatch.setattr("rag_etl.pipeline.CutTransformer", stub)
    monkeypatch.setattr("rag_etl.pipeline.EmbedTransformer", stub)

    transform([Resource(title="root")], make_spec(), refresh="materialize")

    assert ("normalize", False) in caches
    assert ("materialize", True) in caches


def test_run_pipeline_writes_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "rag_etl.config.CONFIG",
        {"CACHE_DIR": str(tmp_path / "cache"), "BASE_PATH": str(tmp_path / "base")},
        raising=False,
    )
    monkeypatch.setattr("rag_etl.pipeline.extract", lambda sources: [Resource(title="root")])
    monkeypatch.setattr("rag_etl.pipeline.transform", lambda roots, spec, refresh=None: roots)
    monkeypatch.setattr(
        "rag_etl.pipeline.ElasticsearchLoader",
        lambda **kwargs: SimpleNamespace(load=lambda roots: None),
    )

    run_pipeline(make_spec())

    [manifest_path] = (tmp_path / "base" / "fake" / "output").glob("*.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["header"]["project"] == "fake"
    assert manifest["header"]["index"].startswith("rag_fake_")
    assert manifest["summary"]["sources"] == 1


def test_run_pipeline_marks_dry_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "rag_etl.config.CONFIG",
        {"CACHE_DIR": str(tmp_path / "cache"), "BASE_PATH": str(tmp_path / "base")},
        raising=False,
    )
    monkeypatch.setattr("rag_etl.pipeline.extract", lambda sources: [Resource(title="root")])
    monkeypatch.setattr("rag_etl.pipeline.transform", lambda roots, spec, refresh=None: roots)

    run_pipeline(make_spec(), dry=True)

    [manifest_path] = (tmp_path / "base" / "fake" / "output").glob("*_manifest_dry.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["header"]["dry"] is True
