"""Unit tests for the fixed-sequence orchestrator."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_etl.core import Resource
from rag_etl.core.models import ModelConfig, ModelConfigs
from rag_etl.extractors.base_extractor import Extractor, SourceUnavailable
from rag_etl.pipeline import embedding_model_of, extract, ocr_model_of, transform
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
        "models": ModelConfigs(ocr=ModelConfig(model="fake-ocr")),
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

    roots = [Resource(title="root")]
    transform(roots, make_spec())

    assert ran == ["FakeStep", "FakeStep"]


def test_ocr_model_of_fails_loud():
    spec = make_spec(models=ModelConfigs())
    with pytest.raises(ValueError):
        ocr_model_of(spec)


def test_embedding_model_of_fails_loud():
    spec = make_spec(models=ModelConfigs())
    with pytest.raises(ValueError):
        embedding_model_of(spec)


def test_transform_bypasses_only_the_refreshed_cache(tmp_path, monkeypatch):
    caches = []

    class RecordingCache:
        def __init__(self, transform, refresh=False):
            caches.append((transform, refresh))

    monkeypatch.setattr("rag_etl.pipeline.Cache", RecordingCache)
    monkeypatch.setattr("rag_etl.pipeline.NormalizeTransformer", lambda **kwargs: SimpleNamespace(transform=lambda roots: roots))
    monkeypatch.setattr("rag_etl.pipeline.MaterializeTransformer", lambda **kwargs: SimpleNamespace(transform=lambda roots: roots))

    transform([Resource(title="root")], make_spec(), refresh="materialize")

    assert ("normalize", False) in caches
    assert ("materialize", True) in caches
