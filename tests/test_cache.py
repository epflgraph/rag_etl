"""Unit tests for the content-hash disk cache: temp dirs, no network."""

import pytest

from rag_etl.cache import Cache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    directory = tmp_path / "cache"
    directory.mkdir()
    monkeypatch.setattr("rag_etl.config.CONFIG", {"CACHE_DIR": str(directory)}, raising=False)
    return directory


def test_missing_cache_dir_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("rag_etl.config.CONFIG", {"CACHE_DIR": str(tmp_path / "nowhere")}, raising=False)
    with pytest.raises(ValueError):
        Cache("materialize")


def test_file_roundtrip(cache_dir, tmp_path):
    cache = Cache("materialize")
    source = tmp_path / "out.md"
    source.write_text("hello", encoding="utf-8")

    cache.set("hash1", source)

    destination = tmp_path / "sub" / "out.md"
    assert cache.get("hash1", destination) is True
    assert destination.read_text(encoding="utf-8") == "hello"


def test_dir_roundtrip_copies_nested(cache_dir, tmp_path):
    cache = Cache("materialize")
    out = tmp_path / "pages"
    (out / "nested").mkdir(parents=True)
    (out / "page_1.md").write_text("one", encoding="utf-8")
    (out / "nested" / "page_2.md").write_text("two", encoding="utf-8")

    cache.set("hash1", out)

    destination = tmp_path / "other" / "pages"
    assert cache.get("hash1", destination) is True
    assert (destination / "page_1.md").read_text(encoding="utf-8") == "one"
    assert (destination / "nested" / "page_2.md").read_text(encoding="utf-8") == "two"


def test_miss_returns_false(cache_dir, tmp_path):
    cache = Cache("materialize")
    assert cache.get("unknown-hash", tmp_path / "x.md") is False


def test_refresh_bypasses_lookup(cache_dir, tmp_path):
    cache = Cache("materialize")
    source = tmp_path / "out.md"
    source.write_text("hello", encoding="utf-8")
    cache.set("hash1", source)

    refreshed = Cache("materialize", refresh=True)
    assert refreshed.get("hash1", tmp_path / "other.md") is False


def test_scopes_are_shared_by_transform_name(cache_dir, tmp_path):
    """Same content, same transform: one entry, whatever the project."""
    cache = Cache("materialize")
    source = tmp_path / "out.md"
    source.write_text("hello", encoding="utf-8")
    cache.set("hash1", source)

    other = Cache("materialize")
    assert other.get("hash1", tmp_path / "sub" / "out.md") is True
