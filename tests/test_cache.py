"""Unit tests for the content-hash disk cache: temp dirs, no network."""

import pytest

from rag_etl.cache import Cache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    directory = tmp_path / "cache"
    directory.mkdir()
    monkeypatch.setenv("CACHE_DIR", str(directory))
    return directory


def test_missing_cache_dir_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(ValueError):
        Cache("micro303", "materialize")


def test_file_roundtrip(cache_dir, tmp_path):
    cache = Cache("micro303", "materialize")
    source = tmp_path / "out.md"
    source.write_text("hello", encoding="utf-8")

    cache.set("hash1", source)

    destination = tmp_path / "sub" / "out.md"
    assert cache.get("hash1", destination) is True
    assert destination.read_text(encoding="utf-8") == "hello"


def test_entries_land_under_project_and_transform(cache_dir, tmp_path):
    cache = Cache("micro303", "materialize")
    source = tmp_path / "out.md"
    source.write_text("hello", encoding="utf-8")

    cache.set("hash1", source)

    assert (cache_dir / "micro303" / "materialize" / "hash1" / "out.md").is_file()


def test_dir_roundtrip_copies_nested(cache_dir, tmp_path):
    cache = Cache("micro303", "materialize")
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
    cache = Cache("micro303", "materialize")
    assert cache.get("unknown-hash", tmp_path / "x.md") is False


def test_refresh_bypasses_lookup(cache_dir, tmp_path):
    cache = Cache("micro303", "materialize")
    source = tmp_path / "out.md"
    source.write_text("hello", encoding="utf-8")
    cache.set("hash1", source)

    refreshed = Cache("micro303", "materialize", refresh=True)
    assert refreshed.get("hash1", tmp_path / "other.md") is False


def test_projects_are_isolated(cache_dir, tmp_path):
    """Same content, same transform: one entry per project."""
    cache = Cache("micro303", "materialize")
    source = tmp_path / "out.md"
    source.write_text("hello", encoding="utf-8")
    cache.set("hash1", source)

    other = Cache("cmi", "materialize")
    assert other.get("hash1", tmp_path / "sub" / "out.md") is False


def test_text_roundtrip(cache_dir):
    cache = Cache("micro303", "embed")
    cache.write("hash1", "embedding.json", "[0.1, 0.2]")

    assert cache.read("hash1", "embedding.json") == "[0.1, 0.2]"


def test_text_read_misses_and_respects_refresh(cache_dir):
    cache = Cache("micro303", "embed")
    assert cache.read("unknown-hash", "embedding.json") is None

    cache.write("hash1", "embedding.json", "[0.1]")
    refreshed = Cache("micro303", "embed", refresh=True)
    assert refreshed.read("hash1", "embedding.json") is None
