"""Unit tests for the content-hash disk cache: temp dirs, no network."""

import pytest

from rag_etl.cache import Cache


@pytest.fixture
def cache_dir(tmp_path):
    # Not created: the cache makes its folders on first write
    return tmp_path / "project" / "cache"


def test_file_roundtrip(cache_dir, tmp_path):
    cache = Cache(cache_dir / "pdf-ocr-1")
    source = tmp_path / "out.md"
    source.write_text("hello", encoding="utf-8")

    cache.set("hash1", source)

    destination = tmp_path / "sub" / "out.md"
    assert cache.get("hash1", destination) is True
    assert destination.read_text(encoding="utf-8") == "hello"


def test_entries_land_in_the_cache_folder(cache_dir, tmp_path):
    cache = Cache(cache_dir / "pdf-ocr-1")
    source = tmp_path / "out.md"
    source.write_text("hello", encoding="utf-8")

    cache.set("hash1", source)

    assert (cache_dir / "pdf-ocr-1" / "hash1" / "out.md").is_file()


def test_dir_roundtrip_copies_nested(cache_dir, tmp_path):
    cache = Cache(cache_dir / "pdf-ocr-1")
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
    cache = Cache(cache_dir / "pdf-ocr-1")
    assert cache.get("unknown-hash", tmp_path / "x.md") is False


def test_refresh_bypasses_lookup(cache_dir, tmp_path):
    cache = Cache(cache_dir / "pdf-ocr-1")
    source = tmp_path / "out.md"
    source.write_text("hello", encoding="utf-8")
    cache.set("hash1", source)

    refreshed = Cache(cache_dir / "pdf-ocr-1", refresh=True)
    assert refreshed.get("hash1", tmp_path / "other.md") is False


def test_text_roundtrip(cache_dir):
    cache = Cache(cache_dir / "embed")
    cache.write("hash1", "embedding.json", "[0.1, 0.2]")

    assert cache.read("hash1", "embedding.json") == "[0.1, 0.2]"


def test_text_read_misses_and_respects_refresh(cache_dir):
    cache = Cache(cache_dir / "embed")
    assert cache.read("unknown-hash", "embedding.json") is None

    cache.write("hash1", "embedding.json", "[0.1]")
    refreshed = Cache(cache_dir / "embed", refresh=True)
    assert refreshed.read("hash1", "embedding.json") is None
