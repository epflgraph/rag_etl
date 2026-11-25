"""Unit tests for the content-hash disk cache: temp dirs, no network."""

import pytest

from rag_etl.cache import INDEX_FILE, Cache


@pytest.fixture
def cache(tmp_path):
    # Not created: the cache makes its folders on first write
    return Cache(tmp_path / "project" / "cache" / "pdf-ocr-1")


def refreshed(cache: Cache) -> Cache:
    return Cache(cache.cache_path, refresh=True)


def test_a_value_is_read_back_as_written(cache):
    cache.write("key1", "page_1.md", "hello")

    assert cache.read("key1", "page_1.md") == "hello"
    # An entry is a folder named by its key, its values files in it
    assert (cache.cache_path / "key1" / "page_1.md").read_text(encoding="utf-8") == "hello"


def test_an_empty_value_is_a_hit(cache):
    # A blank page's transcript is cached like any other: not read again
    cache.write("key1", "page_1.md", "")
    assert cache.read("key1", "page_1.md") == ""


def test_a_read_misses_and_respects_refresh(cache):
    assert cache.read("unknown", "page_1.md") is None

    cache.write("key1", "page_1.md", "hello")
    assert cache.read("key1", "other.md") is None
    assert refreshed(cache).read("key1", "page_1.md") is None


def test_an_index_brings_back_a_documents_parts_in_order(cache):
    # Parts and the document's index sit side by side in the one cache
    cache.write("page-b", "page_2.md", "two")
    cache.write("page-a", "page_1.md", "one")
    cache.write_index("document", [("page_1.md", "page-a"), ("page_2.md", "page-b")])

    assert cache.read_index("document") == [("page_1.md", "one"), ("page_2.md", "two")]
    assert (cache.cache_path / "document" / INDEX_FILE).is_file()


def test_a_document_without_parts_has_an_empty_index(cache):
    cache.write_index("document", [])
    assert cache.read_index("document") == []


def test_an_index_misses_when_absent_malformed_incomplete_or_refreshed(cache):
    assert cache.read_index("unknown") is None

    cache.write("document", INDEX_FILE, "not json")
    assert cache.read_index("document") is None

    # A part the index names is gone: the document is redone, its other parts still cached
    cache.write("page-a", "page_1.md", "one")
    cache.write_index("document", [("page_1.md", "page-a"), ("page_2.md", "page-b")])
    assert cache.read_index("document") is None

    cache.write("page-b", "page_2.md", "two")
    assert cache.read_index("document") is not None
    assert refreshed(cache).read_index("document") is None
