"""Unit tests for the shipped index settings and the index naming."""

from datetime import datetime

from rag_etl.loaders.elasticsearch import alias_name, index_name, index_settings

FIELDS = {
    "id",
    "content",
    "embedding",
    "lineage",
    "url",
    "type",
    "subtype",
    "number",
    "subnumber",
    "week",
    "is_solution",
    "from",
    "until",
}


def test_settings_carry_the_frozen_schema():
    body = index_settings("en")
    properties = body["mappings"]["properties"]

    assert set(properties) == FIELDS
    assert properties["embedding"]["dims"] == 4096
    assert set(body["settings"]["index"]["analysis"]["analyzer"]) == {"base_en", "base_fr"}


def test_analyzer_follows_the_language():
    assert index_settings("fr")["mappings"]["properties"]["content"]["analyzer"] == "base_fr"
    assert index_settings(None)["mappings"]["properties"]["content"]["analyzer"] == "base_en"


def test_naming():
    when = datetime(2026, 10, 19, 15, 30, 45)

    assert alias_name("micro303") == "rag_micro303"
    assert index_name("micro303", when) == "rag_micro303_2026_10_19"
