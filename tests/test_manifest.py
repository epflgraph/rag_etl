"""Unit tests for the manifest loader."""

import json

from datetime import UTC, datetime
from pathlib import Path

from rag_etl.core import Resource
from rag_etl.loaders import ManifestLoader

TIMESTAMP = datetime(2026, 10, 19, 15, 30, 45, tzinfo=UTC)


def tree() -> list[Resource]:
    pdf = Resource(title="Exercise set 3", url="https://moodle/3", type="practice", subtype="homework")
    pdf.add_child(Resource(title="Exercise 2a", type="practice", subtype="homework", number="2", subnumber="a", text="compute"))
    pdf.add_child(
        Resource(title="Exercise 2b", type="practice", subtype="homework", number="2", subnumber="b", is_solution=True, text="solution")
    )
    video = Resource(title="Lecture 5")
    video.add_child(Resource(title="intro", type="theory", subtype="lecture", text="hello"))
    return [pdf, video]


def write(directory: Path, roots: list[Resource], dry: bool = False) -> dict:
    ManifestLoader(directory, project="demo", timestamp=TIMESTAMP, dry=dry).load(roots)
    [path] = directory.glob("*.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_lands_where_told_with_the_timestamp_name(tmp_path):
    directory = tmp_path / "output"
    manifest = write(directory, tree())

    assert (directory / "20261019T153045_manifest.json").exists()
    header = manifest["header"]
    assert header["project"] == "demo"
    assert header["dry"] is False
    assert header["timestamp"] == TIMESTAMP.isoformat()
    assert header["index"] == "rag_demo_20261019_153045"
    assert header["alias"] == "rag_demo"
    assert isinstance(header["pipeline_version"], str)


def test_tree_is_metadata_only(tmp_path):
    manifest = write(tmp_path / "output", tree())

    def walk(node):
        assert "text" not in node
        assert "embedding" not in node
        assert node["content_hash"]
        for child in node["children"]:
            walk(child)

    [pdf, _] = manifest["tree"]
    walk(pdf)
    assert pdf["title"] == "Exercise set 3"
    assert pdf["chunk"] is False
    assert pdf["children"][0]["chunk"] is True
    assert pdf["children"][1]["is_solution"] is True


def test_summary_drills_down_per_source_type_subtype_number(tmp_path):
    manifest = write(tmp_path / "output", tree())
    summary = manifest["summary"]

    assert summary["sources"] == 2
    assert summary["chunks"] == 3
    assert summary["without_url"] == 1

    [pdf_group, video_group] = summary["counts"]
    assert pdf_group["source"] == "Exercise set 3"
    assert pdf_group["chunks"] == 2
    assert pdf_group["without_url"] == 0
    assert pdf_group["is_solution"] == 1

    [practice] = pdf_group["counts"]
    assert practice["type"] == "practice"
    [homework] = practice["counts"]
    assert homework["subtype"] == "homework"
    [number_two] = homework["counts"]
    assert number_two["number"] == "2"
    assert number_two["chunks"] == 2
    assert number_two["counts"] == []

    assert video_group["source"] == "Lecture 5"
    assert video_group["without_url"] == 1
    [number_none] = video_group["counts"][0]["counts"][0]["counts"]
    assert number_none["number"] is None
    assert number_none["chunks"] == 1


def test_dry_manifest_is_named_and_stamped(tmp_path):
    directory = tmp_path / "output"
    manifest = write(directory, tree(), dry=True)

    assert (directory / "20261019T153045_manifest_dry.json").exists()
    assert manifest["header"]["dry"] is True


def test_untyped_chunks_land_in_unknown_buckets(tmp_path):
    root = Resource(title="stray")
    root.add_child(Resource(title="chunk", text="body"))

    manifest = write(tmp_path / "output", [root])

    [group] = manifest["summary"]["counts"]
    assert group["source"] == "stray"
    [type_group] = group["counts"]
    assert type_group["type"] == "unknown"
