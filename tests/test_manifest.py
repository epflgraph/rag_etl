"""Unit tests for the manifest loader."""

import json

from datetime import UTC, datetime
from pathlib import Path

from rag_etl.core import Resource
from rag_etl.loaders import ManifestLoader

TIMESTAMP = datetime(2026, 10, 19, 15, 30, 45, tzinfo=UTC)


def tree() -> list[Resource]:
    pdf = Resource(title="Exercise set 3", url="https://moodle/3", type="practice", subtype="homework")
    pdf.add_child(
        title="Exercise 2a",
        number="2",
        subnumber="a",
        exercise_parts="statement",
        week=3,
        from_="2026-10-06",
        until="2026-10-13",
        text="compute",
    )
    pdf.add_child(
        title="Exercise 2b",
        number="2",
        subnumber="b",
        exercise_parts="solution",
        has_solution=True,
        week=3,
        from_="2026-10-06",
        until="2026-10-13",
        text="solution",
    )
    video = Resource(title="Lecture 5")
    video.add_child(title="intro", type="theory", subtype="lecture", week=5, text="hello")
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
    assert pdf["children"][1]["exercise_parts"] == "solution"
    assert pdf["children"][1]["has_solution"] is True


def test_summary_drills_down_per_source_type_subtype_number_subnumber_parts(tmp_path):
    manifest = write(tmp_path / "output", tree())
    summary = manifest["summary"]

    assert summary["chunks"] == 3
    assert summary["with_url"] == 2
    assert summary["with_week"] == 3
    assert summary["with_from"] == 2
    # url, week and dates vary across sources, so no value is stated at the top
    assert "url" not in summary
    assert "week" not in summary
    assert "from" not in summary

    [pdf_group, video_group] = summary["counts"]
    assert pdf_group["source"] == "Exercise set 3"
    assert pdf_group["chunks"] == 2
    # the whole group carries the pdf's url and the same release dates
    assert pdf_group["with_url"] == 2
    assert pdf_group["url"] == "https://moodle/3"
    assert pdf_group["week"] == 3
    assert pdf_group["from"] == "2026-10-06"
    assert pdf_group["until"] == "2026-10-13"

    [practice] = pdf_group["counts"]
    assert practice["type"] == "practice"
    [homework] = practice["counts"]
    assert homework["subtype"] == "homework"
    [number_two] = homework["counts"]
    assert number_two["number"] == "2"
    assert number_two["chunks"] == 2
    [sub_a, sub_b] = number_two["counts"]
    assert sub_a["subnumber"] == "a"
    assert sub_b["subnumber"] == "b"
    [statement] = sub_a["counts"]
    assert statement["parts"] == "statement"
    # the last level carries no counts key
    assert "counts" not in statement
    [solution] = sub_b["counts"]
    assert solution["parts"] == "solution"
    assert "counts" not in solution

    assert video_group["source"] == "Lecture 5"
    assert video_group["chunks"] == 1
    assert video_group["with_url"] == 0
    assert video_group["week"] == 5
    assert video_group["with_from"] == 0
    # unlabelled buckets stay visible, all the way down
    [theory] = video_group["counts"]
    assert theory["type"] == "theory"
    [lecture] = theory["counts"]
    [number_none] = lecture["counts"]
    assert number_none["number"] is None
    [subnumber_none] = number_none["counts"]
    [parts_none] = subnumber_none["counts"]
    assert parts_none["parts"] is None
    assert parts_none["chunks"] == 1
    assert "counts" not in parts_none


def test_summary_states_values_only_when_unanimous(tmp_path):
    root = Resource(title="Set", url="https://moodle/x", week=3)
    root.add_child(title="one", from_="2026-10-06", text="a")
    root.add_child(title="two", from_="2026-10-07", text="b")
    root.add_child(title="three", text="c")

    manifest = write(tmp_path / "output", [root])
    [group] = manifest["summary"]["counts"]

    assert group["chunks"] == 3
    # unanimous values are stated
    assert group["with_url"] == 3
    assert group["url"] == "https://moodle/x"
    assert group["with_week"] == 3
    assert group["week"] == 3
    # disagreeing values are counted but not stated
    assert group["with_from"] == 2
    assert "from" not in group
    # nobody set one: the count says so, the value stays absent
    assert group["with_until"] == 0
    assert "until" not in group


def test_unlabelled_chunks_land_in_null_buckets(tmp_path):
    root = Resource(title="stray")
    root.add_child(title="chunk", text="body")

    manifest = write(tmp_path / "output", [root])

    [group] = manifest["summary"]["counts"]
    assert group["source"] == "stray"
    [type_group] = group["counts"]
    assert type_group["type"] is None


def test_dry_manifest_is_named_and_stamped(tmp_path):
    directory = tmp_path / "output"
    manifest = write(directory, tree(), dry=True)

    assert (directory / "20261019T153045_manifest_dry.json").exists()
    assert manifest["header"]["dry"] is True
