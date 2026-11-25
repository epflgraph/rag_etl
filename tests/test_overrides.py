import logging

import pytest

from rag_etl.core.overrides import Override, apply_overrides, parse_where
from rag_etl.core.resource import Resource


def test_parse_where():
    assert parse_where('title="a b" AND mime_type=application/pdf') == [
        ("title", "a b"),
        ("mime_type", "application/pdf"),
    ]
    assert parse_where("type=theory") == [("type", "theory")]


def test_glob_matching():
    override = Override('title="*[[]HOMEWORK*]*"')
    assert override.matches(Resource(title="[HOMEWORK_2] Series 2"))
    assert not override.matches(Resource(title="Series 2"))


def test_url_matching():
    override = Override('url="https://mediaspace.epfl.ch/*"')
    assert override.matches(Resource(title="Lecture 1", url="https://mediaspace.epfl.ch/media/t/0_abc"))
    assert not override.matches(Resource(title="Lecture 1", url="https://moodle.epfl.ch/x"))
    assert not override.matches(Resource(title="no url"))


def test_filename_derived_from_path():
    override = Override('filename="*.pdf" AND path="*week*"')
    assert override.matches(Resource(title="ex", path="/data/week1/ex.pdf"))
    assert not override.matches(Resource(title="ex", path="/data/exam/ex.pdf"))


def test_exercise_and_is_leaf_attributes():
    assert Override("has_solution=true").matches(Resource(title="t", has_solution=True))
    assert not Override("has_solution=true").matches(Resource(title="t"))
    assert Override('exercise_parts="solution"').matches(Resource(title="t", exercise_parts="solution"))
    assert not Override('exercise_parts="solution"').matches(Resource(title="t", exercise_parts="statement"))
    assert Override("is_leaf=true").matches(Resource(title="t", text="c"))
    assert not Override("is_leaf=true").matches(Resource(title="t", children=[Resource(title="c")]))


def test_metadata_matches_under_its_document_name():
    assert Override("from=2026-*").matches(Resource(title="t", from_="2026-09-01T08:00:00"))


def test_unknown_where_field_never_matches(caplog):
    with caplog.at_level(logging.WARNING):
        assert not Override("bogus=x").matches(Resource(title="t"))
    assert "bogus" in caplog.text


def test_unknown_metadata_field_fails_at_construction():
    with pytest.raises(ValueError, match="subtipe"):
        Override('title="x"', subtipe="exam")


def test_apply_corrects_metadata():
    r = Resource(title="[EXAM_2024] old exam", type="theory", cut="text")
    apply_overrides(r, [Override('title="*[[]EXAM*]*"', type="exam", subtype="previous_year_exam")])
    assert r.type == "exam"
    assert r.subtype == "previous_year_exam"
    assert r.cut == "text"


def test_most_specific_wins():
    where = 'title="*[[]EXAM*]*"'
    specific = 'title="*[[]EXAM*]*" AND mime_type="application/pdf"'
    for overrides in (
        [Override(where, type="exam"), Override(specific, type="practice")],
        [Override(specific, type="practice"), Override(where, type="exam")],
    ):
        r = Resource(title="[EXAM_2024] old exam", mime_type="application/pdf")
        apply_overrides(r, overrides)
        assert r.type == "practice"


def test_equal_specificity_first_declared_wins(caplog):
    r = Resource(title="[EXAM_2024] old exam")
    with caplog.at_level(logging.WARNING):
        apply_overrides(
            r,
            [
                Override('title="*[[]EXAM*]*"', type="exam"),
                Override('title="*[[]EXAM*]*"', type="practice"),
            ],
        )
    assert r.type == "exam"
    assert "Ambiguous" in caplog.text


def test_no_match_leaves_resource_untouched():
    r = Resource(title="random talk", type="theory")
    assert apply_overrides(r, [Override('title="*[[]HOMEWORK*]*"', type="practice")]) is False
    assert r.type == "theory"


def test_excluding_override_wins_and_sets_nothing():
    r = Resource(title="Week 1 - Raw HTML", type="theory")
    overrides = [Override('title="*Raw HTML"', exclude=True), Override('title="Week 1*"', type="practice")]

    assert apply_overrides(r, overrides) is True
    assert r.type == "theory"


def test_excluding_override_takes_no_labels():
    with pytest.raises(ValueError, match="sets nothing else"):
        Override('title="*Raw HTML"', exclude=True, type="practice")
