import logging

from rag_etl.core.resource import Resource
from rag_etl.core.rules import Rule, apply, parse_where


def test_parse_where():
    assert parse_where('title="a b" AND mime_type=application/pdf') == [
        ("title", "a b"),
        ("mime_type", "application/pdf"),
    ]
    assert parse_where("type=theory") == [("type", "theory")]


def test_glob_matching():
    rule = Rule(where='title="*[[]HOMEWORK*]*"')
    assert rule.matches(Resource(title="[HOMEWORK_2] Series 2"))
    assert not rule.matches(Resource(title="Series 2"))


def test_filename_derived_from_path():
    rule = Rule(where='filename="*.pdf" AND path="*week*"')
    assert rule.matches(Resource(path="/data/week1/ex.pdf"))
    assert not rule.matches(Resource(path="/data/exam/ex.pdf"))


def test_bool_and_is_leaf_attributes():
    assert Rule(where="is_solution=true").matches(Resource(is_solution=True))
    assert not Rule(where="is_solution=true").matches(Resource())
    assert Rule(where="is_leaf=true").matches(Resource(text="c"))
    assert not Rule(where="is_leaf=true").matches(Resource(children=[Resource()]))


def test_unknown_field_never_matches(caplog):
    with caplog.at_level(logging.WARNING):
        assert not Rule(where="bogus=x").matches(Resource(title="t"))
    assert "bogus" in caplog.text


def test_apply_corrects_labels():
    r = Resource(title="[EXAM_2024] old exam", type="theory", cut="text")
    apply(r, [Rule(where='title="*[[]EXAM*]*"', type="exam", subtype="previous_year_exam")])
    assert r.type == "exam"
    assert r.subtype == "previous_year_exam"
    assert r.cut == "text"


def test_most_specific_wins():
    where = 'title="*[[]EXAM*]*"'
    specific = 'title="*[[]EXAM*]*" AND mime_type="application/pdf"'
    for exceptions in (
        [Rule(where=where, type="exam"), Rule(where=specific, type="practice")],
        [Rule(where=specific, type="practice"), Rule(where=where, type="exam")],
    ):
        r = Resource(title="[EXAM_2024] old exam", mime_type="application/pdf")
        apply(r, exceptions)
        assert r.type == "practice"


def test_equal_specificity_first_declared_wins(caplog):
    r = Resource(title="[EXAM_2024] old exam")
    with caplog.at_level(logging.WARNING):
        apply(
            r,
            [
                Rule(where='title="*[[]EXAM*]*"', type="exam"),
                Rule(where='title="*[[]EXAM*]*"', type="practice"),
            ],
        )
    assert r.type == "exam"
    assert "Ambiguous" in caplog.text


def test_no_match_leaves_resource_untouched():
    r = Resource(title="random talk", type="theory")
    apply(r, [Rule(where='title="*[[]HOMEWORK*]*"', type="practice")])
    assert r.type == "theory"
