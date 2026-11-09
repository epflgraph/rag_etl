"""Unit tests for the judge stub: default cuts, overrides on top. No LLM."""

import ragtools.mime_types as mt
from rag_etl.core import Override, Resource
from rag_etl.transformers.judge import JudgeTransformer


def test_readable_leaf_cuts_whole():
    root = Resource(title="root")
    leaf = root.add_child(title="notes", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert leaf.cut == "whole_document"


def test_container_with_text_children_cuts_per_child():
    root = Resource(title="root")
    doc = root.add_child(title="doc", mime_type=mt.PDF)
    doc.add_child(title="page 1", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert doc.cut == "per_child"


def test_roots_are_never_cut():
    root = Resource(title="root", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert root.cut is None


def test_unreadable_leaves_stay_unlabelled():
    root = Resource(title="root")
    caption = root.add_child(title="captions", mime_type=mt.SRT)

    JudgeTransformer().transform([root])

    assert caption.cut is None


def test_containers_without_text_children_stay_unlabelled():
    root = Resource(title="root")
    folder = root.add_child(title="folder")
    folder.add_child(title="doc", mime_type=mt.PDF)

    JudgeTransformer().transform([root])

    assert folder.cut is None


def test_overrides_correct_the_labels():
    override = Override('title="big notes"', cut="text")
    root = Resource(title="root")
    leaf = root.add_child(title="big notes", mime_type=mt.MARKDOWN)

    JudgeTransformer(overrides=(override,)).transform([root])

    assert leaf.cut == "text"


def test_titles_without_cues_get_the_defaults():
    root = Resource(title="root")
    leaf = root.add_child(title="group lists", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert leaf.type == "theory" and leaf.subtype == "unknown"


def test_ancestor_titles_label_the_leaves():
    root = Resource(title="root")
    doc = root.add_child(title="MICRO-303 2026 Lecture 5 CVD.pdf", mime_type=mt.PDF)
    page = doc.add_child(title="page 1", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert doc.type == "theory" and doc.subtype == "lecture_slides"
    assert page.type == "theory" and page.subtype == "lecture_slides"


def test_exam_questions_become_practice():
    root = Resource(title="root")
    leaf = root.add_child(title="list of questions for the oral exams", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert leaf.type == "practice" and leaf.subtype == "exam"


def test_series_titles_carry_their_number():
    root = Resource(title="root")
    leaf = root.add_child(title="Série 3", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert leaf.type == "practice" and leaf.subtype == "exercise" and leaf.number == "3"


def test_video_titles_become_video_lectures():
    root = Resource(title="root")
    leaf = root.add_child(title="Chapter 4_video 1.pdf", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert leaf.type == "theory" and leaf.subtype == "video_lecture"


def test_theory_slides_folder_labels_its_children():
    root = Resource(title="[THEORY_SLIDES]Complementary Slides")
    doc = root.add_child(title="2025 Introduction Micro332.pdf", mime_type=mt.PDF)
    page = doc.add_child(title="page 1", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert page.type == "theory" and page.subtype == "theory_slides"


def test_word_boundaries_keep_examples_theory():
    root = Resource(title="root")
    leaf = root.add_child(title="MEMS examples", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert leaf.type == "theory" and leaf.subtype == "unknown"


def test_extractor_labels_are_never_overwritten():
    root = Resource(title="root")
    leaf = root.add_child(title="a quiz", mime_type=mt.MARKDOWN, type="practice", subtype="mooc_quiz", week=2)

    JudgeTransformer().transform([root])

    assert leaf.type == "practice" and leaf.subtype == "mooc_quiz" and leaf.week == 2


def test_children_inherit_the_parent_context():
    root = Resource(title="root")
    doc = root.add_child(
        title="MICRO-303 2026 Lecture 5 CVD.pdf", mime_type=mt.PDF,
        week=5, from_="2026-09-07T08:00:00", until="2027-01-31T20:00:00",
    )
    page = doc.add_child(title="page 1", mime_type=mt.MARKDOWN)
    own = doc.add_child(title="page 2", mime_type=mt.MARKDOWN, from_="2026-10-01T08:00:00", week=6)

    JudgeTransformer().transform([root])

    assert page.from_ == "2026-09-07T08:00:00"
    assert page.until == "2027-01-31T20:00:00"
    assert page.week == 5
    assert own.from_ == "2026-10-01T08:00:00" and own.week == 6


def test_children_inherit_the_parents_labels():
    root = Resource(title="root")
    section = root.add_child(title="Série 3", mime_type=mt.PDF)
    doc = section.add_child(title="exercises.pdf", mime_type=mt.PDF)
    page = doc.add_child(title="page 1", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert section.type == "practice" and section.subtype == "exercise" and section.number == "3"
    assert doc.type == "practice" and doc.subtype == "exercise" and doc.number == "3"
    assert page.type == "practice" and page.subtype == "exercise" and page.number == "3"


def test_a_childs_own_cues_beat_the_inherited_ones():
    root = Resource(title="root")
    section = root.add_child(title="Lecture slides CVD", mime_type=mt.PDF)
    doc = section.add_child(title="Quiz 1.pdf", mime_type=mt.PDF)
    page = doc.add_child(title="page 1", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert section.subtype == "lecture_slides"
    assert doc.type == "practice" and doc.subtype == "quiz"
    assert page.type == "practice" and page.subtype == "quiz"


def test_overrides_win_over_the_defaults():
    override = Override('title="recap"', subtype="recap")
    root = Resource(title="root")
    leaf = root.add_child(title="recap", mime_type=mt.MARKDOWN)

    JudgeTransformer(overrides=(override,)).transform([root])

    assert leaf.subtype == "recap"
