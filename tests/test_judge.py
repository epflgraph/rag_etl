"""Unit tests for the judge stub: default cuts, exceptions on top. No LLM."""

import ragtools.mime_types as mt
from rag_etl.core import Resource
from rag_etl.core.rules import Rule
from rag_etl.transformers.judge.judge_transformer import JudgeTransformer


def test_readable_leaf_cuts_whole():
    root = Resource(title="root")
    leaf = root.add_child(Resource(title="notes", mime_type=mt.MARKDOWN))

    JudgeTransformer().transform([root])

    assert leaf.cut == "whole_document"


def test_container_with_text_children_cuts_per_child():
    root = Resource(title="root")
    doc = root.add_child(Resource(title="doc", mime_type=mt.PDF))
    doc.add_child(Resource(title="doc > page 1", mime_type=mt.MARKDOWN))

    JudgeTransformer().transform([root])

    assert doc.cut == "per_child"


def test_roots_are_never_cut():
    root = Resource(title="root", mime_type=mt.MARKDOWN)

    JudgeTransformer().transform([root])

    assert root.cut is None


def test_unreadable_leaves_stay_unlabelled():
    root = Resource(title="root")
    caption = root.add_child(Resource(title="captions", mime_type=mt.SRT))

    JudgeTransformer().transform([root])

    assert caption.cut is None


def test_containers_without_text_children_stay_unlabelled():
    root = Resource(title="root")
    folder = root.add_child(Resource(title="folder"))
    folder.add_child(Resource(title="doc", mime_type=mt.PDF))

    JudgeTransformer().transform([root])

    assert folder.cut is None


def test_exceptions_correct_the_labels():
    rule = Rule(where='title="big notes"', cut="text")
    root = Resource(title="root")
    leaf = root.add_child(Resource(title="big notes", mime_type=mt.MARKDOWN))

    JudgeTransformer(exceptions=(rule,)).transform([root])

    assert leaf.cut == "text"


def test_titles_without_cues_get_the_defaults():
    root = Resource(title="root")
    leaf = root.add_child(Resource(title="group lists", mime_type=mt.MARKDOWN))

    JudgeTransformer().transform([root])

    assert leaf.type == "theory" and leaf.subtype == "unknown"


def test_ancestor_titles_label_the_leaves():
    root = Resource(title="root")
    doc = root.add_child(Resource(title="MICRO-303 2026 Lecture 5 CVD.pdf", mime_type=mt.PDF))
    page = doc.add_child(Resource(title="doc > page 1", mime_type=mt.MARKDOWN))

    JudgeTransformer().transform([root])

    assert doc.type == "theory" and doc.subtype == "lecture_slides"
    assert page.type == "theory" and page.subtype == "lecture_slides"


def test_exam_questions_become_practice():
    root = Resource(title="root")
    leaf = root.add_child(Resource(title="list of questions for the oral exams", mime_type=mt.MARKDOWN))

    JudgeTransformer().transform([root])

    assert leaf.type == "practice" and leaf.subtype == "exam"


def test_series_titles_carry_their_number():
    root = Resource(title="root")
    leaf = root.add_child(Resource(title="Série 3", mime_type=mt.MARKDOWN))

    JudgeTransformer().transform([root])

    assert leaf.type == "practice" and leaf.subtype == "exercise" and leaf.number == "3"


def test_video_titles_become_video_lectures():
    root = Resource(title="root")
    leaf = root.add_child(Resource(title="Chapter 4_video 1.pdf", mime_type=mt.MARKDOWN))

    JudgeTransformer().transform([root])

    assert leaf.type == "theory" and leaf.subtype == "video_lecture"


def test_theory_slides_folder_labels_its_children():
    root = Resource(title="[THEORY_SLIDES]Complementary Slides")
    doc = root.add_child(Resource(title="2025 Introduction Micro332.pdf", mime_type=mt.PDF))
    page = doc.add_child(Resource(title="doc > page 1", mime_type=mt.MARKDOWN))

    JudgeTransformer().transform([root])

    assert page.type == "theory" and page.subtype == "theory_slides"


def test_word_boundaries_keep_examples_theory():
    root = Resource(title="root")
    leaf = root.add_child(Resource(title="MEMS examples", mime_type=mt.MARKDOWN))

    JudgeTransformer().transform([root])

    assert leaf.type == "theory" and leaf.subtype == "unknown"


def test_extractor_labels_are_never_overwritten():
    root = Resource(title="root")
    leaf = root.add_child(
        Resource(title="a quiz", mime_type=mt.MARKDOWN, type="practice", subtype="mooc_quiz", week=2)
    )

    JudgeTransformer().transform([root])

    assert leaf.type == "practice" and leaf.subtype == "mooc_quiz" and leaf.week == 2


def test_exceptions_win_over_the_defaults():
    rule = Rule(where='title="recap"', subtype="recap")
    root = Resource(title="root")
    leaf = root.add_child(Resource(title="recap", mime_type=mt.MARKDOWN))

    JudgeTransformer(exceptions=(rule,)).transform([root])

    assert leaf.subtype == "recap"
