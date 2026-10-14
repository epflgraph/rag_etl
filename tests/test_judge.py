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
