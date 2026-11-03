"""Judge: labels the trees the materializer produced.

The judge fills type, subtype, is_solution and the cut for every resource
— or drops it — and the spec's exceptions correct on top of its output.
There is no type/subtype → cut mapping here: how the judge decides is
Javier's business; this is a stub for his inference.
"""

from __future__ import annotations

import re

from dataclasses import dataclass

import ragtools.mime_types as mt

from rag_etl.core import Resource, is_text_node
from rag_etl.core.rules import Rule, apply
from rag_etl.transformers.base import Transformer

# The stub's broad regex defaults, applied where no extractor set a label
# and corrected by the exceptions. A cue is looked for in the node's own
# title first, then its ancestors', nearest first: a page of a deck is
# lecture material too, a page of a series is practice.

# Anything quiz-, exam- or exercise-shaped is practice; everything else is theory
PRACTICE_RE = re.compile(
    r"\bquiz|\bexams?\b|\bexamens?\b|\bexercices?\b|\bs[ée]ries?\b|\bhomework\b|\bquestions?\b"
    r"|\bsolutions?\b|corrig[ée]",
    re.IGNORECASE,
)

# The first rule matching a context title names the subtype; the order is meaningful
SUBTYPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("video_lecture", re.compile(r"video|recording", re.IGNORECASE)),
    ("quiz", re.compile(r"\bquiz", re.IGNORECASE)),
    ("exam", re.compile(r"\bexams?\b|\bexamens?\b", re.IGNORECASE)),
    ("solution", re.compile(r"\bsolutions?\b|corrig[ée]", re.IGNORECASE)),
    ("exercise", re.compile(r"\bexercices?\b|\bs[ée]ries?\b|\bhomework\b", re.IGNORECASE)),
    ("theory_slides", re.compile(r"theory[ _-]?slides", re.IGNORECASE)),
    ("lecture_slides", re.compile(r"slides?|\blecture|presentation", re.IGNORECASE)),
    ("theory", re.compile(r"theory", re.IGNORECASE)),
    ("thesis", re.compile(r"(?:\b|_)thesis", re.IGNORECASE)),
    ("guide", re.compile(r"\bguide|tutorial", re.IGNORECASE)),
    ("schedule", re.compile(r"schedule|planning", re.IGNORECASE)),
)

# The series/exercise number a practice title carries, e.g. 'Série 3' or 'Exercise 2.1'
NUMBER_RE = re.compile(
    r"(?:\bs[ée]ries?|\bexercices?|\bquiz|\bhomework)\s*[:#.]?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

DEFAULT_TYPE = "theory"      # a technical course's unlabelled residual
DEFAULT_SUBTYPE = "unknown"  # an honest flag for the titles no rule matched


@dataclass(frozen=True)
class JudgeTransformer(Transformer):
    """Labels the trees: the stub's regex defaults and cuts, corrected by the exceptions.

    The stub fills type/subtype/number from broad title regexes where no
    extractor set a label, applies the exceptions (the permanent behavior)
    and fills the cut a tree needs to produce chunks. is_solution is left
    to the extractors and the exercise cut; the real inference lands later.
    """

    exceptions: tuple[Rule, ...] = ()

    def transform(self, roots: list[Resource]) -> list[Resource]:
        for root in roots:
            for node in root.walk():
                self._default_cut(node)
                self._default_labels(node)
                apply(node, self.exceptions)
        return roots

    def _default_cut(self, node: Resource) -> None:
        """The stub's default cut: what a tree needs to produce chunks.

        Roots are never cut whole; readable leaves cut whole; containers
        of text cut per child. Everything else stays unlabelled and is
        therefore not selected.
        """
        if node.cut is not None or node.parent is None:
            return

        if node.is_leaf:
            if is_text_node(node):
                node.cut = "whole_document"
        elif any(child.mime_type == mt.MARKDOWN for child in node.children):
            node.cut = "per_child"

    def _default_labels(self, node: Resource) -> None:
        """The stub's regex defaults: fill the labels no extractor set."""
        context = [link.title or "" for link in reversed(node.lineage())]
        self._default_type(node, context)
        self._default_subtype(node, context)
        self._default_number(node, context)

    def _default_type(self, node: Resource, context: list[str]) -> None:
        """Practice on a quiz/exam/exercise cue in the titles, theory otherwise."""
        if node.type is None:
            node.type = "practice" if any(PRACTICE_RE.search(title) for title in context) else DEFAULT_TYPE

    def _default_subtype(self, node: Resource, context: list[str]) -> None:
        """The first rule matching a context title names the subtype."""
        if node.subtype is not None:
            return
        for title in context:
            for name, pattern in SUBTYPES:
                if pattern.search(title):
                    node.subtype = name
                    return
        node.subtype = DEFAULT_SUBTYPE

    def _default_number(self, node: Resource, context: list[str]) -> None:
        """A practice resource keeps the series/exercise number its titles carry."""
        if node.number is not None or node.type != "practice":
            return
        for title in context:
            found = NUMBER_RE.search(title)
            if found:
                node.number = found.group(1)
                return
