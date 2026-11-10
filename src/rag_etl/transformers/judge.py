"""Judge: labels the trees the materializer produced.

The judge fills type, subtype and the cut for every resource
— or drops it — and the project's overrides correct on top of its output.
There is no type/subtype → cut mapping here: how the judge decides is
Javier's business; this is a stub for his inference.
"""

from __future__ import annotations

import re

from dataclasses import dataclass

import ragtools.mime_types as mt

from rag_etl.core import METADATA_FIELDS, Override, Resource, apply_overrides
from rag_etl.transformers.base import Transformer

# Chunk-level facts, derived from the chunks themselves: never inherited
CHUNK_LEVEL_FIELDS = ("exercise_parts", "has_solution")

# The stub's broad regex defaults, applied where no extractor set a label
# and corrected by the overrides. A node's own title speaks first; what
# it leaves unset the lineage inherits; the fallbacks come last.

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
    """Labels the trees: the stub's regex defaults and cuts, corrected by the overrides.

    The stub works in three passes: what each node itself says (its cut
    and the labels its own title carries), what the lineage adds (every
    metadata field a node lacks it inherits from its parent, then the
    overrides correct), and the fallbacks that leave nothing empty.
    Exercise parts stay with the extractors and the exercise cut; the real
    inference lands later.
    """

    name = "judge"

    overrides: tuple[Override, ...] = ()

    def transform(self, roots: list[Resource]) -> list[Resource]:
        excluded: list[Resource] = []
        for root in roots:
            for node in root.walk():
                self._default_cut(node)
                self._own_labels(node)
            for node in root.walk():
                self._inherit(node)
                if apply_overrides(node, self.overrides):
                    excluded.append(node)
            for node in root.walk():
                self._fallback(node)

        # Dropped once the walks are over, so no tree changes while it is walked.
        # A container the exclusion leaves empty, with no file of its own, holds
        # nothing anymore and goes with it
        for node in excluded:
            parent = node.parent
            node.detach()
            while parent is not None and not parent.children and parent.path is None:
                emptied, parent = parent, parent.parent
                emptied.detach()

        # A root has no parent to leave: an excluded one leaves the run
        return [root for root in roots if root not in excluded]

    def _default_cut(self, node: Resource) -> None:
        """The stub's default cut: what a tree needs to produce chunks.

        Roots are never cut whole; readable leaves cut whole; containers
        of text cut per child. Everything else stays unlabelled and is
        therefore not selected.
        """
        if node.cut is not None or node.parent is None:
            return

        if node.is_leaf:
            if mt.is_text(node.mime_type):
                node.cut = "whole_document"
        elif any(child.mime_type == mt.MARKDOWN for child in node.children):
            node.cut = "per_child"

    def _own_labels(self, node: Resource) -> None:
        """What the node's own title says: the type, subtype and number its cues carry."""
        title = node.title
        if node.type is None and PRACTICE_RE.search(title):
            node.type = "practice"
        if node.subtype is None:
            for name, pattern in SUBTYPES:
                if pattern.search(title):
                    node.subtype = name
                    break
        if node.number is None and (found := NUMBER_RE.search(title)):
            node.number = found.group(1)

    def _inherit(self, node: Resource) -> None:
        """The metadata a node lacks, it inherits from its parent.

        The walk is top-down, so a parent's labels are final before its
        children are visited, and a node's own labels always beat the
        inherited ones. cut is not inherited — an instruction for
        splitting the node itself, a leaf with a container-cut would
        never become a chunk — and neither are the chunk-level facts,
        derived from the chunks themselves.
        """
        if node.parent is None:
            return
        for name in METADATA_FIELDS:
            if name not in CHUNK_LEVEL_FIELDS and getattr(node, name) is None:
                setattr(node, name, getattr(node.parent, name))

    def _fallback(self, node: Resource) -> None:
        """No metadata is left empty: what nobody determined is defaulted.

        The fallbacks run after inheritance, so a default never travels
        down the tree ahead of a child's own cues.
        """
        if node.type is None:
            node.type = DEFAULT_TYPE
        if node.subtype is None:
            node.subtype = DEFAULT_SUBTYPE
