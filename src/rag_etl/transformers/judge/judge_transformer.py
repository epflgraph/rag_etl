"""Judge: labels the trees the materializer produced.

The judge fills type, subtype, is_solution and the cut for every resource
— or drops it — and the spec's exceptions correct on top of its output.
There is no type/subtype → cut mapping here: how the judge decides is
Javier's business; this is a stub for his inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import ragtools.mime_types as mt

from rag_etl.core import Resource, is_text_node
from rag_etl.core.rules import Rule, apply
from rag_etl.transformers.base import Transformer


@dataclass(frozen=True)
class JudgeTransformer(Transformer):
    """Labels the trees: the stub's default cuts, corrected by the exceptions.

    The stub applies the exceptions (the permanent behavior) and fills the
    cut a tree needs to produce chunks; type/subtype/is_solution stay
    unlabelled until the real inference lands.
    """

    exceptions: tuple[Rule, ...] = ()

    def transform(self, roots: list[Resource]) -> list[Resource]:
        for root in roots:
            for node in root.walk():
                self._default_cut(node)
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
