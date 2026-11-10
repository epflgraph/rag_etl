from __future__ import annotations

import logging

from lxml.etree import _Element

import ragtools.mime_types as mt
from rag_etl.core import Resource
from rag_etl.extractors.mooc.utils import Export, extract_number
from ragtools.edx_quiz import parse_quiz, quiz_to_markdown
from ragtools.html import clean_text

logger = logging.getLogger(__name__)


class QuizParser:
    """
    Parse an Open edX problem into two leaves, its markdown written beside the export:
    - the quiz without its solutions
    - the quiz again, its answers marked and its explanation added
    """

    def parse(
        self,
        export: Export,
        elem_vertical: _Element,
        vertical_display_name: str,
        week: int | None = None,
    ) -> list[Resource]:
        url_name = elem_vertical.get("url_name", "")
        if not url_name:
            logger.warning("QuizParser: missing url_name in elem_vertical")
            return []

        root = export.load("problem", url_name)
        if root is None:
            return []

        quiz = parse_quiz(root)
        if quiz is None:
            # Unsupported quiz type
            return []

        out_dir = export.dir / "quiz"
        out_dir.mkdir(parents=True, exist_ok=True)

        quiz_md_path = out_dir / f"{url_name}.quiz.md"
        quiz_sol_md_path = out_dir / f"{url_name}.quiz_solutions.md"

        quiz_md_path.write_text(quiz_to_markdown(quiz, include_solutions=False), encoding="utf-8")
        quiz_sol_md_path.write_text(quiz_to_markdown(quiz, include_solutions=True), encoding="utf-8")

        # The quiz is titled by its own name (the unit's is its parent's title
        # already), while its number may sit in either
        quiz_title = clean_text(root.get("display_name", ""))
        number = extract_number(f"{vertical_display_name} - {quiz.display_name}".strip(" -"))

        # Statement and statement + solution, the assignment/solution pair Tutor Bot expects.
        # Both have a solution: the export ships it with every quiz
        return [
            Resource(
                title=quiz_title,
                path=path,
                mime_type=mt.guess_mime_type(str(path)),
                exercise_parts=parts,
                has_solution=True,
                number=number,
                week=week,
            )
            for path, parts in ((quiz_md_path, "statement"), (quiz_sol_md_path, "statement + solution"))
        ]
