"""A MOOC quiz (an Open edX problem): two leaves, written as Markdown into the export's markdown folder.

- the quiz without its solutions
- the quiz again, its answers marked and its explanation added
"""

import logging

from lxml.etree import _Element

import ragtools.mime_types as mt
from rag_etl.core import Resource
from rag_etl.extractors.mooc.export import Export
from ragtools.edx_quiz import parse_quiz, quiz_to_markdown
from ragtools.html import clean_text

logger = logging.getLogger(__name__)


def parse_quiz_item(export: Export, element: _Element, vertical_title: str) -> list[Resource]:
    """The quiz's statement and statement + solution leaves; none for a kind of problem not supported."""

    url_name = element.get("url_name", "")
    if not url_name:
        logger.warning("A problem without a url_name")
        return []

    block = export.load("problem", url_name)
    if block is None:
        return []

    quiz = parse_quiz(block)
    if quiz is None:
        return []

    export.markdown_dir.mkdir(parents=True, exist_ok=True)
    statement_path = export.markdown_dir / f"{url_name}.quiz.md"
    solution_path = export.markdown_dir / f"{url_name}.quiz_solutions.md"
    statement_path.write_text(quiz_to_markdown(quiz, include_solutions=False), encoding="utf-8")
    solution_path.write_text(quiz_to_markdown(quiz, include_solutions=True), encoding="utf-8")

    # The quiz is titled by its own name (the unit's is its parent's title
    # already), while its number may sit in either
    title = clean_text(block.get("display_name", ""))
    number = _number(f"{vertical_title} - {quiz.display_name}".strip(" -"))

    # Statement and statement + solution, the assignment/solution pair Tutor Bot expects.
    # Both have a solution: the export ships it with every quiz
    return [
        Resource(
            title=title,
            path=path,
            mime_type=mt.MARKDOWN,
            exercise_parts=parts,
            has_solution=True,
            number=number,
        )
        for path, parts in ((statement_path, "statement"), (solution_path, "statement + solution"))
    ]


def _number(title: str) -> str | None:
    """
    The first run of digits and dots in a title, wherever it sits ("1.3.3. Digital Images - ..." → "1.3.3").

    None for a title carrying no digits at all.
    """

    number = ""
    for character in title:
        if character.isdigit() or character == "." and number:
            number += character
        elif number:
            break

    return number.strip(".") or None
