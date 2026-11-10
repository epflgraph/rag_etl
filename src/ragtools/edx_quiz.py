"""Open edX problems (single- and multiple-choice quizzes) → Markdown.

`parse_quiz` reads a problem's XML into a `Quiz`; `quiz_to_markdown`
renders it, with or without its answers marked and its explanation.
Problems of any other kind are not supported and parse to None.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lxml.etree import _Element

from ragtools.html import clean_text, escape_markdown, normalize_markdown

logger = logging.getLogger(__name__)

# Tags read as blocks of the statement inside a response node
STATEMENT_TAGS: set[str] = {
    "label",
    "description",
    "p",
    "div",
    "strong",
    "br",
}


@dataclass
class QuizOption:
    text_md: str
    is_correct: bool


@dataclass
class Quiz:
    display_name: str
    quiz_type: str  # "single" | "multi"
    statement_md: str
    options: list[QuizOption]
    general_feedback_md: str | None = None


def parse_quiz(root: _Element) -> Quiz | None:
    """Read a problem's root element into a Quiz; None for an unsupported kind of problem."""

    display_name = clean_text(root.get("display_name", "")) or "Quiz"

    mcr = root.find(".//multiplechoiceresponse")
    cr = root.find(".//choiceresponse")

    if mcr is not None and cr is not None:
        logger.info("Unsupported quiz (multiple response types found)")
        return None

    if mcr is not None:
        return _single_choice(display_name, mcr)

    if cr is not None:
        return _checkbox_choice(display_name, cr)

    logger.info("Unsupported quiz (no recognized response node)")
    return None


def quiz_to_markdown(quiz: Quiz, include_solutions: bool) -> str:
    """Render a quiz: its statement and options, and with the solutions its answers marked and its explanation."""

    lines: list[str] = []

    lines.append(f"# {quiz.display_name}")
    lines.append("")

    if quiz.statement_md.strip():
        lines.append(quiz.statement_md)
        lines.append("")

    if quiz.quiz_type == "single":
        lines.append("## Answers (single choice)")
    else:
        lines.append("## Answers (multiple choice)")
    lines.append("")

    for opt in quiz.options:
        if quiz.quiz_type == "single":
            bullet = "(x)" if (include_solutions and opt.is_correct) else "( )"
        else:
            bullet = "- [x]" if (include_solutions and opt.is_correct) else "- [ ]"

        opt_text = opt.text_md.strip()
        if not opt_text:
            lines.append(f"{bullet}")
        elif "\n" not in opt_text:
            lines.append(f"{bullet} {opt_text}")
        else:
            first, *rest = opt_text.splitlines()
            lines.append(f"{bullet} {first}")
            for r in rest:
                lines.append(f"  {r}")

    if include_solutions and quiz.general_feedback_md and quiz.general_feedback_md.strip():
        lines.append("")
        lines.append("## Explanation")
        lines.append("")
        lines.append(quiz.general_feedback_md.strip())

    return normalize_markdown("\n".join(lines)) + "\n"


def _single_choice(display_name: str, mcr: _Element) -> Quiz | None:
    choicegroup = mcr.find("./choicegroup")
    if choicegroup is None:
        choicegroup = mcr.find(".//choicegroup")
    if choicegroup is None:
        logger.info("Unsupported multiple choice (no choicegroup)")
        return None

    return Quiz(
        display_name=display_name,
        quiz_type="single",
        statement_md=_statement_md(mcr, stop_tags={"choicegroup", "solution", "demandhint"}),
        options=_options(choicegroup),
        general_feedback_md=_general_solution_md(mcr),
    )


def _checkbox_choice(display_name: str, cr: _Element) -> Quiz | None:
    checkboxgroup = cr.find("./checkboxgroup")
    if checkboxgroup is None:
        checkboxgroup = cr.find(".//checkboxgroup")
    if checkboxgroup is None:
        logger.info("Unsupported checkbox quiz (no checkboxgroup)")
        return None

    return Quiz(
        display_name=display_name,
        quiz_type="multi",
        statement_md=_statement_md(cr, stop_tags={"checkboxgroup", "solution", "demandhint"}),
        options=_options(checkboxgroup),
        general_feedback_md=_general_solution_md(cr),
    )


def _statement_md(parent: _Element, stop_tags: set[str]) -> str:
    """The statement: the blocks before the options."""

    chunks: list[str] = []

    for child in parent:
        if child.tag in stop_tags:
            break
        if child.tag in STATEMENT_TAGS:
            md = _xml_to_markdown(child)
            if md.strip():
                chunks.append(md)

    return normalize_markdown("\n\n".join(chunks))


def _options(group: _Element) -> list[QuizOption]:
    options: list[QuizOption] = []

    for choice in group.findall("./choice"):
        is_correct = choice.get("correct", "").lower() == "true"
        options.append(QuizOption(text_md=_choice_text(choice), is_correct=is_correct))

    return options


def _choice_text(choice: _Element) -> str:
    """An option's text, its hints left out."""

    parts: list[str] = []
    if choice.text and clean_text(choice.text):
        parts.append(escape_markdown(clean_text(choice.text)))

    for child in choice:
        if child.tag == "choicehint":
            continue
        md = _xml_to_markdown(child)
        if md.strip():
            parts.append(md)

    return normalize_markdown("\n\n".join(parts))


def _general_solution_md(parent: _Element) -> str | None:
    """The explanation the problem gives with its solution, if any."""

    sol = parent.find(".//solution")
    if sol is None:
        return None

    # <solution><div class="detailed-solution"><p>...</p>...</div></solution>
    # An lxml element's truth is whether it has children, not whether it
    # was found, so the fallbacks test for None
    container = sol.find("./div[@class='detailed-solution']")
    if container is None:
        container = sol.find("./div")
    if container is None:
        container = sol

    chunks: list[str] = []

    for p in container.findall("./p"):
        md = _xml_to_markdown(p)
        if md.strip():
            chunks.append(md)

    solution_text = "\n\n".join(chunks)
    solution_text = solution_text.replace("Explanation", "")
    out = normalize_markdown(solution_text)
    return out or None


def _xml_to_markdown(elem: _Element | None) -> str:
    if elem is None:
        return ""

    tag = elem.tag

    if tag in {"label", "description", "div", "p"}:
        return _children_to_md(elem)

    elif tag == "br":
        return "\n"

    elif tag == "strong":
        inner = normalize_markdown(_children_to_md(elem))
        return f"**{inner}**" if inner else ""

    # Any other tag: its children speak for it
    return _children_to_md(elem)


def _children_to_md(elem: _Element) -> str:
    parts: list[str] = []

    if elem.text and clean_text(elem.text):
        parts.append(escape_markdown(clean_text(elem.text)))

    for child in elem:
        child_md = _xml_to_markdown(child)
        if child_md:
            if child.tag in {"div", "p"}:
                if parts and not parts[-1].endswith("\n"):
                    parts.append("\n")
                parts.append(child_md)
                parts.append("\n")
            else:
                parts.append(child_md)

        if child.tail and clean_text(child.tail):
            parts.append(escape_markdown(clean_text(child.tail)))

    return normalize_markdown("".join(parts))
