from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from lxml.etree import _Element

import rag_etl.utils.mime_types as mt
from rag_etl.resources.mooc_resource import MOOCResource
from rag_etl.extractors.mooc.utils import (
    load_root_elem_from_mooc_xml,
    clean_text,
    normalize_markdown,
    escape_markdown,
)


logger = logging.getLogger(__name__)


# Tags we treat as blocks inside response nodes
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
class QuizData:
    display_name: str
    quiz_type: str  # "single" | "multi"
    statement_md: str
    options: list[QuizOption]
    general_feedback_md: str | None = None


class QuizParser:
    """
    Parse Open edX problem XML quizzes to Markdown, returning two MOOCResources:
    - quiz without solutions
    - quiz with solutions and hints if any
    """

    def parse(
        self,
        course_path: str,
        elem_vertical: _Element,
        vertical_display_name: str,
        tag_metadata: dict,
    ) -> list[MOOCResource]:
        mooc_resources: list[MOOCResource] = []

        url_name = elem_vertical.get("url_name", "")
        if not url_name:
            logger.warning("QuizParser: missing url_name in elem_vertical")
            return []

        quiz_xml_path = Path(course_path) / "problem" / f"{url_name}.xml"
        root = load_root_elem_from_mooc_xml(quiz_xml_path)
        if root is None:
            return []

        quiz_data = self.extract_quiz_data(root)
        if quiz_data is None:
            # Unsupported quiz type
            return []

        resource_title = f"{vertical_display_name} - {quiz_data.display_name}".strip(" -")

        out_dir = Path(course_path) / "quiz"
        out_dir.mkdir(parents=True, exist_ok=True)

        quiz_md_path = out_dir / f"{url_name}.quiz.md"
        quiz_sol_md_path = out_dir / f"{url_name}.quiz_solutions.md"

        quiz_md_text = self.render_markdown(quiz_data, include_solutions=False)
        quiz_sol_md_text = self.render_markdown(quiz_data, include_solutions=True)

        quiz_md_path.write_text(quiz_md_text, encoding="utf-8")
        quiz_sol_md_path.write_text(quiz_sol_md_text, encoding="utf-8")

        # We split the quiz between quiz and quiz with solutions for keeping the structure of
        # assignment vs solution used by Tutor Bot
        number_str = self.extract_quiz_number(resource_title=resource_title)

        tag_name = "MOOC_QUIZ"
        tag_dict = tag_metadata.get(tag_name)

        # Quiz
        quiz_res: MOOCResource = MOOCResource(
            source="mooc",
            title=resource_title,
            url=None,
            path=str(quiz_md_path),
            mime_type=mt.guess_mime_type(str(quiz_md_path)),
            is_video=False,
            is_solution=False,
            is_gemini_processed_video=False,
            model=None,
            tag=tag_name,
            type=tag_dict.get("type"),
            subtype=tag_dict.get("subtype"),
            number=number_str,
            one_chunk_per_page=tag_dict.get("one_chunk_per_page"),
            one_chunk_per_doc=tag_dict.get("one_chunk_per_doc"),
            processing_method=tag_dict.get("processing_method"),
        )
        mooc_resources.append(quiz_res)

        # Quiz with solution
        quiz_sol_res: MOOCResource = MOOCResource(
            source="mooc",
            title=resource_title,
            url=None,
            path=str(quiz_sol_md_path),
            mime_type=mt.guess_mime_type(str(quiz_sol_md_path)),
            is_video=False,
            is_solution=True,
            is_gemini_processed_video=False,
            model=None,
            tag=tag_name,
            type=tag_dict.get("type"),
            subtype=tag_dict.get("subtype"),
            number=number_str,
            one_chunk_per_page=tag_dict.get("one_chunk_per_page"),
            one_chunk_per_doc=tag_dict.get("one_chunk_per_doc"),
            processing_method=tag_dict.get("processing_method"),
        )
        mooc_resources.append(quiz_sol_res)

        return mooc_resources

    def extract_quiz_number(self, resource_title: str) -> str | None:
        """
        Extract quiz number by taking everything from the first digit to the end of the string.
        """

        for idx, ch in enumerate(resource_title):
            if ch.isdigit():
                return resource_title[idx:].strip()
        return None

    def extract_quiz_data(self, root: _Element) -> QuizData | None:
        """
        Extract quiz data from the xml element
        """

        display_name = clean_text(root.get("display_name", "")) or "Quiz"

        mcr = root.find(".//multiplechoiceresponse")
        cr = root.find(".//choiceresponse")

        if mcr is not None and cr is not None:
            logger.info("QuizParser: unsupported quiz (multiple response types found)")
            return None

        if mcr is not None:
            return self.extract_single_choice(display_name, mcr)

        if cr is not None:
            return self.extract_checkbox_choice(display_name, cr)

        logger.info("QuizParser: unsupported quiz (no recognized response node)")
        return None

    def extract_single_choice(self, display_name: str, mcr: _Element) -> QuizData | None:
        """
        Extract single choice
        """

        choicegroup = mcr.find("./choicegroup")
        if choicegroup is None:
            choicegroup = mcr.find(".//choicegroup")
        if choicegroup is None:
            logger.info("QuizParser: unsupported multiple choice (no choicegroup)")
            return None

        statement_md = self.extract_statement_md(
            parent=mcr,
            stop_tags={"choicegroup", "solution", "demandhint"},
        )

        options = self.extract_options(choicegroup)
        general_feedback_md = self.extract_general_solution_md(mcr)

        return QuizData(
            display_name=display_name,
            quiz_type="single",
            statement_md=statement_md,
            options=options,
            general_feedback_md=general_feedback_md,
        )

    def extract_checkbox_choice(self, display_name: str, cr: _Element) -> QuizData | None:
        """
        Extract Multiple choice
        """

        checkboxgroup = cr.find("./checkboxgroup")
        if checkboxgroup is None:
            checkboxgroup = cr.find(".//checkboxgroup")
        if checkboxgroup is None:
            logger.info("QuizParser: unsupported checkbox quiz (no checkboxgroup)")
            return None

        statement_md = self.extract_statement_md(
            parent=cr,
            stop_tags={"checkboxgroup", "solution", "demandhint"},
        )

        options = self.extract_options(checkboxgroup)
        general_feedback_md = self.extract_general_solution_md(cr)

        return QuizData(
            display_name=display_name,
            quiz_type="multi",
            statement_md=statement_md,
            options=options,
            general_feedback_md=general_feedback_md,
        )

    def extract_statement_md(self, parent: _Element, stop_tags: set[str]) -> str:
        """
        Extract statement from quiz
        """
        chunks: list[str] = []

        for child in parent:
            if child.tag in stop_tags:
                break
            if child.tag in STATEMENT_TAGS:
                md = self.xml_to_markdown(child)
                if md.strip():
                    chunks.append(md)

        return normalize_markdown("\n\n".join(chunks))

    def extract_options(self, group: _Element) -> list[QuizOption]:
        """
        Extract options from quiz
        """

        options: list[QuizOption] = []

        for choice in group.findall("./choice"):
            is_correct = choice.get("correct", "").lower() == "true"
            text_md = self.choice_text(choice)
            options.append(QuizOption(text_md=text_md, is_correct=is_correct))

        return options

    def choice_text(self, choice: _Element) -> tuple[str, str | None]:
        """
        Extract text
        """
        # tex
        parts: list[str] = []
        if choice.text and clean_text(choice.text):
            parts.append(escape_markdown(clean_text(choice.text)))

        for child in choice:
            if child.tag == "choicehint":
                continue
            md = self.xml_to_markdown(child)
            if md.strip():
                parts.append(md)

        text_md = normalize_markdown("\n\n".join(parts))
        return text_md

    def extract_general_solution_md(self, parent: _Element) -> str | None:
        """
        Extract solution  from quiz
        """

        sol = parent.find(".//solution")
        if sol is None:
            return None

        # <solution><div class="detailed-solution"><p>...</p>...</div></solution>
        container = sol.find("./div[@class='detailed-solution']") or sol.find("./div") or sol

        chunks: list[str] = []

        for p in container.findall("./p"):
            md = self.xml_to_markdown(p)
            if md.strip():
                chunks.append(md)

        solution_text = "\n\n".join(chunks)
        solution_text = solution_text.replace("Explanation", "")
        out = normalize_markdown(solution_text)
        return out or None

    def render_markdown(self, quiz: QuizData, include_solutions: bool) -> str:
        """
        Markdown from quiz data.
        """

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

    def xml_to_markdown(self, elem: _Element | None) -> str:
        """
        From a parsed XML element, generate markdown
        """

        if elem is None:
            return ""

        tag = elem.tag

        if tag in {"label", "description", "div", "p"}:
            return self.children_to_md(elem)

        elif tag == "br":
            return "\n"

        elif tag == "strong":
            inner = normalize_markdown(self.children_to_md(elem))
            return f"**{inner}**" if inner else ""

        # If another tag -> fallback
        return self.children_to_md(elem)

    def children_to_md(self, elem: _Element) -> str:
        """
        XML children nodes to markdown
        """

        parts: list[str] = []

        if elem.text and clean_text(elem.text):
            parts.append(escape_markdown(clean_text(elem.text)))

        for child in elem:
            child_md = self.xml_to_markdown(child)
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
