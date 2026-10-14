"""The per_exercise annotator: exercise number + is-solution, chunk-level.

One LLM pass over the stitched document: the text is split into
heading-bounded sections, each annotated with line numbers, and the model
returns the exercises as line ranges with their number and whether the
chunk holds a solution. Nothing is annotated unless the per_exercise cut
asks for it.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, Field

from ragtools.llm import RCPClient

from rag_etl.transformers.cut.headings import MAX_SECTION_LINES, merge_up_to_max_lines, split_by_most_common_heading

logger = logging.getLogger(__name__)

# The annotator's system prompt, kept as a document next to this module
EXERCISE_SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "annotate_exercises.md").read_text(encoding="utf-8")

# How the model is asked to split a document into exercises
ANNOTATOR_NAME = "split-exercises"


@dataclass(frozen=True)
class ExerciseSnippet:
    """One exercise cut from a stitched document."""

    number: str
    is_solution: bool
    text: str


class Exercise(BaseModel):
    start_line: int = Field(
        ..., description="The first line of the Markdown chunk. Its contents are included in the chunk."
    )
    end_line: int = Field(
        ..., description="The last line of the Markdown chunk. Its contents are included in the chunk."
    )
    number: str = Field(
        ...,
        description="The exercise number, as referenced in the Markdown chunk. Typically one or more integers, as in `3` or `2.7`.",
    )
    is_solution: bool = Field(
        ...,
        description="Whether the Markdown chunk contains the solution of the exercise, as opposed to only the question.",
    )


class ExerciseList(BaseModel):
    exercises: list[Exercise]


def annotate(
    text: str,
    llm: RCPClient,
    model: str,
    params: Mapping[str, Any] | None = None,
) -> list[ExerciseSnippet]:
    """Annotate a stitched document, one (number, is-solution, text) per exercise.

    The document is split into heading-bounded sections so one call never
    sees more than MAX_SECTION_LINES lines. Exercises appearing more than
    once (statement and solution, or an exercise cut across pages) produce
    one snippet each; duplicates merge by (number, is-solution).
    """
    snippets: dict[tuple[str, bool], str] = {}

    for section in merge_up_to_max_lines(split_by_most_common_heading(text), MAX_SECTION_LINES):
        lines = section.splitlines()

        # Annotate the section with line numbers (ensuring empty line at the end)
        annotated = "\n".join([f"[L{i}] {line}" for i, line in enumerate(lines, start=1)] + [""])

        exercise_list = llm.structured(
            annotated,
            ExerciseList,
            model=model,
            system=EXERCISE_SYSTEM_PROMPT,
            name=ANNOTATOR_NAME,
            params=params,
        )

        for exercise in exercise_list.exercises:
            # Skip exercises whose lines make no sense
            if not 1 <= exercise.start_line <= exercise.end_line <= len(lines):
                logger.warning(
                    f"Exercise lines out of bounds: start {exercise.start_line}, end {exercise.end_line}, "
                    f"total {len(lines)}. Skipping..."
                )
                continue

            snippet = "\n".join(lines[exercise.start_line - 1 : exercise.end_line])
            key = (exercise.number, exercise.is_solution)
            snippets[key] = snippets.get(key, "") + snippet

    return [
        ExerciseSnippet(number=number, is_solution=is_solution, text=snippet)
        for (number, is_solution), snippet in snippets.items()
    ]
