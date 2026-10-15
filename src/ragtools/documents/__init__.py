"""Documents: generic Markdown capabilities over texts, usable standalone.

Heading-aware splitting, page stitching and exercise splitting take plain
texts and hand text or snippets back; the model each one pins and the
endpoint come from the config.
"""

from ragtools.documents.exercises import Exercise, ExerciseList, ExerciseSnippet, split_exercises
from ragtools.documents.headings import MAX_SECTION_LINES, merge_up_to_max_lines, split_by_most_common_heading
from ragtools.documents.stitch import PAGE_MARKER, llm_stitch, stitch

__all__ = [
    "MAX_SECTION_LINES",
    "Exercise",
    "ExerciseList",
    "ExerciseSnippet",
    "PAGE_MARKER",
    "llm_stitch",
    "merge_up_to_max_lines",
    "split_by_most_common_heading",
    "split_exercises",
    "stitch",
]
