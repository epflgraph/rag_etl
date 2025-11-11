from pathlib import Path

from typing import List
from pydantic import BaseModel, Field

from rag_etl.utils.llms import send_llm_request


def split_md_into_exercises(md_path, exercises_path):
    # Normalise to Paths
    md_path = Path(md_path)
    exercises_path = Path(exercises_path)

    # Read Markdown file to be split
    md_text = md_path.read_text(encoding='utf-8')

    # Prepare prompts
    system_prompt = """
You are a careful Markdown document segmenter.
Your task is to read a long Markdown file containing multiple exercises and split it into separate snippets, one per exercise.
You also need to identify the exercise numbers as well as whether the snippets correspond to the exercise statement or the solution.  

Rules:
- Infer exercise boundaries according to headings. Typically, exercises start with a heading containing the text "Exercise N", "Problem N" or even "Solution N".
- Ignore any introductory or preface material that appears before the first exercise.
- If an exercise (e.g. "Exercise 1") appears more than once (for instance, this can happen with the exercise statement and solution), produce a snippet for each occurrence.
- Preserve all Markdown exactly as written.
  - Keep code blocks, math, images, tables, and formatting intact.
  - Do not modify or reformat text within each exercise.
  - Keep footnotes in the exercise they are referenced in, even if they appear later in the document.
- Ensure each snippet is self-contained and logically complete.
- Each exercise should include its title, text, and any associated content.
- The goal is to produce clean, coherent exercise segments suitable for saving as individual Markdown files.
"""

    user_prompt = f"""
Here's a Markdown file containing multiple exercises.
Split it into separate snippets (one per exercise) following the system instructions.

---

{md_text}
"""

    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Prepare response format
    class Exercise(BaseModel):
        snippet: str = Field(..., description="The Markdown snippet of one exercise. Nothing else.")
        number: str = Field(..., description="The exercise number, as referenced in its Markdown snippet. Typically an integer.")
        is_solution: bool = Field(..., description="Whether the Markdown snippet contains the solution of the exercise, as opposed to only the .")

    class ExerciseList(BaseModel):
        exercises: List[Exercise]

    # Call LLM to split into exercises
    rcp_model = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
    exercise_list = send_llm_request(rcp_model, messages, response_format=ExerciseList)

    # Exercises could be repeated (statement and solution). Make unique by number by prioritising the solution
    exercises = {}
    for exercise in exercise_list.exercises:
        if not exercise.number:
            continue

        if exercise.is_solution or exercise.number not in exercises:
            exercises[exercise.number] = exercise

    exercises = list(exercises.values())

    # Store exercises as individual Markdown files
    exercises_path.mkdir(parents=True, exist_ok=True)
    for exercise in exercises:
        exercise_path = exercises_path / exercise.number
        exercise_path = exercise_path.with_suffix('.md')

        exercise_path.write_text(exercise.snippet, encoding="utf-8")
