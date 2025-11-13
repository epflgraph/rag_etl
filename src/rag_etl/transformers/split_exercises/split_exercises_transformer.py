from __future__ import annotations

from typing import List, Sequence
from pathlib import Path

import logging

from rag_etl.transformers import BaseTransformer
from rag_etl.resources import BaseResource

from rag_etl.transformers.split_exercises.utils import split_md_into_exercises

import rag_etl.utils.mime_types as mt


class SplitExercisesTransformer(BaseTransformer):
    """
    Transformer that splits resources containing exercises into a resource per exercise.

    Only Markdown resources are considered. Any PDF should first be converted to Markdown before splitting.
    """

    def __init__(self, type_subtypes=None) -> None:
        self.type_subtypes = type_subtypes

    def transform(self, resources: Sequence[BaseResource]) -> List[BaseResource]:
        """
        Splits Markdown resources containing exercises into a resource per exercise.
        Non-Markdown resources as well as resources not matching the specified type_subtypes are left unchanged.
        """

        transformed_resources: List[BaseResource] = []

        for resource in resources:
            # Skip if resource is not in the specified list of types and subtypes
            if self.type_subtypes and (resource.type, resource.subtype) not in self.type_subtypes:
                transformed_resources.append(resource)
                continue

            # Skip if resource is not Markdown
            if resource.mime_type != mt.MARKDOWN:
                transformed_resources.append(resource)
                continue

            # Build paths of md file and exercises folder
            md_path = Path(resource.path)
            exercises_path = md_path.parent / 'exercises'

            # Only split if not cached
            if not exercises_path.exists():
                logging.debug(f"Splitting {resource.path} into exercises")
                split_md_into_exercises(md_path, exercises_path)

            # Build resource for each exercise file
            for exercise_md_path in sorted(exercises_path.glob("*.md")):
                new_resource = resource.copy_with(
                    title=f"{resource.title} > Exercise {exercise_md_path.stem}",
                    path=str(exercise_md_path),
                    sub_number=exercise_md_path.stem,
                    processing_method=None,
                    one_chunk_per_doc=True,
                )
                transformed_resources.append(new_resource)

        return transformed_resources
