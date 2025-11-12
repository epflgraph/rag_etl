from __future__ import annotations

from typing import List, Sequence
from pathlib import Path

import logging

from rag_etl.transformers import BaseTransformer
from rag_etl.resources import BaseResource

from rag_etl.transformers.jupyter_to_markdown.utils import convert_ipynb_to_md


class JupyterToMarkdownTransformer(BaseTransformer):
    """
    Transformer that converts Jupyter notebook resources into Markdown resources.

    Non-Jupyter resources are left unchanged.
    """

    def transform(self, resources: Sequence[BaseResource]) -> List[BaseResource]:
        """
        Converts Jupyter notebook resources into Markdown resources.

        Non-Jupyter resources are left unchanged.
        """

        transformed_resources: List[BaseResource] = []

        for resource in resources:
            # Skip if resource is not a Jupyter notebook
            if resource.mime_type != "application/x-ipynb+json":
                transformed_resources.append(resource)
                continue

            # Build paths of ipynb file and md file
            ipynb_path = Path(resource.path)
            md_path = ipynb_path.with_suffix('.md')

            # Only convert if not cached
            if not md_path.exists():
                logging.debug(f"Splitting {resource.path} into exercises")
                convert_ipynb_to_md(ipynb_path, md_path)

            # Build transformed resource and append it
            new_resource = resource.copy_with(
                path=str(md_path),
                mime_type="text/markdown",
                processing_method=None,
            )
            transformed_resources.append(new_resource)

        return transformed_resources
