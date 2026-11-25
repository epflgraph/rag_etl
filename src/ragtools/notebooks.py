"""Notebooks: Jupyter notebooks (.ipynb) turned into Markdown.

The notebook is exported with nbconvert, style and script tags stripped,
and every image it shows left as a Markdown image reference that resolves
from the Markdown file: the outputs (plots) written beside it, the images
on disk rewritten to point at them. Reading the images is left to whoever
reads the Markdown; nothing here calls a model.
"""

from __future__ import annotations

import os
import re

from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import Preprocessor

# Markdown images, ![alt](src), and HTML images, <img ... src="...">
MD_IMAGE = re.compile(r"!\[(.*?)\]\((.*?)\)")
HTML_IMAGE = re.compile(r'<img\s+[^>]*src="([^"]+)"[^>]*>', re.IGNORECASE)
HTML_ALT = re.compile(r'alt="([^"]*)"', re.IGNORECASE)


class _CleanHTMLTags(Preprocessor):
    """Strips style and script tags from HTML outputs and Markdown cells."""

    TAG_RE = re.compile(r"<(style|script).*?>.*?</\1>", flags=re.DOTALL | re.IGNORECASE)

    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type == "code" and "outputs" in cell:
            for output in cell.outputs:
                if "text/html" in output.get("data", {}):
                    output["data"]["text/html"] = self.TAG_RE.sub("", output["data"]["text/html"])

        if cell.cell_type == "markdown":
            cell.source = self.TAG_RE.sub("", cell.source)

        return cell, resources


def convert_ipynb_to_md(ipynb_path: str | Path, md_path: str | Path) -> None:
    """Convert a Jupyter notebook to a Markdown file, its output images written beside it."""

    ipynb_path = Path(ipynb_path)
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ipynb_path, encoding="utf-8") as f:
        notebook_node = nbformat.read(f, as_version=4)

    exporter = MarkdownExporter()
    exporter.register_preprocessor(_CleanHTMLTags(), enabled=True)
    text, resources = exporter.from_notebook_node(notebook_node)

    # The outputs (plots, rendered figures) come back in memory, referenced
    # by bare name: written beside the Markdown, the references resolve
    for name, data in resources.get("outputs", {}).items():
        (md_path.parent / name).write_bytes(data)

    # An image the notebook shows from disk sits relative to the notebook; the
    # Markdown may be written elsewhere, so its reference is rewritten
    # relative to the Markdown (in <> so spaces survive)
    def reference(src: str) -> str:
        image = (ipynb_path.parent / src).resolve()
        return f"<{os.path.relpath(image, md_path.parent.resolve())}>" if image.is_file() else src

    text = MD_IMAGE.sub(lambda m: f"![{m.group(1)}]({reference(m.group(2))})", text)

    def html_image(match: re.Match[str]) -> str:
        alt = HTML_ALT.search(match.group(0))
        return f"![{alt.group(1) if alt else ''}]({reference(match.group(1))})"

    text = HTML_IMAGE.sub(html_image, text)

    md_path.write_text(text, encoding="utf-8")
