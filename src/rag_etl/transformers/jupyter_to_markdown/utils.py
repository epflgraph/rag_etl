from pathlib import Path

import re

import nbformat
from nbconvert import MarkdownExporter

from rag_etl.utils.llms import generate_alt_text


def convert_ipynb_to_md(ipynb_path, md_path):
    """
    Convert a Jupyter Notebook (.ipynb) to a Markdown (.md) file.

    Parameters:
        ipynb_path (str or Path): Path to the input Jupyter notebook file.
        md_path (str or Path): Path for the output Markdown file.
    """

    ################################################################
    # Convert notebook to Markdown                                 #
    ################################################################

    # Normalise to pathlib Paths
    ipynb_path = Path(ipynb_path)
    md_path = Path(md_path)

    # Read notebook
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook_node = nbformat.read(f, as_version=4)

    # Export to markdown
    markdown_exporter = MarkdownExporter()
    text, _ = markdown_exporter.from_notebook_node(notebook_node)

    ################################################################
    # Replace images with ALT texts                                #
    ################################################################

    # Replace Markdown images (![alt](src))
    def md_image_replace(match):
        alt, src = match.group(1), match.group(2)
        p = (md_path.parent / src).resolve()

        if p.exists():
            alt = generate_alt_text(str(p))

        return f"![{alt}]({src})"

    md_image_re = re.compile(r'!\[(.*?)\]\((.*?)\)')
    text = md_image_re.sub(md_image_replace, text)

    # Replace HTML images (<img...)
    def html_image_replace(m):
        tag, src = m.group(0), m.group(1)
        p = (md_path.parent / src).resolve()

        if p.exists():
            alt = generate_alt_text(str(p))
        else:
            alt_match = re.search(r'alt="([^"]*)"', tag, re.IGNORECASE)
            alt = alt_match.group(1) if alt_match else ""

        return f"![{alt}]({src})"

    html_image_re = re.compile(r'<img\s+[^>]*src="([^"]+)"[^>]*>', re.IGNORECASE)
    text = html_image_re.sub(html_image_replace, text)

    # Write md file
    md_path.write_text(text, encoding="utf-8")
