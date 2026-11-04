"""Image → Markdown for Ed post images.

Moved out of the pre-refactor transformers (image_to_md) when the legacy
tree moved to legacy/: the Ed extractor is the only live consumer. The
reading itself is ragtools' shared image reader — the slide prompt, the
answer cap and the fence strip live there now, worded exactly as here.
"""

from __future__ import annotations

from pathlib import Path

from ragtools.images import ocr_image


def convert_image_to_md(image_path: Path, md_path: Path) -> None:
    """Convert one image into Markdown and write it to md_path."""

    markdown = ocr_image(image_path)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
