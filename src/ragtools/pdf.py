"""PDFs: deterministic page rendering, at the resolution each page's print needs.

A PDF is rendered to one PNG per page with PyMuPDF. The same PDF always
renders to the same page images — this is what makes per-page cache keys
stable — and each page is rendered just large enough for a vision model to
read its smallest print: a slide in large type stays small, a page of
subscripts or a table in small print is enlarged. Pages are clamped to a
maximum size before they are handed to a vision model.
"""

from collections.abc import Iterator
from pathlib import Path

import pymupdf
from PIL import Image

# How tall, in pixels, a page's smallest text is rendered. Measured on a table
# of numbers in 6.7 pt: at 7 px (72 dpi) the model wrote 83 values that are not
# on the page, at 10 px a handful, from 12 px none
SMALL_PRINT_PX = 12

# A font size in points is its height in pixels at 72 dpi, so the zoom over
# 72 dpi is SMALL_PRINT_PX over the smallest font size. Never below 72 dpi,
# and never above 216: a stray tiny glyph only costs its page a larger render
MIN_ZOOM = 1.0
MAX_ZOOM = 3.0

# A page with no text layer (a scan, text drawn as outlines) says nothing of
# its print: rendered at 150 dpi, the usual resolution for reading scans
NO_TEXT_ZOOM = 150 / 72

# Conservative clamp per page, in case the render is bigger than what a
# vision model handles well: scaling down keeps the small print sharp
MAX_PAGE_WIDTH = 2048
MAX_PAGE_HEIGHT = 3072


def smallest_font_size(page: pymupdf.Page) -> float | None:
    """The smallest font size, in points, of the text in the page's text layer; None when it has none.

    The smallest, not a typical one: a subscript or the cells of a small
    table are a few characters, and the ones a misreading costs most. Only
    the sizes are read, which are the page's geometry and hold even where
    the layer's characters are wrong; the text itself is never used.
    """
    blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)["blocks"]
    sizes = [
        span["size"]
        for block in blocks
        for line in block.get("lines", [])
        for span in line["spans"]
        if span["text"].strip() and span["size"] > 0
    ]
    return min(sizes) if sizes else None


def page_zoom(page: pymupdf.Page) -> float:
    """The zoom over 72 dpi a page is rendered at, for its smallest print to be SMALL_PRINT_PX tall."""
    smallest = smallest_font_size(page)
    if smallest is None:
        return NO_TEXT_ZOOM
    # Two decimals: the same page always gets the same zoom, whatever the float noise in its sizes
    return round(min(MAX_ZOOM, max(MIN_ZOOM, SMALL_PRINT_PX / smallest)), 2)


def render_pdf_pages(pdf_path: str | Path, dpi: int | None = None) -> Iterator[Image.Image]:
    """Render each page of a PDF to a PIL image, one at a time, in page order.

    Each page is rendered at the resolution its own print needs (`page_zoom`),
    or all of them at `dpi` when one is given. Pages are yielded as they are
    rendered, so a long PDF is never held in memory whole.
    """
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            zoom = dpi / 72 if dpi else page_zoom(page)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            yield Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def downscale_if_needed(img: Image.Image, max_w: int = MAX_PAGE_WIDTH, max_h: int = MAX_PAGE_HEIGHT) -> Image.Image:
    """Downscale only if image exceeds given bounds; preserve sharpness with LANCZOS."""

    w, h = img.size
    if w <= max_w and h <= max_h:
        return img

    scale = min(max_w / w, max_h / h)
    new_size = (int(w * scale), int(h * scale))

    return img.resize(new_size, Image.Resampling.LANCZOS)


def save_pdf_pages(pdf_path: str | Path, out_dir: str | Path, dpi: int | None = None) -> list[Path]:
    """
    Render a PDF to one PNG per page, written into out_dir in page order.

    Returns the page paths, named page_1.png, page_2.png, ... so the same
    PDF always renders to the same files. Each page is rendered at the
    resolution its print needs, unless one `dpi` is given for all.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_paths = []
    for page_number, pil_page in enumerate(render_pdf_pages(pdf_path, dpi), start=1):
        page_path = out_dir / f"page_{page_number}.png"
        downscale_if_needed(pil_page).save(page_path, format="PNG")
        page_paths.append(page_path)

    if not page_paths:
        raise ValueError(f"{pdf_path} has no pages.")

    return page_paths
