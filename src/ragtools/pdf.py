"""PDFs: deterministic page rendering.

A PDF is rendered to one PNG per page with PyMuPDF, at a fixed zoom, so
the same PDF always renders to the same page images — this is what makes
per-page cache keys stable. Pages are clamped to a maximum size before
they are handed to a vision model.
"""

from __future__ import annotations

import base64
import io

from pathlib import Path

import pymupdf
from PIL import Image

# Conservative clamp per page, in case the render is bigger than what a
# vision model handles well: scaling down keeps the small print sharp
MAX_PAGE_WIDTH = 2048
MAX_PAGE_HEIGHT = 3072


def render_pdf_pages(pdf_path: str | Path, dpi: int | None = None) -> list[Image.Image]:
    """
    Render each PDF page to a PIL Image using PyMuPDF (fitz).
    If dpi is provided, scale accordingly; otherwise use default (~72 DPI).

    Returns:
        A list of PIL Images
    """

    doc = pymupdf.open(pdf_path)
    pages = []

    try:
        for page in doc:
            zoom = (dpi / 72.0) if dpi else 1.0
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(img)
    finally:
        doc.close()

    if not pages:
        raise ValueError("PDF has no pages.")

    return pages


def downscale_if_needed(img: Image.Image, max_w: int = MAX_PAGE_WIDTH, max_h: int = MAX_PAGE_HEIGHT) -> Image.Image:
    """Downscale only if image exceeds given bounds; preserve sharpness with LANCZOS."""

    w, h = img.size
    if w <= max_w and h <= max_h:
        return img

    scale = min(max_w / w, max_h / h)
    new_size = (int(w * scale), int(h * scale))

    return img.resize(new_size, Image.Resampling.LANCZOS)


def to_data_uri(img: Image.Image) -> str:
    """Encode a PIL Image as a base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def save_pdf_pages(pdf_path: str | Path, out_dir: str | Path, dpi: int | None = None) -> list[Path]:
    """
    Render a PDF to one PNG per page, written into out_dir in page order.

    Returns the page paths, named page_1.png, page_2.png, ... so the same
    PDF always renders to the same files.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_paths = []
    for page_number, pil_page in enumerate(render_pdf_pages(pdf_path, dpi), start=1):
        pil_page = downscale_if_needed(pil_page)
        page_path = out_dir / f"page_{page_number}.png"
        pil_page.save(page_path, format="PNG")
        page_paths.append(page_path)

    return page_paths
