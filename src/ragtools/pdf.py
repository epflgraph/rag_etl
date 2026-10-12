"""PDFs: deterministic page rendering.

A PDF is rendered to one PNG per page with PyMuPDF, at a fixed zoom, so
the same PDF always renders to the same page images — this is what makes
per-page cache keys stable. Pages are clamped to a maximum size before
they are handed to a vision model.
"""

from __future__ import annotations

import pymupdf
from PIL import Image

from pathlib import Path

# PDF page prompt: written for single pages of documents, where layout
# and math environments have to survive page by page
PDF_SYSTEM_PROMPT = """
You are an expert PDF→Markdown converter. Convert the visual content of a *single PDF page* into **clean, semantically-accurate GitHub-Flavored Markdown**.

Hard rules:
- Output ONLY valid Markdown (no explanations, metadata, or commentary).
- Keep exact reading order within the page.
- Maintain complete fidelity to the original layout, hierarchy, and text.
- Use proper headings (#, ##, ###) that reflect the page's visual hierarchy.
- Preserve paragraphs, lists, blockquotes, code blocks, links, footnotes, and captions.
- Tables must be valid **GFM tables** with header rows when present.
- Figures/diagrams become Markdown images with concise descriptive ALT text: `![…]` (put captions as normal text below if visible).
- Math: keep LaTeX. Inline: `$…$`; block: `$$…$$`. When equations were aligned, render as one block within `$$…$$`.
- Do not invent or omit content. Ignore page numbers/footers/headers repeated on every page.
"""

PDF_USER_PROMPT = """
Convert the following PDF page to GitHub-Flavored Markdown.

Context: This is only one page of possibly many pages in the original PDF file. Do not reference other pages. Output **only** the Markdown for this page.

Follow these MANDATORY rules:

1. **Structure Preservation**
- Use proper Markdown headings (#, ##, ###) matching the visual hierarchy.
- Always include titles, subtitles, and section headers as part of the Markdown hierarchy.
- Maintain paragraphs, bullet and numbered lists, blockquotes, and inline formatting (bold, italics, monospace).
- Preserve links and footnotes accurately.

2. **Tables**
- Represent all tables as **GitHub-Flavored Markdown tables**.
- Align columns and retain header rows and data integrity.

3. **Figures and Images**
- Replace figures, diagrams, or embedded images with a concise and descriptive ALT text in Markdown image syntax: `![ALT text]`.
- Include any text, variable names or similar annotations from the Figure as part of its ALT text.
- The ALT text should briefly describe the image so that a visually impaired reader can picture it clearly in their mind.
- The ALT text is not the caption.
- Never reproduce the figure's visual labels as text or math unless they are part of surrounding body text.

4. **Mathematical Content**
- Preserve mathematical expressions as LaTeX:
  - Inline math: `$ ... $`
  - Block math: `$$ ... $$`
  - When multiple aligned equations are detected, render them as a single block math region within `$$ ... $$` instead.

5. - **Code**:
- Use 'triple backticks' Markdown notation, indicating the programming language to separate code from the rest of the content on the page. e.g.
```cpp
# Here is some C++ code on the slide
using namespace std;
```

6. **Fidelity and Consistency**
- Keep content in the exact reading order.
- Do not paraphrase, summarize, or add commentary.
- Include all visible textual elements (titles, captions, labels) except for page numbers.

Output only the final, complete GitHub Flavored Markdown document—nothing else.
"""

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
