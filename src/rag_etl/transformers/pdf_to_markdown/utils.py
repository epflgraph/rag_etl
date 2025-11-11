import io
import asyncio
import base64

from typing import Optional, List

import pymupdf
from PIL import Image

from rag_etl.utils.llms import send_llm_request


def render_pdf_pages(pdf_path: str, dpi: Optional[int] = None) -> List[Image.Image]:
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


def downscale_if_needed(img: Image.Image, max_w: int = 2048, max_h: int = 3072) -> Image.Image:
    """Downscale only if image exceeds given bounds; preserve sharpness with LANCZOS."""

    w, h = img.size
    if w <= max_w and h <= max_h:
        return img

    scale = min(max_w / w, max_h / h)
    new_size = (int(w * scale), int(h * scale))

    return img.resize(new_size, Image.LANCZOS)


def to_data_uri(img: Image.Image) -> str:
    """Encode a PIL Image as base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def convert_page_pdf_to_md(pil_page):
    # Prompts
    system_prompt = """
    You are an expert PDF→Markdown converter. Convert the visual content of a *single PDF page* into **clean, semantically-accurate GitHub-Flavored Markdown**.

    Hard rules:
    - Output ONLY valid Markdown (no explanations, metadata, or commentary).
    - Keep exact reading order within the page.
    - Maintain complete fidelity to the original layout, hierarchy, and text.
    - Use proper headings (#, ##, ###) that reflect the page’s visual hierarchy.
    - Preserve paragraphs, lists, blockquotes, code blocks, links, footnotes, and captions.
    - Tables must be valid **GFM tables** with header rows when present.
    - Figures/diagrams become Markdown images with concise descriptive ALT text: `![…]` (put captions as normal text below if visible).
    - Math: keep LaTeX. Inline: `$…$`; block: `$$…$$`. When equations were aligned, render as one block within `$$…$$`.
    - Do not invent or omit content. Ignore page numbers/footers/headers repeated on every page.
    """

    user_prompt = """
    Convert the following PDF page to GitHub-Flavored Markdown.

    Context: This is only one page of possibly many pages in the original PDF file. Do not reference other pages. Output **only** the Markdown for this page.

    Follow these MANDATORY rules:

    1. **Structure Preservation**
    - Use proper Markdown headings (#, ##, ###) matching the visual hierarchy.
    - Always include titles, subtitles, and section headers as part of the Markdown hierarchy.
    - Maintain paragraphs, bullet and numbered lists, blockquotes, code blocks, and inline formatting (bold, italics, monospace).
    - Preserve links and footnotes accurately.

    2. **Tables**
    - Represent all tables as **GitHub-Flavored Markdown tables**.
    - Align columns and retain header rows and data integrity.

    3. **Figures and Images**
    - Replace figures, diagrams, or embedded images with a concise and descriptive ALT text in Markdown image syntax: `![ALT text]`.
    - Include any text, variable names or similar annotations from the Figure as part of its ALT text.
    - The ALT text should briefly describe the image so that a visually impaired reader can picture it clearly in their mind.
    - The ALT text is not the caption.
    - Never reproduce the figure’s visual labels as text or math unless they are part of surrounding body text.

    4. **Mathematical Content**
    - Preserve mathematical expressions as LaTeX:
      - Inline math: `$ ... $`
      - Block math: `$$ ... $$`
      - When multiple aligned equations are detected, render them as a single block math region within `$$ ... $$` instead.

    5. **Fidelity and Consistency**
    - Keep content in the exact reading order.
    - Do not paraphrase, summarize, or add commentary.
    - Include all visible textual elements (titles, captions, labels) except for page numbers.

    Output only the final, complete GitHub Flavored Markdown document—nothing else.
    """

    ################################################################
    # NOTE                                                         #
    # We are trying to impose GitHub-flavored Markdown:            #
    # https://github.github.com/gfm/                               #
    #                                                              #
    # This should faithfully translate both document structure and #
    # math environments reasonably well. However, we are always at #
    # the mercy of the LLM, so in some cases the resulting         #
    # document could not render as expected.                       #
    #                                                              #
    # The following online renderer seems to do a good job with    #
    # the generated Markdown files for visual checks:              #
    # https://kerzol.github.io/markdown-mathjax/editor.html        #
    ################################################################

    # Convert image to data uri
    data_uri = to_data_uri(pil_page)

    # Prepare messages
    user_message_content = [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": data_uri}}
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_content},
    ]

    # Send LLM requests and store results
    rcp_model = 'Qwen/Qwen2.5-VL-72B-Instruct'
    md_page = send_llm_request(rcp_model, messages).strip()

    return md_page


def stitch_md_pages(md_pages):
    # Make LLM call to fix possible Markdown issues due to processing page by page
    system_prompt = """
    You will receive multiple Markdown snippets, one per PDF page, enclosed in triple backticks, in strict page order.

    Goal: stitch them into a single clean GitHub-Flavored Markdown document **without mixing distinct sections** (e.g., problem statements vs. solutions), and **without reordering** content.

    Core constraints (must follow all):
    - Preserve page order exactly; do not reorder snippets. Absolutely no cross-page interleaving.
    - Keep all authored content; **do not** summarize, paraphrase, or invent text.
    - Normalize heading levels so hierarchy is consistent across pages (ensure exactly one top-level `#` document title if present).
    - Merge paragraphs/lists if clearly split across two consecutive pages (continue ordered list numbering correctly); fix soft hyphenation at line ends.
    - Merge multi-page tables if a table is clearly split across two consecutive pages; preserve valid GFM table syntax.
    - Merge multi-page display equations if a block is split across two consecutive pages; keep LaTeX integrity.
    - Keep image `![ALT]` items and captions in place; do not generate images.

    Output **only** the final Markdown (no explanations, metadata, or commentary).
    """

    md_text = '\n\n'.join(['```\n' + md_page + '\n```' for md_page in md_pages])

    user_prompt = f"""
    Stitch the following page-level Markdown snippets into one cohesive GitHub-Flavored Markdown document.

    Each snippet is enclosed in triple backticks and appears **in order**.

    Emit **only** the final Markdown (no fences).

    {md_text}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    rcp_model = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
    md_text = send_llm_request(rcp_model, messages).strip()

    return md_text


def convert_pdf_to_md(pdf_path, md_path):
    ################################################################
    # PDF to page images                                           #
    ################################################################

    # Render pages
    pil_pages = render_pdf_pages(pdf_path)

    # Optional conservative clamp per page (in case image is too big)
    pil_pages = [downscale_if_needed(pil_page) for pil_page in pil_pages]

    ################################################################
    # Page images to page Markdown (in parallel threads)           #
    ################################################################

    # Parse PDF pages to Markdown individually
    async def run_all(pil_pages):
        tasks = [asyncio.to_thread(convert_page_pdf_to_md, pil_page) for pil_page in pil_pages]
        return await asyncio.gather(*tasks)

    md_pages = asyncio.run(run_all(pil_pages))

    ################################################################
    # Stitch page Markdown into one coherent Markdown              #
    ################################################################

    md_text = stitch_md_pages(md_pages)

    ################################################################
    # Store result in file                                         #
    ################################################################

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")
