import io
import asyncio
import base64

from difflib import SequenceMatcher


import openai
import pymupdf
from PIL import Image

from rag_etl.utils.llms import send_llm_request

from rag_etl.config import CONFIG


def render_pdf_pages(pdf_path: str, dpi: int | None = None) -> list[Image.Image]:
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


async def convert_page_pdf_to_md(pil_page, semaphore: asyncio.Semaphore, page_number: int):
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
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_content},
    ]

    # Send LLM requests and store results
    rcp_model = CONFIG["RCP_VISION_MODEL"]
    max_retries = 1

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"attempting page={page_number} attempt={attempt}")
                md_page = await asyncio.to_thread(
                    send_llm_request,
                    rcp_model,
                    messages,
                    name="pdf-page-to-markdown",
                    enable_thinking=False,
                    timeout=120,  # 2 min for OCR without thinking, 10 min by default
                )
                if md_page is not None:
                    md_page = md_page.strip()
                    print(f"finished page={page_number} attempt={attempt}")
                else:
                    md_page = ""
                    print(f"finished EMPTY page={page_number} attempt={attempt}")

                return md_page

            # The client raises its own timeout, which is an APIConnectionError
            # rather than the builtin TimeoutError, so catching the builtin
            # here would let every real timeout escape unretried
            except openai.APITimeoutError:
                if attempt == max_retries:
                    # raise
                    pass
                await asyncio.sleep(2**attempt)


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

    md_text = "\n\n".join(["```\n" + md_page + "\n```" for md_page in md_pages])

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

    rcp_model = CONFIG["RCP_BASE_MODEL"]
    md_text = send_llm_request(rcp_model, messages, name="stitch-markdown-pages", enable_thinking=False)
    if md_text is not None:
        md_text = md_text.strip()

    return md_text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def best_overlap_concat(a: str, b: str, min_ratio: float = 0.8):
    a_lines = a.strip().splitlines()
    b_lines = b.strip().splitlines()

    best = {
        "overlap": 0,
        "ratio": 0.0,
    }

    max_possible = min(len(a_lines), len(b_lines))

    for k in range(1, max_possible + 1):
        a_suffix = "\n".join(a_lines[-k:])
        b_prefix = "\n".join(b_lines[:k])
        r = similarity(a_suffix, b_prefix)

        # Prefer higher ratio; break ties by longer overlap
        if r > best["ratio"] or (r == best["ratio"] and k > best["overlap"]):
            best = {"overlap": k, "ratio": r}

    if best["ratio"] >= min_ratio:
        merged = a_lines + b_lines[best["overlap"] :]
    else:
        merged = a_lines + b_lines

    return "\n".join(merged)


def batch_stitch_md_pages(md_pages):
    print("batch_stitch_md_pages")
    batch_n_pages = 10
    overlap = 2

    md_text = ""
    for i in range(0, len(md_pages), batch_n_pages - overlap):
        print(f"stitch_md_pages start from page {i} to {(i + batch_n_pages)}")
        chunk_md_text = stitch_md_pages(md_pages[i : i + batch_n_pages])
        if chunk_md_text is not None:
            md_text = best_overlap_concat(md_text, chunk_md_text)
        print(f"stitch_md_pages end from page {i} to {(i + batch_n_pages)}")
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
    # Page images to page Markdown (bounded concurrency)           #
    ################################################################

    max_concurrent_pages = 20

    # Parse PDF pages to Markdown individually
    async def run_all(pil_pages):
        semaphore = asyncio.Semaphore(max_concurrent_pages)
        tasks = []
        for page_number, pil_page in enumerate(pil_pages, start=1):
            tasks.append(convert_page_pdf_to_md(pil_page, semaphore, page_number))

        return await asyncio.gather(*tasks)

    md_pages = asyncio.run(run_all(pil_pages))

    ################################################################
    # Stitch page Markdown into one coherent Markdown              #
    ################################################################

    print("before batch_stitch_md_pages")
    md_text = batch_stitch_md_pages(md_pages)
    print("after batch_stitch_md_pages")
    ################################################################
    # Store result in file                                         #
    ################################################################

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")


# if __name__ == '__main__':
#     md_pages = [
#         r"""
#             # Exercise sheet 4
#             ## Exercise 1
#
#             A block of mass \(m = 2\,\text{kg}\) is pushed along a horizontal surface with a constant force \(F = 10\,\text{N}\). The surface is frictionless.""",
#         r"""
#             1. Compute the acceleration of the block.
#             2. Determine the velocity after \(t = 5\,\text{s}\) if the block starts from rest.
#
#             Use Newton's second law:
#
#             \[
#             F = ma
#             \]
#         """,
#         r"""
#             ## Exercise 2
#
#             A car moves with constant acceleration \(a = 3\,\text{m/s}^2\). Its initial velocity is \(v_0 = 4\,\text{m/s}\).
#         """,
#         r"""
#             1. Write the expression for the velocity as a function of time.
#             2. Compute the velocity after \(6\,\text{s}\).
#             """,
#         r"""
#             3. Compute the distance travelled in that time.
#
#             Recall the kinematic equations:
#
#             \[
#             v(t) = v_0 + at
#             \]
#             """,
#         r"""
#
#             \[
#             x(t) = v_0 t + \tfrac{1}{2} a t^2
#             \]
#
#             # Exercise 3
#         """,
#         r"""
#             A ball is thrown vertically upward with an initial velocity \(v_0 = 12\,\text{m/s}\). Assume gravitational acceleration \(g = 9.81\,\text{m/s}^2\).
#
#             1. Determine the time required to reach the maximum height.
#             2. Compute the maximum height reached by the ball.
#         """,
#         r"""
#             Use the relation:
#
#             \[
#             v^2 = v_0^2 - 2 g h
#             \]
#         """,
#         r"""
#             ## Exercise 4
#
#             A spring with spring constant \(k = 200\,\text{N/m}\) is compressed by \(x = 0.10\,\text{m}\).
#         """,
#         r"""
#             1. Compute the elastic potential energy stored in the spring.
#             2. If the spring releases a \(0.5\,\text{kg}\) mass on a frictionless surface, determine the velocity of the mass.
#         """,
#         r"""
#             Use Hooke’s law and the energy relation:
#
#             \[
#             F = -kx
#             \]
#         """,
#         r"""
#             \[
#             E = \tfrac{1}{2} k x^2
#             \]
#
#             # Exercise 5
#         """,
#         r"""
#             A resistor of resistance \(R = 8\,\Omega\) is connected to a battery providing voltage \(V = 12\,\text{V}\).
#
#             1. Compute the current flowing through the resistor.
#             2. Determine the electrical power dissipated.
#         """,
#         r"""
#             Use Ohm’s law:
#
#             \[
#             V = IR
#             \]
#         """,
#         r"""
#             and the power relation:
#
#             \[
#             P = VI
#             \]
#
#             # Exercise 6
#         """,
#         r"""
#             A wave travels along a string with frequency \(f = 50\,\text{Hz}\) and wavelength \(\lambda = 0.60\,\text{m}\).
#         """,
#         r"""
#             1. Compute the wave speed.
#             2. Write the general relation between speed, wavelength, and frequency.
#
#             \[
#             v = f\lambda
#             \]
#         """,
#         r"""
#             # Exercise 7
#
#             A gas occupies a volume \(V = 0.02\,\text{m}^3\) at pressure \(P = 1.5\times10^5\,\text{Pa}\) and temperature \(T = 300\,\text{K}\).
#         """,
#         r"""
#             1. Determine the number of moles of gas present.
#
#             Use the ideal gas law:
#         """,
#         r"""
#             \[
#             PV = nRT
#             \]
#
#             with \(R = 8.314\,\text{J/(mol·K)}\).
#         """,
#         r"""
#             ### Exercise 8
#
#             A photon has wavelength \(\lambda = 500\,\text{nm}\).
#
#             1. Compute the frequency of the photon.
#             2. Determine its energy.
#         """,
#         r"""
#             Use:
#
#             \[
#             c = \lambda f
#             \]
#         """,
#         r"""
#             and
#
#             \[
#             E = hf
#             \]
#         """,
#         r"""
#             where \(h = 6.63\times10^{-34}\,\text{J·s}\) and \(c = 3.0\times10^8\,\text{m/s}\).
#
#             # Exercise 9
#         """,
#         r"""
#             A satellite orbits Earth in a circular orbit of radius \(r = 7.0\times10^6\,\text{m}\).
#
#             1. Write the expression for the gravitational force acting on the satellite.
#             2. Explain how this force provides the centripetal acceleration.
#         """,
#         r"""
#             Use Newton’s law of gravitation:
#
#             \[
#             F = \frac{GMm}{r^2}
#             \]
#         """,
#         r"""
#             # Exercise 10
#
#             Two charges \(q_1 = 2\,\mu\text{C}\) and \(q_2 = -3\,\mu\text{C}\) are separated by a distance \(r = 0.40\,\text{m}\).
#         """,
#         r"""
#             1. Compute the magnitude of the electrostatic force between them.
#             2. State whether the force is attractive or repulsive.
#
#             Use Coulomb’s law:
#         """,
#         r"""
#             \[
#             F = k\frac{|q_1 q_2|}{r^2}
#             \]
#
#             where \(k = 8.99\times10^9\,\text{N·m}^2/\text{C}^2\).
#             """,
#     ]
#
#     md_text = batch_stitch_md_pages(md_pages)
#
#     print('-' * 64)
#     print(md_text)
