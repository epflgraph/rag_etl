import base64
import logging
from pathlib import Path

from rag_etl.config import CONFIG
from rag_etl.utils.llms import send_llm_request
import rag_etl.utils.mime_types as mt

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert image→Markdown converter. Convert the visual content of a *single lecture slide or figure* into **clean, semantically-accurate GitHub-Flavored Markdown**.

Hard rules:
- Output ONLY valid Markdown (no explanations, metadata, or commentary).
- Keep exact reading order within the image.
- Maintain complete fidelity to the original layout, hierarchy, and text.
- Use proper headings (#, ##, ###) that reflect the slide's visual hierarchy.
- Preserve paragraphs, lists, blockquotes, code blocks, links, footnotes, and captions.
- Tables must be valid **GFM tables** with header rows when present.
- Transcribe handwritten annotations as well as printed text, in the position they belong to.
- Figures/diagrams become Markdown images with concise descriptive ALT text: `![…]`.
- Math: keep LaTeX. Inline: `$…$`; block: `$$…$$`. When equations were aligned, render as one block within `$$…$$`.
- Do not invent or omit content. Ignore slide numbers and the footer repeated on every slide.
"""

USER_PROMPT = """
Convert this image (or lecture slide) to GitHub-Flavored Markdown.

Context: This image might be a slide out of many in a recorded lecture. Do not reference other slides. Output **only** the Markdown for this slide.

Follow these MANDATORY rules:

1. **Structure Preservation**
- Use proper Markdown headings (#, ##, ###) matching the visual hierarchy.
- Always include the slide title as part of the Markdown hierarchy.
- Maintain paragraphs, bullet and numbered lists, blockquotes, code blocks, and inline formatting (bold, italics, monospace).

2. **Handwriting**
- The lecturer writes on the slide while speaking. Transcribe those handwritten annotations too, using the same LaTeX rules as for printed mathematics.
- Place each annotation where it appears in the reading order, not in a separate section.
- If an annotation is cut off or unfinished, transcribe what is visible and stop; do not complete it.

3. **Tables**
- Represent all tables as **GitHub-Flavored Markdown tables**.

4. **Figures and Images**
- Replace figures, diagrams, or embedded images with a concise and descriptive ALT text in Markdown image syntax: `![ALT text]`.
- Include any text, variable names or similar annotations from the figure as part of its ALT text.
- The ALT text should briefly describe the image so that a visually impaired reader can picture it clearly in their mind.
- Never reproduce the figure's visual labels as text or math unless they are part of surrounding body text.

5. **Mathematical Content**
- Preserve mathematical expressions as LaTeX:
  - Inline math: `$ ... $`
  - Block math: `$$ ... $$`
  - When multiple aligned equations are detected, render them as a single block math region within `$$ ... $$` instead.

6. **Fidelity and Consistency**
- Keep content in the exact reading order.
- Do not paraphrase, summarize, or add commentary.
- Ignore the recurring footer (author, lecture name, date, slide number).

If the slide is blank or carries no readable content, output nothing at all.

Output only the final, complete GitHub Flavored Markdown document—nothing else.
"""


def to_data_uri(image_path: Path) -> str:
    """Encode an image file as a base64 data URI."""

    mime_type = mt.guess_mime_type(str(image_path)) or mt.JPEG
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def strip_code_fence(markdown: str) -> str:
    """
    Remove a fence wrapping the whole answer.

    Asked for Markdown, the model sometimes returns the document inside a
    ```markdown block. Left in place, the entire slide would render as a code
    listing, so an outer fence is unwrapped. Fenced code *within* the slide is
    untouched: only a fence on the very first line qualifies.
    """

    lines = markdown.strip().splitlines()

    if len(lines) < 2:
        return markdown.strip()

    if not lines[0].strip().startswith("```"):
        return markdown.strip()

    if lines[0].strip().lower() not in ("```", "```markdown", "```md"):
        return markdown.strip()

    if lines[-1].strip() != "```":
        return markdown.strip()

    return "\n".join(lines[1:-1]).strip()


def convert_image_to_md(image_path: Path, md_path: Path) -> None:
    """
    Convert one image into Markdown and write it to md_path

    Thinking is disabled: a fraction of the tokens with comparable results
    """

    user_message_content = [
        {"type": "text", "text": USER_PROMPT},
        {"type": "image_url", "image_url": {"url": to_data_uri(image_path), "detail": "high"}},
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message_content},
    ]

    rcp_model = CONFIG["RCP_VISION_MODEL"]
    markdown = send_llm_request(
        rcp_model,
        messages,
        name="image-to-markdown",
        enable_thinking=False,
    ).strip()

    # Remove fence wrapping in case there is any
    markdown_cleaned = strip_code_fence(markdown)

    # Store result in file
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown_cleaned, encoding="utf-8")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m rag_etl.transformers.image_to_md.utils <image path> [md path]")

    image_path = Path(sys.argv[1])

    if len(sys.argv) > 2:
        md_path = Path(sys.argv[2])
    else:
        md_path = image_path.with_suffix(".md")

    convert_image_to_md(image_path, md_path)

    logging.info(f"Wrote {md_path}")
    print(md_path.read_text(encoding="utf-8"))
