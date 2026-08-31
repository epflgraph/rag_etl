import base64
import logging
from pathlib import Path

from rag_etl.config import CONFIG
from rag_etl.transformers.image_to_md.prompts import SYSTEM_PROMPT, USER_PROMPT
from rag_etl.utils.llms import send_llm_request
import rag_etl.utils.mime_types as mt

logger = logging.getLogger(__name__)


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
