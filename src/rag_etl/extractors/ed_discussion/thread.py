"""One Ed Discussion thread as Markdown: its question, then the answers kept, in the forum's language."""

from __future__ import annotations

import logging
import os
import re

from collections.abc import Iterator
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

STAFF_ROLES = ("admin", "tutor", "staff")

# An image a post shows, in the post's XML content
IMAGE = re.compile(r'<image[^>]+src="([^"]+)"')

# What an image's name is tried with in the export's files/ (Ed drops the extension of some)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

DOWNLOAD_TIMEOUT = 30


def thread_markdown(
    thread: dict, users: list[dict], images_dir: Path, language: str, include_endorsed: bool
) -> str | None:
    """The thread's question and kept answers as Markdown; None when it has no question or no answer kept.

    Kept are the answers and comments of staff, and, if asked, those of
    students that staff endorsed. A post's images are shown as Markdown
    images, which the images step reads.
    """

    title = thread.get("title", "")
    if thread.get("type") != "question":
        return None

    roles = {user["id"]: user.get("course_role") for user in users}
    answers = [_post_text(post, images_dir, title) for post in _kept_answers(thread, roles, include_endorsed)]
    if not answers:
        return None

    return format_qa(f"{title}\n\n{_post_text(thread, images_dir, title)}", answers, language)


def _kept_answers(thread: dict, roles: dict, include_endorsed: bool) -> Iterator[dict]:
    """The posts kept, in thread order: each answer then its comments (nested), then the thread's comments."""

    def kept(post: dict) -> bool:
        return roles.get(post.get("user_id")) in STAFF_ROLES or (include_endorsed and post.get("is_endorsed", False))

    def with_comments(post: dict) -> Iterator[dict]:
        if kept(post):
            yield post
        for comment in post.get("comments", []):
            yield from with_comments(comment)

    for post in [*thread.get("answers", []), *thread.get("comments", [])]:
        yield from with_comments(post)


def _post_text(post: dict, images_dir: Path, thread_title: str) -> str:
    """A post's text, every image it shows appended as a Markdown image."""

    text = post.get("document", "")
    for src in IMAGE.findall(post.get("content", "")):
        image = _image(src, images_dir)
        if image is None:
            logger.warning(f"{thread_title}: image {src} could not be found or downloaded, left out")
        else:
            text += f"\n\n![](<{image}>)"
    return text


def _image(src: str, images_dir: Path) -> Path | None:
    """The image in the export's files/, downloaded there first when it is remote and missing."""

    name = os.path.basename(src)
    stem = os.path.splitext(name)[0]
    for candidate in (name, stem, *(stem + extension for extension in IMAGE_EXTENSIONS)):
        if (images_dir / candidate).is_file():
            return images_dir / candidate

    return _download(src, stem, images_dir) if src.startswith("http") else None


def _download(url: str, stem: str, images_dir: Path) -> Path | None:
    """Download a remote image into files/, once: the next extraction finds it there."""

    try:
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as error:
        logger.warning(f"Failed to download image {url}: {error}")
        return None

    content_type = response.headers.get("content-type", "").lower()
    path = images_dir / f"{stem}{'.png' if 'png' in content_type else '.jpg'}"
    images_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    logger.info(f"Downloaded image to: {path}")
    return path


def format_qa(question: str, answers: list[str], language: str) -> str:
    """The question and its answers under headings in the forum's language (an ISO code: French for "fr", English otherwise)."""

    # French puts a space before the colon
    french = language == "fr"
    colon = " :" if french else ":"
    answer_label = "# Réponse" if french else "# Answer"

    lines = [f"# Question{colon}", question, ""]
    for i, answer in enumerate(answers, 1):
        number = "" if len(answers) == 1 else f" {i}"
        lines += [f"{answer_label}{number}{colon}", answer, ""]

    return "\n".join(lines[:-1])
