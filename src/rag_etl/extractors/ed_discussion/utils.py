from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import requests

from rag_etl.config import CONFIG
from rag_etl.extractors.ed_discussion.prompts import (
    CLASSIFY_THREAD_SYSTEM_PROMPT,
    CLASSIFY_THREAD_USER_PROMPT,
)
from rag_etl.utils.llms import generate_alt_text, send_llm_request

logger = logging.getLogger(__name__)

STAFF_ROLES = ["admin", "tutor", "staff"]

IMAGE_EXTENSIONS = ["", ".jpg", ".jpeg", ".png"]

MESSAGE_TYPES = [
    "theory",
    "practice",
    "exam",
    "admin",
    "logistics",
    "bug_or_typo_report",
    "exception_request",
    "other",
]


def get_user_roles(users: list[dict]) -> dict[int, str]:
    """Build a mapping from user_id to course_role."""

    return {user["id"]: user.get("course_role") for user in users}


def find_image_path(url: str, images_dir: Path) -> tuple[bool, str | None, str | None]:
    """Find local image path from URL reference."""

    match = re.search(r'<image[^>]+src="([^"]+)"', url or "")
    if not match:
        return False, None, None

    image_url = match.group(1)
    filename = os.path.basename(image_url)
    base_name, _ = os.path.splitext(filename)

    if image_url.startswith("http"):
        local_path = find_local_image(base_name, images_dir)
        if local_path:
            return True, image_url, local_path
        downloaded_path = download_remote_image(image_url, base_name, images_dir)
        return True, image_url, downloaded_path

    candidate = image_url.lstrip("./")
    local_path = find_local_image(os.path.basename(candidate), images_dir)
    if local_path:
        return True, None, local_path

    return True, None, f"ERROR: {images_dir}/{candidate} not found"


def find_local_image(base_name: str, images_dir: Path) -> str | None:
    """Search for image file with various extensions."""

    if not images_dir:
        return None
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / (base_name + ext)
        if candidate.exists():
            return str(candidate)
    return None


def download_remote_image(url: str, base_name: str, images_dir: Path) -> str | None:
    """Download remote image and save locally."""

    if not images_dir:
        return None

    try:
        logger.info(f"Downloading remote image: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        elif "png" in content_type:
            ext = ".png"
        else:
            ext = ".jpg"

        local_path = images_dir / (base_name + ext)
        images_dir.mkdir(parents=True, exist_ok=True)

        with open(local_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded image to: {local_path}")
        return str(local_path)

    except Exception as e:
        logger.warning(f"Failed to download image {url}: {e}")
        return None


def inject_alt_text_into_content(
    image_path: str | None,
    content_html: str,
    document_text: str,
) -> tuple[str, str]:
    """Inject alt text into content if image is present"""

    if image_path is None or image_path.startswith("ERROR"):
        return content_html, document_text

    try:
        alt_text = generate_alt_text(image_path)
    except Exception as e:
        logger.warning(f"Failed to generate alt text for {image_path}: {e}")
        return content_html, document_text

    if not alt_text:
        return content_html, document_text

    # Inject alt text generated with RCP LLM
    if "</figure>" in content_html:
        updated_html = content_html.replace(
            "</figure>",
            f"</figure>\n{alt_text}\n",
        )
    else:
        updated_html = content_html + f"\n{alt_text}"

    updated_doc = document_text + alt_text
    return updated_html, updated_doc


def classify_thread_with_llm(
    thread_category: str,
    thread_subcategory: str,
    all_messages_html: str,
    all_types: list[str],
    subtype_options: str,
) -> dict[str, Any] | None:
    """Classify thread type using LLM from RCP."""

    try:
        format_instructions = (
            'Output JSON with fields: "type", "subtype", "doc_number", '
            '"doc_subnumber", "week". Use null for missing values.'
        )

        user_content = CLASSIFY_THREAD_USER_PROMPT.format(
            thread_category=thread_category,
            thread_subcategory=thread_subcategory,
            all_messages_html=all_messages_html,
            all_types=", ".join(all_types),
            subtype_options=subtype_options,
            format_instructions=format_instructions,
        )

        messages = [
            {"role": "system", "content": CLASSIFY_THREAD_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        model = CONFIG["RCP_VISION_MODEL"]
        content = send_llm_request(model, messages, name="classify-thread")

        return json.loads(content)

    except Exception as e:
        logger.warning(f"Thread classification failed: {e}")
        return None


def extract_messages_from_thread(
    thread: dict,
    user_roles: dict[int, str],
    thread_title: str,
    include_student_endorsed: bool,
    images_dir: Path,
) -> list[dict]:
    """Extract and annotate messages from a thread."""

    results = []

    if thread.get("type") == "question":
        message = create_message(
            msg_type="question",
            subtype="question",
            raw_content=thread.get("content", ""),
            raw_document=thread.get("document", ""),
            thread_title=thread_title,
            images_dir=images_dir,
        )
        results.append(message)

    for answer in thread.get("answers", []):
        role = user_roles.get(answer.get("user_id"))
        endorsed = answer.get("is_endorsed", False)

        if role in STAFF_ROLES:
            message = create_message(
                msg_type="answer",
                subtype="answer_from_staff",
                raw_content=answer.get("content", ""),
                raw_document=answer.get("document", ""),
                thread_title=thread_title,
                images_dir=images_dir,
            )
            results.append(message)
        elif include_student_endorsed and endorsed:
            subtype = "answer_from_student_endorsed"
            message = create_message(
                msg_type="answer",
                subtype=subtype,
                raw_content=answer.get("content", ""),
                raw_document=answer.get("document", ""),
                thread_title=thread_title,
                images_dir=images_dir,
            )
            results.append(message)

        for comment in answer.get("comments", []):
            comment_messages = process_comment(
                comment=comment,
                user_roles=user_roles,
                include_student_endorsed=include_student_endorsed,
                thread_title=thread_title,
                images_dir=images_dir,
                parent_is_answer=True,
            )
            results.extend(comment_messages)

    for comment in thread.get("comments", []):
        comment_messages = process_comment(
            comment=comment,
            user_roles=user_roles,
            include_student_endorsed=include_student_endorsed,
            thread_title=thread_title,
            images_dir=images_dir,
            parent_is_answer=False,
        )
        results.extend(comment_messages)

    return results


def process_comment(
    comment: dict,
    user_roles: dict[int, str],
    include_student_endorsed: bool,
    thread_title: str,
    images_dir: Path,
    parent_is_answer: bool,
) -> list[dict]:
    """Process a comment and its nested comments."""

    results = []
    role = user_roles.get(comment.get("user_id"))
    endorsed = comment.get("is_endorsed", False)

    if role in STAFF_ROLES:
        subtype = (
            "answer_from_staff_followup" if parent_is_answer else "answer_from_staff"
        )
        message = create_message(
            msg_type=comment.get("type", "comment"),
            subtype=subtype,
            raw_content=comment.get("content", ""),
            raw_document=comment.get("document", ""),
            thread_title=thread_title,
            images_dir=images_dir,
        )
        results.append(message)
    elif include_student_endorsed and endorsed:
        message = create_message(
            msg_type=comment.get("type", "comment"),
            subtype="answer_followup_from_student_endorsed",
            raw_content=comment.get("content", ""),
            raw_document=comment.get("document", ""),
            thread_title=thread_title,
            images_dir=images_dir,
        )
        results.append(message)

    for nested_comment in comment.get("comments", []):
        nested_messages = process_comment(
            comment=nested_comment,
            user_roles=user_roles,
            include_student_endorsed=include_student_endorsed,
            thread_title=thread_title,
            images_dir=images_dir,
            parent_is_answer=False,
        )
        results.extend(nested_messages)

    return results


def create_message(
    msg_type: str,
    subtype: str,
    raw_content: str,
    raw_document: str,
    thread_title: str,
    images_dir: Path,
) -> dict:
    """Create an message with image processing."""

    content = raw_content
    document = raw_document
    document_with_title = f"{thread_title}\n\n" + document

    is_img, remote_url, local_path = find_image_path(content, images_dir)

    if is_img and local_path and not local_path.startswith("ERROR"):
        content, document = inject_alt_text_into_content(local_path, content, document)

    return {
        "msg_type": msg_type,
        "subtype": subtype,
        "is_there_image": is_img,
        "remote_image_url": remote_url,
        "local_image_filepath": local_path,
        "content": content,
        "document": document,
        "document_with_title": document_with_title,
    }


def extract_qa_content(messages: list[dict]) -> dict[str, Any]:
    """Extract question and answer content from messages."""

    question_content = ""
    answers = []
    counter = 0
    for message in messages:
        msg_type = message.get("msg_type", "")
        subtype = message.get("subtype", "")
        document = message.get("document", "")
        document_with_title = message.get("document_with_title", "")

        if msg_type == "question" and subtype == "question":
            if counter == 0:
                question_content = document_with_title
            else:
                question_content = document

        elif msg_type in ["answer", "comment"] and "answer" in subtype:
            answers.append(document)
        counter += 1

    return {"question": question_content, "answers": answers}


def format_qa(qa_data: dict[str, Any], language: str) -> str:
    """Format Q&A data."""

    lines = []

    if language.lower() == "french":
        question_label = "# Question :"
        answer_label = "# Réponse"
    else:
        question_label = "# Question:"
        answer_label = "# Answer"

    if qa_data["question"]:
        lines.append(question_label)
        lines.append(qa_data["question"])
        lines.append("")

    num_answers = len(qa_data["answers"])
    for i, answer in enumerate(qa_data["answers"], 1):
        if num_answers == 1:
            if language.lower() == "french":
                lines.append(f"{answer_label} :")
            else:
                lines.append(f"{answer_label}:")

        else:
            if language.lower() == "french":
                lines.append(f"{answer_label} {i} :")
            else:
                lines.append(f"{answer_label} {i}:")
        lines.append(answer)
        if i < num_answers:
            lines.append("")

    return "\n".join(lines)
