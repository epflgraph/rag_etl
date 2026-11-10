import json
import logging
from dataclasses import dataclass
from pathlib import Path

import ragtools.mime_types as mt
from rag_etl.core import Resource
from rag_etl.extractors.base import Extractor, SourceUnavailable
from rag_etl.extractors.ed_discussion.utils import (
    extract_messages_from_thread,
    extract_qa_content,
    format_qa,
    get_user_roles,
)

logger = logging.getLogger(__name__)

# The levels Ed files a thread under, outermost first, as named in the export
CATEGORY_LEVELS = ("category", "subcategory", "subsubcategory")


@dataclass(frozen=True)
class EdDiscussionExtractor(Extractor):
    """Extractor for the questions answered on Ed Discussion.

    The tree follows Ed's own filing: one container per category, then
    per subcategory, down to one leaf per answered thread, its question
    and answers written as markdown. Leaves carry facts only; what a
    thread is about, and which threads to leave out, is the judge's to
    decide and the project's overrides to correct — the category titles
    ("Exams", "Series 3", ...) are what both read.

    The export is read from `dir / academic_year`: one JSON file per
    thread, the images its posts embed under `files/`.
    """

    academic_year: str  # as "2025_2026"
    language: str       # of the Q&A headings: "french" or anything else for English
    # A student's answer is kept only once staff endorsed it, and only if asked
    include_student_endorsed: bool = False

    def extract(self) -> Resource:
        """Extract the answered threads as leaves, nested under their categories."""

        ed_dir = self.dir / self.academic_year
        if not ed_dir.is_dir():
            raise SourceUnavailable(f"No Ed Discussion export at {ed_dir}")

        markdown_dir = ed_dir / "markdown"
        markdown_dir.mkdir(parents=True, exist_ok=True)

        root = Resource(title=f"Ed Discussion {self.academic_year}")
        containers: dict[tuple[str, ...], Resource] = {}

        for json_path in ed_dir.glob("*.json"):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            thread = data.get("thread", {})

            leaf = self.thread_leaf(thread, data.get("users", []), ed_dir / "files", markdown_dir / f"{json_path.stem}.md")
            if leaf is None:
                logger.info(f"Skipping {json_path.name}: no question or no answer kept")
                continue

            # The thread goes under the categories Ed filed it in, each level
            # a container created the first time a thread names it
            parent = root
            names: tuple[str, ...] = ()
            for level in CATEGORY_LEVELS:
                name = thread.get(level) or ""
                if not name:
                    break
                names += (name,)
                if names not in containers:
                    containers[names] = parent.add_child(Resource(title=name))
                parent = containers[names]

            parent.add_child(leaf)

        logger.info(f"Extracted {sum(1 for node in root.walk() if node.path is not None)} Ed Discussion threads")
        return root

    def thread_leaf(self, thread: dict, users: list[dict], images_dir: Path, md_path: Path) -> Resource | None:
        """Write one thread's question and kept answers as markdown; None when either is missing."""

        title = thread.get("title", "")
        messages = extract_messages_from_thread(
            thread=thread,
            user_roles=get_user_roles(users),
            thread_title=title,
            include_student_endorsed=self.include_student_endorsed,
            images_dir=images_dir,
        )

        qa_data = extract_qa_content(messages)
        if not qa_data["question"] or not qa_data["answers"]:
            return None

        md_path.write_text(format_qa(qa_data, self.language), encoding="utf-8")

        return Resource(title=title, path=md_path, mime_type=mt.MARKDOWN)
