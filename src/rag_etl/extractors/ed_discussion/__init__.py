"""The Ed Discussion extractor: an export of a course's forum, one leaf per answered thread."""

import json
import logging
import shutil

from dataclasses import dataclass

import ragtools.mime_types as mt
from rag_etl.core import Resource
from rag_etl.extractors.base import Extractor, SourceUnavailable
from rag_etl.extractors.ed_discussion.thread import thread_markdown

__all__ = ["EdDiscussionExtractor"]

logger = logging.getLogger(__name__)

# The levels Ed files a thread under, outermost first, as named in the export
CATEGORY_LEVELS = ("category", "subcategory", "subsubcategory")

# The subtype every thread carries
FORUM_QA = "forum_qa"


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
    thread, the images its posts embed under `files/` (where a remote
    image is downloaded to, once). The Q&A Markdown is written under
    `markdown/`, rewritten at every extraction.
    """

    academic_year: str  # as "2025_2026"
    language: str       # the forum's, as an ISO code ("fr", "en"): the language of the Q&A headings
    # A student's answer is kept only once staff endorsed it, and only if asked
    include_student_endorsed: bool = False

    def extract(self) -> Resource:
        """Extract the answered threads as leaves, nested under their categories."""

        ed_dir = self.dir / self.academic_year
        if not ed_dir.is_dir():
            raise SourceUnavailable(f"No Ed Discussion export at {ed_dir}")

        # Rewritten from scratch, so a thread the export lost does not linger
        markdown_dir = ed_dir / "markdown"
        shutil.rmtree(markdown_dir, ignore_errors=True)
        markdown_dir.mkdir(parents=True)

        root = Resource(title=f"Ed Discussion {self.academic_year}")
        containers: dict[tuple[str, ...], Resource] = {}

        for json_path in ed_dir.glob("*.json"):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            thread = data.get("thread", {})

            markdown = thread_markdown(
                thread, data.get("users", []), ed_dir / "files", self.language, self.include_student_endorsed
            )
            if markdown is None:
                logger.info(f"Skipping {json_path.name}: no question or no answer kept")
                continue

            md_path = markdown_dir / f"{json_path.stem}.md"
            md_path.write_text(markdown, encoding="utf-8")

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

            # That a thread is a forum Q&A is Ed's to say, not the judge's to guess: it
            # tells the judge and the consumers apart from the material it discusses
            parent.add_child(
                Resource(title=thread.get("title", ""), path=md_path, mime_type=mt.MARKDOWN, subtype=FORUM_QA)
            )

        logger.info(f"Extracted {sum(1 for node in root.walk() if node.path is not None)} Ed Discussion threads")
        return root
