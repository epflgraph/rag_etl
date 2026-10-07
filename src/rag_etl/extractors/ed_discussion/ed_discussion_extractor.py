import json
import logging
from pathlib import Path

from rag_etl.config import CONFIG
from rag_etl.extractors import BaseExtractor
from rag_etl.extractors.ed_discussion.catalogue import build_catalogue, find_entry, render_catalogue_entry
from rag_etl.extractors.ed_discussion.utils import (
    MESSAGE_TYPES,
    ThreadClassification,
    classify_thread_with_cascade,
    extract_messages_from_thread,
    extract_qa_content,
    format_qa,
    get_user_roles,
)
from rag_etl.resources.ed_discussion_resource import EdDiscussionResource

logger = logging.getLogger(__name__)

# Threads of these types are only useful tied to a specific document: without a catalogue entry they are
# left out of the resources and reclassified at every update, in case the update added their document
CATALOGUE_REQUIRED_TYPES = ("practice", "exam")


class EdDiscussionExtractor(BaseExtractor):
    """Extractor for retrieving previously answered questions from Ed Discussion."""

    def __init__(
        self,
        ed_discussion_base_path: str,
        academic_year: str,
        categories: list[str],
        language: str,
        semester: int,
        include_student_endorsed: bool,
        catalogue_sources: list[str] | None = None,
        force_regeneration: bool = False,
        mime_types: list[str] | None = None,
        models: list[str] | None = None,
    ) -> None:
        self.ed_discussion_base_path = Path(ed_discussion_base_path)
        self.academic_year = academic_year
        self.categories = categories
        self.language = language
        self.semester = semester
        self.include_student_endorsed = include_student_endorsed
        self.catalogue_sources = catalogue_sources
        self.force_regeneration = force_regeneration
        self.models = models if models is not None else self.default_models()
        self.mime_types = mime_types

        self.exam_year = self.compute_exam_year()

        metadata_dir = self.ed_discussion_base_path / "output" / "metadata"
        self.catalogue = build_catalogue(metadata_dir, self.catalogue_sources)
        logger.info(f"Built Ed Discussion catalogue with {len(self.catalogue)} entries from {metadata_dir}")

    @staticmethod
    def default_models() -> list[str]:
        """Build the juror model list from config: the required model plus any optional jurors."""

        models = [CONFIG["RCP_EDSTEM_EXTRACTOR_MODEL"]]
        for key in ("RCP_EDSTEM_EXTRACTOR_MODEL_2", "RCP_EDSTEM_EXTRACTOR_MODEL_3"):
            extra_model = CONFIG.get(key)
            if extra_model:
                models.append(extra_model)
        return models

    def compute_exam_year(self) -> str:
        """Compute exam year based on academic_year ("2025_2026") and semester."""

        years = self.academic_year.split("_")

        # For Spring semester (2) return second year from the academic year
        if len(years) == 2:
            return years[1] if self.semester == 2 else years[0]
        return self.academic_year

    def extract(self) -> list[EdDiscussionResource]:
        """Extract resources for Ed Discussion Q&A threads."""

        # The catalogue comes from the previous run's output, so a course's first run has none. Classifying
        # against an empty catalogue would leave every practice/exam thread unmatched, so wait for the next run
        if not self.catalogue:
            logger.warning(
                "Skipping Ed Discussion: no theory/practice/exam documents in the previous run's metadata. "
                "This run writes them, the next one will classify the threads"
            )
            return []

        ed_dir = self.ed_discussion_base_path / "ed_discussion" / self.academic_year

        # Intermediate JSON files
        processed_dir = ed_dir / "processed"

        # Markdown (final) files created from the intermediate JSON files (no LLM calls involved)
        markdown_dir = ed_dir / "markdown"

        # Path to the image files that are part of the exported threads
        images_dir = ed_dir / "files"

        processed_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)

        # Intermediate JSON files are useful to detect errors in the message classification
        # The errors can be manually fixed in the JSON files without having to make extra LLM calls
        intermediate_jsons = self.get_or_create_intermediate_jsons(ed_dir, processed_dir, images_dir)

        # Markdown (final) files are created from the JSON files
        resources = self.create_resources_from_jsons(intermediate_jsons, markdown_dir)

        return resources

    def get_or_create_intermediate_jsons(
        self,
        ed_dir: Path,
        processed_dir: Path,
        images_dir: Path,
    ) -> dict[str, Path]:
        """Load existing intermediate JSONs or create them."""

        intermediate_jsons = {}

        # We can force re-generation of the intermediate files
        needs_generation = self.force_regeneration

        for category in self.categories:
            json_path = processed_dir / f"ed_discussion_{category}.json"
            intermediate_jsons[category] = json_path

            # By default, if they exist the LLM would not be called again
            if not json_path.exists():
                needs_generation = True

        if needs_generation:
            self.generate_intermediate_jsons(ed_dir, processed_dir, images_dir)
        else:
            self.revisit_unmatched_threads(processed_dir)

        return intermediate_jsons

    def generate_intermediate_jsons(
        self,
        ed_dir: Path,
        processed_dir: Path,
        images_dir: Path,
    ) -> None:
        """Process raw JSON files and generate intermediate JSONs per category."""

        input_files = list(ed_dir.glob("*.json"))
        logger.info(f"Processing {len(input_files)} JSON files from {ed_dir}")

        categorized = {msg_type: [] for msg_type in MESSAGE_TYPES}
        failed_threads = []
        disputed_threads = []

        for i, json_path in enumerate(input_files, 1):
            logger.info(f"Processing {i}/{len(input_files)}: {json_path.name}")

            thread_record, disputed_entry = self.process_single_thread(json_path, images_dir)
            if disputed_entry is not None:
                disputed_threads.append(disputed_entry)
                continue

            if thread_record is None:
                # Append failed threads to the list
                failed_threads.append({"filename": json_path.name, "reason": "processing_failed"})
                continue

            thread_type = thread_record.get("type")
            if thread_type in categorized:
                categorized[thread_type].append(thread_record)
            else:
                failed_threads.append(
                    {
                        "filename": json_path.name,
                        "reason": f"unknown_type_{thread_type}",
                    }
                )

        # Save JSON files for each one of the MESSAGE_TYPES
        for msg_type in MESSAGE_TYPES:
            output_path = processed_dir / f"ed_discussion_{msg_type}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(categorized[msg_type], f, ensure_ascii=False, indent=2)

            logger.info(f"Wrote {len(categorized[msg_type])} threads to {output_path}")

        # Save JSON file with the failed threads
        if failed_threads:
            failed_path = processed_dir / "failed_threads.json"
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(failed_threads, f, ensure_ascii=False, indent=2)
            logger.warning(f"Wrote {len(failed_threads)} failed threads to {failed_path}")

        # Always write the disputed threads file, even when empty, so a run's outcome is explicit
        disputed_path = processed_dir / "disputed_threads.json"
        with open(disputed_path, "w", encoding="utf-8") as f:
            json.dump(disputed_threads, f, ensure_ascii=False, indent=2)
        if disputed_threads:
            logger.warning(f"Wrote {len(disputed_threads)} disputed threads to {disputed_path}")

    def process_single_thread(
        self,
        json_path: Path,
        images_dir: Path,
    ) -> tuple[dict | None, dict | None]:
        """Process a single thread JSON file, returning (thread_record, disputed_entry)."""

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {json_path}: {e}")
            return None, None

        thread = data.get("thread", {})
        users = data.get("users", [])
        user_roles = get_user_roles(users)

        thread_title = thread.get("title", "")

        messages = extract_messages_from_thread(
            thread=thread,
            user_roles=user_roles,
            thread_title=thread_title,
            include_student_endorsed=self.include_student_endorsed,
            images_dir=images_dir,
        )

        if len(messages) <= 1:
            logger.info(f"Skipping {json_path.name}: Thread without answers")
            return None, None

        thread_info = {
            "filename": json_path.name,
            "thread_id": thread.get("id"),
            "thread_title": thread_title,
            "thread_category": thread.get("category", ""),
            "thread_subcategory": thread.get("subcategory", ""),
            "messages": messages,
        }
        thread_record, disputed_entry = self.classify_thread(thread_info)
        return thread_record, disputed_entry

    def classify_thread(self, thread_info: dict) -> tuple[dict | None, dict | None]:
        """
        Classify a thread, returning (thread_record, disputed_entry).

        `thread_info` needs filename, thread_id, thread_title, thread_category, thread_subcategory and messages,
        so a stored thread record can be passed back in to reclassify it.
        """

        filename = thread_info["filename"]
        messages = thread_info["messages"]

        contents = []
        for message in messages:
            contents.append(message["content"])
        all_html = "\n---\n".join(contents)

        winner, votes = classify_thread_with_cascade(
            thread_title=thread_info["thread_title"],
            thread_category=thread_info["thread_category"],
            thread_subcategory=thread_info["thread_subcategory"],
            all_messages_html=all_html,
            catalogue=self.catalogue,
            academic_year=self.academic_year.replace("_", "-"),
            exam_year=self.exam_year,
            models=self.models,
        )

        real_votes = []
        for vote in votes:
            if vote is not None:
                real_votes.append(vote)

        if winner is None and not real_votes:
            logger.warning(f"Classification failed for {filename}")
            return None, None

        vote_dicts = self.votes_to_dicts(votes)

        if winner is None:
            logger.info(f"{filename}: disputed")
            disputed_entry = {
                "filename": filename,
                "thread_id": thread_info["thread_id"],
                "thread_title": thread_info["thread_title"],
                "votes": vote_dicts,
            }
            return None, disputed_entry

        self.log_cascade_outcome(filename, winner, votes)

        subtype, doc_number, doc_subnumber, week = self.resolve_classification(winner)

        thread_record = {
            "filename": filename,
            "thread_id": thread_info["thread_id"],
            "thread_title": thread_info["thread_title"],
            "type": winner.type,
            "subtype": subtype,
            "doc_number": doc_number,
            "doc_subnumber": doc_subnumber,
            "week": week,
            "thread_category": thread_info["thread_category"],
            "thread_subcategory": thread_info["thread_subcategory"],
            "messages": messages,
            "confidence": winner.confidence,
            "reason": winner.reason,
            "votes": vote_dicts,
        }
        return thread_record, None

    def is_unmatched(self, thread: dict) -> bool:
        """Whether a practice/exam thread record points at no entry of the current catalogue."""

        if thread.get("type") not in CATALOGUE_REQUIRED_TYPES:
            return False

        entry = find_entry(
            self.catalogue,
            thread.get("type"),
            thread.get("subtype"),
            thread.get("doc_number"),
            thread.get("doc_subnumber"),
        )
        has_no_entry = entry is None

        return has_no_entry

    def revisit_unmatched_threads(self, processed_dir: Path) -> None:
        """Reclassify the unmatched practice/exam threads of the extracted categories against the current catalogue."""

        threads_by_type = {}
        for msg_type in MESSAGE_TYPES:
            threads_by_type[msg_type] = load_threads(processed_dir / f"ed_discussion_{msg_type}.json")

        # Unmatched threads of categories this course does not extract are never turned into resources,
        # so reclassifying them would only cost LLM calls
        unmatched_threads = []
        for msg_type in CATALOGUE_REQUIRED_TYPES:
            if msg_type not in self.categories:
                continue

            matched_threads = []
            for thread in threads_by_type[msg_type]:
                if self.is_unmatched(thread):
                    unmatched_threads.append(thread)
                else:
                    matched_threads.append(thread)
            threads_by_type[msg_type] = matched_threads

        if not unmatched_threads:
            return

        logger.info(f"Revisiting {len(unmatched_threads)} threads with no catalogue entry")

        resolved_count = 0
        for i, thread in enumerate(unmatched_threads, 1):
            logger.info(f"Revisiting {i}/{len(unmatched_threads)}: {thread['filename']}")

            thread_record, _ = self.classify_thread(thread)

            # A disputed or failed reclassification keeps the previous record, retried at the next update
            if thread_record is None:
                thread_record = thread
            elif not self.is_unmatched(thread_record):
                resolved_count += 1

            threads_by_type[thread_record["type"]].append(thread_record)

        for msg_type in MESSAGE_TYPES:
            output_path = processed_dir / f"ed_discussion_{msg_type}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(threads_by_type[msg_type], f, ensure_ascii=False, indent=2)

        logger.info(f"{resolved_count}/{len(unmatched_threads)} revisited threads now have a catalogue entry")

    def votes_to_dicts(self, votes: list[ThreadClassification | None]) -> list[dict | None]:
        """Render each cascade vote (or None for a failed juror) into a storable dict."""

        vote_dicts = []
        for model, vote in zip(self.models, votes):
            vote_dicts.append(self.vote_to_dict(model, vote))
        return vote_dicts

    def vote_to_dict(self, model: str, vote: ThreadClassification | None) -> dict | None:
        """Render one juror's vote into a storable dict, or None when the juror failed."""

        if vote is None:
            return None

        catalogue_label = None
        if vote.catalogue_id is not None and 0 <= vote.catalogue_id < len(self.catalogue):
            catalogue_label = render_catalogue_entry(self.catalogue[vote.catalogue_id])

        return {
            "model": model,
            "type": vote.type,
            "catalogue_id": vote.catalogue_id,
            "catalogue_label": catalogue_label,
            "mentioned_number": vote.mentioned_number,
            "week": vote.week,
            "confidence": vote.confidence,
            "reason": vote.reason,
        }

    def log_cascade_outcome(
        self,
        filename: str,
        winner: ThreadClassification,
        votes: list[ThreadClassification | None],
    ) -> None:
        """Log which cascade path produced the winning classification."""

        if len(votes) == 1:
            logger.info(f"{filename}: single juror")
        elif len(votes) == 2:
            if votes[1] is None:
                logger.info(f"{filename}: juror 2 failed, kept first vote")
            else:
                logger.info(f"{filename}: juror 2 agreed")
        elif winner is votes[0]:
            logger.info(f"{filename}: juror 3 sided with 1")
        else:
            logger.info(f"{filename}: juror 3 sided with 2")

    def resolve_classification(
        self,
        classification: ThreadClassification,
    ) -> tuple[str | None, str | None, str | None, int | None]:
        """Resolve a ThreadClassification into the (subtype, doc_number, doc_subnumber, week) to store."""

        if classification.type in ("theory", "practice", "exam") and classification.catalogue_id is not None:
            entry = self.catalogue[classification.catalogue_id]
            week = entry.week if entry.week is not None else classification.week
            return entry.subtype, entry.number, entry.sub_number, week

        return None, None, None, classification.week

    def create_resources_from_jsons(
        self,
        intermediate_jsons: dict[str, Path],
        markdown_dir: Path,
    ) -> list[EdDiscussionResource]:
        """Create EdDiscussionResource instances from intermediate JSONs."""

        resources = []

        # Only create resources for the input categories
        for category in self.categories:
            json_path = intermediate_jsons.get(category)
            if json_path is None or not json_path.exists():
                logger.warning(f"Intermediate JSON not found for category: {category}")
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    threads = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {json_path}: {e}")
                continue

            for thread in threads:
                resource = self.create_resource_from_thread(thread, category, json_path, markdown_dir)
                if resource is not None:
                    resources.append(resource)

        logger.info(f"Created {len(resources)} EdDiscussionResource instances")
        return resources

    def create_resource_from_thread(
        self,
        thread: dict,
        category: str,
        json_path: Path,
        markdown_dir: Path,
    ) -> EdDiscussionResource | None:
        """Create an EdDiscussionResource from a Ed Discussion thread."""

        if self.is_unmatched(thread):
            logger.info(f"Skipping {thread.get('filename')}: no catalogue entry yet, revisited at the next update")
            return None

        messages = thread.get("messages", [])
        qa_data = extract_qa_content(messages)

        if not qa_data["question"] or not qa_data["answers"]:
            logger.info(f"Skipping {thread.get('filename')}: no question or answers")
            return None

        content = format_qa(qa_data, self.language)
        base_filename = thread.get("filename", "unknown.json")
        md_filename = base_filename.replace(".json", ".md")
        md_path = markdown_dir / md_filename

        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"Failed to write markdown {md_path}: {e}")
            return None

        thread_type = thread.get("type", category)
        thread_title = thread.get("thread_title", "")
        subtype = thread.get("subtype")
        doc_number = thread.get("doc_number")
        doc_subnumber = thread.get("doc_subnumber")
        week = thread.get("week")

        entry = find_entry(self.catalogue, thread_type, subtype, doc_number, doc_subnumber)
        if entry is not None:
            from_ = entry.from_
            until = entry.until
            week = entry.week if entry.week is not None else week
        else:
            from_ = None
            until = None
            if thread_type in ("theory", "practice", "exam") and subtype is not None:
                logger.warning(
                    f"No catalogue entry for {thread.get('filename')} ('{thread_title}'): "
                    f"type={thread_type}, subtype={subtype}, number={doc_number}, sub_number={doc_subnumber}"
                )

        return EdDiscussionResource(
            title=thread_title,
            source="ed_discussion",
            url=None,
            path=str(md_path),
            mime_type="text/markdown",
            type=thread_type,
            subtype=subtype,
            is_solution=False,
            is_qa=True,
            is_video=False,
            is_gemini_processed_video=False,
            week=week,
            number=doc_number,
            sub_number=doc_subnumber,
            from_=from_,
            until=until,
            one_chunk_per_page=False,
            one_chunk_per_doc=True,
            category=category,
            path_to_intermediate_json_file=str(json_path),
        )


def load_threads(json_path: Path) -> list[dict]:
    """Load the thread records of an intermediate JSON, or none when the file is missing."""

    if not json_path.exists():
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        threads = json.load(f)

    return threads
