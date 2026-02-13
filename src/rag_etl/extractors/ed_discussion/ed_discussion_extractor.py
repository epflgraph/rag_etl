from __future__ import annotations

import json
import logging
from pathlib import Path

from rag_etl.extractors import BaseExtractor
from rag_etl.extractors.ed_discussion.utils import (
    MESSAGE_TYPES,
    classify_thread_with_llm,
    extract_messages_from_thread,
    extract_qa_content,
    format_qa,
    get_user_roles,
)
from rag_etl.resources.ed_discussion_resource import EdDiscussionResource

logger = logging.getLogger(__name__)


class EdDiscussionExtractor(BaseExtractor):
    """Extractor for retrieving previously answered questions from Ed Discussion."""

    def __init__(
        self,
        ed_discussion_base_path: str,
        academic_year: str,
        tags: list[str],
        tag_metadata: dict,
        categories: list[str],
        language: str,
        semester: int,
        include_student_endorsed: bool,
        force_regeneration: bool = False,
        mime_types: list[str] | None = None,
    ) -> None:
        self.ed_discussion_base_path = Path(ed_discussion_base_path)
        self.academic_year = academic_year
        self.tags = tags
        self.tag_metadata = tag_metadata
        self.categories = categories
        self.language = language
        self.semester = semester
        self.include_student_endorsed = include_student_endorsed
        self.force_regeneration = force_regeneration
        self.mime_types = mime_types

        self.exam_year = self.compute_exam_year()
        self.subtype_options = self.build_subtype_options()

    def build_subtype_options(self) -> str:
        """Build subtype options string for LLM prompt from tag_metadata."""

        type_to_subtypes = {}
        for tag in self.tags:
            tag_config = self.tag_metadata.get(tag, {})
            tag_type = tag_config.get("type")
            tag_subtype = tag_config.get("subtype")
            if tag_type and tag_subtype:
                if tag_type not in type_to_subtypes:
                    type_to_subtypes[tag_type] = []
                if tag_subtype not in type_to_subtypes[tag_type]:
                    type_to_subtypes[tag_type].append(tag_subtype)

        lines = []
        for type_name, subtypes in type_to_subtypes.items():
            subtypes_str = ", ".join(subtypes)
            lines.append(f"- For {type_name}: {subtypes_str}")
        return "\n".join(lines)

    def compute_exam_year(self) -> str:
        """Compute exam year based on academic_year and semester."""

        years = self.academic_year.split("_")

        # For Spring semester (2) return second year from the academic year
        if len(years) == 2:
            return years[1] if self.semester == 2 else years[0]
        return self.academic_year

    def extract(self) -> list[EdDiscussionResource]:
        """Extract resources for Ed Discussion Q&A threads."""

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
        intermediate_jsons = self.get_or_create_intermediate_jsons(
            ed_dir, processed_dir, images_dir
        )

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

        for i, json_path in enumerate(input_files, 1):
            logger.info(f"Processing {i}/{len(input_files)}: {json_path.name}")

            thread_record = self.process_single_thread(json_path, images_dir)
            if thread_record is None:
                # Append failed threads to the list
                failed_threads.append(
                    {"filename": json_path.name, "reason": "processing_failed"}
                )
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
            logger.warning(
                f"Wrote {len(failed_threads)} failed threads to {failed_path}"
            )

    def process_single_thread(
        self,
        json_path: Path,
        images_dir: Path,
    ) -> dict | None:
        """Process a single thread JSON file."""

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {json_path}: {e}")
            return None

        thread = data.get("thread", {})
        users = data.get("users", [])
        user_roles = get_user_roles(users)

        thread_id = thread.get("id")
        thread_category = thread.get("category", "")
        thread_subcategory = thread.get("subcategory", "")
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
            return None

        all_html = "\n---\n".join(e["content"] for e in messages)
        classification = classify_thread_with_llm(
            thread_category=thread_category,
            thread_subcategory=thread_subcategory,
            all_messages_html=all_html,
            all_types=MESSAGE_TYPES,
            subtype_options=self.subtype_options,
        )

        if classification is None:
            logger.warning(f"Classification failed for {json_path.name}")
            return None

        return {
            "filename": json_path.name,
            "thread_id": thread_id,
            "thread_title": thread_title,
            "type": classification.get("type"),
            "subtype": classification.get("subtype"),
            "doc_number": classification.get("doc_number"),
            "doc_subnumber": classification.get("doc_subnumber"),
            "week": classification.get("week"),
            "thread_category": thread_category,
            "thread_subcategory": thread_subcategory,
            "messages": messages,
        }

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
                resource = self.create_resource_from_thread(
                    thread, category, json_path, markdown_dir
                )
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

        if thread_type == "exam" and not doc_number:
            doc_number = self.exam_year

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
            from_=None,
            until=None,
            one_chunk_per_page=False,
            one_chunk_per_doc=True,
            category=category,
            path_to_intermediate_json_file=str(json_path),
        )
