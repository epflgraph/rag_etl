"""Materialize: files that hold text in pixels are turned into text.

Every PDF page is rendered deterministically and read by a vision model,
one child per page; images are read the same way; videos are turned into
slides — slide changes are detected, one frame per slide segment is
extracted straight from the video's url, every frame is read, and the
words spoken over the segment are appended to it.

Everything read by a model is cached by content hash: a page that changed
is read again, a page that did not is not.
"""

from __future__ import annotations

import logging

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ragtools.mime_types as mt
from ragtools.images import SLIDE_SYSTEM_PROMPT, SLIDE_USER_PROMPT, ocr_image
from ragtools.llm import RCPClient
from ragtools.pdf import PDF_SYSTEM_PROMPT, PDF_USER_PROMPT, save_pdf_pages
from ragtools.video import SlideDetector, extract_frame, filter_close_timestamps, frame_time, parse_srt, text_between

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.core.resource import hash_bytes
from rag_etl.transformers.base import Transformer
from rag_etl.transformers.materialize.videos import KalturaVideoResolver, entry_id_of
from rag_etl.utils.kaltura import build_entry_url
from rag_etl.utils.graphai import GraphAIError

logger = logging.getLogger(__name__)

# The transcript is quoted rather than put under a heading
TRANSCRIPT_PREFIX = "> "

# Page and frame reads run bounded: twenty concurrent calls is what kept
# the old pipeline well behaved against RCP
MAX_CONCURRENT_OCR = 20


def child_inheriting(node: Resource, **overrides: Any) -> Resource:
    """One child of a node, carrying the parent's descriptive metadata."""
    return Resource(
        type=node.type,
        subtype=node.subtype,
        number=node.number,
        subnumber=node.subnumber,
        week=node.week,
        is_solution=node.is_solution,
        from_=node.from_,
        until=node.until,
        **overrides,
    )


@dataclass(frozen=True)
class MaterializeTransformer(Transformer):
    """Turns files that hold text in pixels into text files:

    - every PDF page is rendered and read, the pdf gaining one child per page,
    - images are read the same way, in place,
    - videos gain one slide child per slide segment, captions appended.
    """

    cache: Cache
    llm: RCPClient
    vision_model: str
    detector: SlideDetector
    mediaspace_url: str | None = None
    min_slide_seconds: int = 40
    language: str | None = None

    def transform(self, roots: list[Resource]) -> list[Resource]:
        resolver = KalturaVideoResolver()

        for root in roots:
            for node in list(root.walk()):
                # A pdf is materialized once, when it has no pages yet; a
                # video once, when it has no slides yet; an image by mime,
                # since it is read in place
                if node.mime_type == mt.PDF and node.path is not None and node.is_leaf:
                    self._materialize_pdf(node)
                elif node.mime_type in (mt.JPEG, mt.PNG) and node.path is not None:
                    self._materialize_image(node)
                elif node.mime_type == mt.MP4 and not any(child.mime_type == mt.MARKDOWN for child in node.children):
                    self._materialize_video(node, resolver)

        return roots

    def _materialize_pdf(self, node: Resource) -> None:
        pdf_path = node.path
        if pdf_path is None:
            return

        # Pages live in a folder named after the pdf, like archives do
        pages_dir = pdf_path.with_suffix("")
        page_paths = save_pdf_pages(pdf_path, pages_dir)

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OCR) as executor:
            md_paths = list(executor.map(self._ocr_pdf_page, page_paths))

        for page_path, md_path in zip(page_paths, md_paths):
            if not md_path.read_text(encoding="utf-8").strip():
                logger.debug(f"Dropping {page_path.name} of {node.title}: no readable content")
                continue

            page_number = int(page_path.stem.removeprefix("page_"))
            node.add_child(
                child_inheriting(
                    node,
                    title=f"{node.title} > page {page_number}",
                    path=md_path,
                    mime_type=mt.MARKDOWN,
                )
            )

        logger.info(f"{node.title}: {len(node.children)} pages")

    def _ocr_pdf_page(self, page_path: Path) -> Path:
        return self._ocr_file(page_path, system=PDF_SYSTEM_PROMPT, user=PDF_USER_PROMPT)

    def _ocr_file(self, image_path: Path, system: str, user: str) -> Path:
        """Read one image into the markdown file next to it, cached by content."""
        md_path = image_path.with_suffix(".md")
        content_hash = hash_bytes(image_path.read_bytes())

        if not self.cache.get(content_hash, md_path):
            text = ocr_image(image_path, self.llm, self.vision_model, system=system, user=user)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(text, encoding="utf-8")
            self.cache.set(content_hash, md_path)

        return md_path

    def _materialize_image(self, node: Resource) -> None:
        image_path = node.path
        if image_path is None:
            return

        md_path = self._ocr_file(image_path, system=SLIDE_SYSTEM_PROMPT, user=SLIDE_USER_PROMPT)

        if not md_path.read_text(encoding="utf-8").strip():
            logger.info(f"Dropping {node.title}: no readable content on it")
            if node.parent is not None:
                node.parent.children.remove(node)
            return

        node.path = md_path
        node.mime_type = mt.MARKDOWN

    def _materialize_video(self, node: Resource, resolver: KalturaVideoResolver) -> None:
        entry_id = entry_id_of(node)
        if entry_id is None:
            # A YouTube video has no Kaltura entry to reach: its captions
            # are kept as they are, the video itself is left as it is
            logger.warning(f"Leaving {node.title} as it is: no Kaltura entry id to reach the video with")
            return

        # Frames live in a folder named after the file the video stands for
        if node.path is None:
            return
        frames_path = node.path.with_suffix("")
        content_hash = node.content_hash

        if not self.cache.get(content_hash, frames_path):
            video_url = resolver.url(entry_id)
            if not video_url:
                logger.warning(f"Dropping {node.title}: no video file for entry {entry_id}")
                if node.parent is not None:
                    node.parent.children.remove(node)
                return

            logger.info(f"Detecting slides in {node.title} (entry {entry_id})")

            # The service failing on one video costs that video rather than
            # the whole run
            try:
                timestamps = self.detector.detect(video_url, self.language)
            except GraphAIError as error:
                logger.warning(f"Dropping {node.title}: {error}")
                if node.parent is not None:
                    node.parent.children.remove(node)
                return

            logger.info(f"Entry {entry_id}: {len(timestamps)} slide changes detected")

            self._write_frames(video_url, timestamps, frames_path)
            self.cache.set(content_hash, frames_path)

        self._build_slides(node, entry_id, frames_path)

        if not any(child.mime_type == mt.MARKDOWN for child in node.children):
            logger.warning(f"Dropping {node.title}: no frames were extracted from it")
            if node.parent is not None:
                node.parent.children.remove(node)

    def _write_frames(self, video_url: str, timestamps: list[int], frames_path: Path) -> None:
        """Extract one frame per interval into frames_path.

        The last reported timestamp is the end of the video rather than a
        slide, so it only bounds the final interval.
        """

        kept = filter_close_timestamps(timestamps, self.min_slide_seconds)

        if len(kept) < 2:
            logger.warning(f"Only {len(kept)} slide timestamps, nothing to cut into intervals")
            return

        frames_path.mkdir(parents=True, exist_ok=True)

        for position in range(len(kept) - 1):
            start = kept[position]
            end = kept[position + 1]

            # Each frame is named after the second its interval starts at,
            # which is where the slides and their deep links are read back
            # from. Keeping it in the name means the cached folder is enough
            # on its own, with no extra file listing which frame belongs to
            # which moment. Six digits so that sorting the names sorts them
            # chronologically.
            output_path = frames_path / f"{start:06d}.jpg"
            if not extract_frame(video_url, frame_time(start, end), output_path):
                logger.warning(f"Could not extract the frame for the interval starting at {start}s")

    def _build_slides(self, node: Resource, entry_id: str, frames_path: Path) -> None:
        """Turn the frames on disk into one slide child each, read and captioned."""

        frame_paths = sorted(frames_path.glob("*.jpg"))

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OCR) as executor:
            md_paths = list(executor.map(self._ocr_frame, frame_paths))

        for frame_path, md_path in zip(frame_paths, md_paths):
            if not md_path.read_text(encoding="utf-8").strip():
                logger.debug(f"Dropping {frame_path.name} of {node.title}: no readable content")
                continue

            start_seconds = int(frame_path.stem)

            # Each slide carries the public watch url of its own moment, so a
            # chunk cut from it links straight into that moment of the lecture
            node.add_child(
                child_inheriting(
                    node,
                    title=f"{node.title} > {start_seconds // 60}:{start_seconds % 60:02d}",
                    path=md_path,
                    mime_type=mt.MARKDOWN,
                    url=build_entry_url(entry_id, self.mediaspace_url, start_seconds),
                    cut="whole_document",
                )
            )

        self._merge_captions(node)

    def _ocr_frame(self, frame_path: Path) -> Path:
        return self._ocr_file(frame_path, system=SLIDE_SYSTEM_PROMPT, user=SLIDE_USER_PROMPT)

    def _merge_captions(self, node: Resource) -> None:
        """Append the words spoken over each slide to it.

        Slides carry only what is written on them, which is the scaffold of
        the lecture rather than its substance; appending the matching stretch
        of transcript puts both in the same chunk, so a question answered out
        loud retrieves the slide it was answered on.
        """

        caption = next((child for child in node.children if child.mime_type == mt.SRT), None)
        if caption is None or caption.path is None:
            return

        try:
            cues = parse_srt(caption.path)
        except OSError as error:
            # Without the transcript there is nothing to append, so the
            # slides keep the text generated by the vision model
            logger.warning(f"Could not read {caption.path}: {error}. Leaving its slides unchanged.")
            return

        # Chronological order is what lets each slide be bounded by the next;
        # slides are named after the second their interval starts at
        slide_paths = sorted(
            (child.path for child in node.children if child is not caption and child.path is not None),
            key=lambda path: int(path.stem),
        )

        for position, md_path in enumerate(slide_paths):
            # A slide runs from its own start to the start of the next one.
            # The last slide has no successor, so it runs to the end of the
            # recording, which text_between reads as an open interval
            start = int(md_path.stem)
            if position + 1 < len(slide_paths):
                end = int(slide_paths[position + 1].stem)
            else:
                end = None

            # Everything said while this slide was on screen
            spoken = text_between(cues, start, end)

            if not spoken:
                # A silent stretch, so the slide is left as it is rather
                # than gaining an empty heading
                logger.debug(f"Nothing spoken over {md_path.name}")
                continue

            # The Markdown is rewritten in place, so a second run over an
            # already-merged file must not append the transcript twice
            slide_markdown = md_path.read_text(encoding="utf-8").rstrip()
            starts_quoted = slide_markdown.startswith(TRANSCRIPT_PREFIX)
            contains_quote = f"\n{TRANSCRIPT_PREFIX}" in slide_markdown

            if not starts_quoted and not contains_quote:
                md_path.write_text(
                    f"{slide_markdown}\n\n{TRANSCRIPT_PREFIX}{spoken}\n",
                    encoding="utf-8",
                )
        # The subtitles are now part of the slides, and the file they came
        # from is never loaded, so the caption leaf only leaves a reference
        # that resolves nowhere
        node.children.remove(caption)
