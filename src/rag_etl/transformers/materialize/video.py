"""Video materialization: slide changes detected, one frame per segment read."""

from __future__ import annotations

import logging
import shutil

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import ragtools.mime_types as mt
from ragtools.images import SLIDE_SYSTEM_PROMPT, SLIDE_USER_PROMPT
from ragtools.video import SlideDetector, extract_frame, filter_close_timestamps, frame_time, parse_srt, text_between

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.transformers.materialize.ocr import MAX_CONCURRENT_OCR, ocr_cached
from rag_etl.transformers.materialize.videos import KalturaVideoResolver, entry_id_of
from ragtools.graphai import GraphAIError
from ragtools.kaltura import build_entry_url

logger = logging.getLogger(__name__)

# The transcript is quoted rather than put under a heading
TRANSCRIPT_PREFIX = "> "


@dataclass(frozen=True)
class VideoMaterializer:
    """Turns one video into one slide child per segment, captions appended."""

    cache: Cache
    work_root: Path
    detector: SlideDetector
    mediaspace_url: str | None = None
    min_slide_seconds: int = 40
    language: str | None = None

    def materialize(self, node: Resource, resolver: KalturaVideoResolver) -> None:
        entry_id = entry_id_of(node)
        if entry_id is None:
            # A YouTube video has no Kaltura entry to reach: its captions
            # are kept as they are, the video itself is left as it is
            logger.warning(f"Leaving {node.title} as it is: no Kaltura entry id to reach the video with")
            return

        # Frames live in the work area of the video's content
        if node.path is None:
            return
        frames_path = self.work_root / node.content_hash
        content_hash = node.content_hash

        # A crashed run can leave frames behind: they must not survive into
        # the new slides
        shutil.rmtree(frames_path, ignore_errors=True)

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
                title=f"{node.title} > {start_seconds // 60}:{start_seconds % 60:02d}",
                path=md_path,
                mime_type=mt.MARKDOWN,
                url=build_entry_url(entry_id, self.mediaspace_url, start_seconds),
                cut="whole_document",
            )

        self._merge_captions(node)

    def _ocr_frame(self, frame_path: Path) -> Path:
        return ocr_cached(
            self.cache, frame_path, SLIDE_SYSTEM_PROMPT, SLIDE_USER_PROMPT, frame_path.with_suffix(".md")
        )

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
