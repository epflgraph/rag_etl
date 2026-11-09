"""Video materialization: slide changes detected, one frame per segment read.

Cached as a Merkle tree, like the pdf: the leaves are the per-segment
transcripts, each stored exactly once under its frame's hash; the video's
node record, keyed by the video's entry and language, lists its ordered
slide leaves. A node hit adopts the whole subtree —
no slide detection, no frame extraction, no read; a miss detects, extracts
and falls back to the leaves, so an unchanged frame is never read again.
Either way the node is written back, so the run after a miss hits at the
top again. The frames themselves are scratch: re-derivable, never cached.
"""

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
from rag_etl.core.resource import hash_bytes, hash_text
from rag_etl.transformers.materialize.ocr import MAX_CONCURRENT_OCR, ocr_cached
from rag_etl.transformers.materialize.videos import KalturaVideoResolver, entry_id_of
from ragtools.graphai import GraphAIError
from ragtools.kaltura import build_entry_url

logger = logging.getLogger(__name__)

# The transcript is quoted rather than put under a heading
TRANSCRIPT_PREFIX = "> "

# The cache scope the slides live in. Bumped whenever detecting, extracting
# or reading changes what the same video means (detector, min slide seconds,
# frame naming, prompts, model): the new scope starts empty, so every video
# is detected and every frame read again
CACHE_VERSION = "video-slides-1"


def _node_key(entry_id: str, language: str | None) -> str:
    """The video node's key: the video's identity plus the language it is detected in.

    The descriptor is deliberately not part of it: its bytes churn on every
    re-export while the video itself stays the same, and a re-export must
    not re-detect, re-extract and re-read a lecture nobody touched.
    """
    return hash_text(f"{entry_id}|{language}")


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

        # The slides are built in the work area of the video's content
        if node.path is None:
            return
        work_dir = self.work_root / node.content_hash

        # A crashed run can leave slides behind: they must not survive into
        # the new ones
        shutil.rmtree(work_dir, ignore_errors=True)

        cache = self.cache.scope(CACHE_VERSION)
        key = _node_key(entry_id, self.language)
        md_paths = cache.adopt(key, work_dir)
        if md_paths is None:
            video_url = resolver.url(entry_id)
            if not video_url:
                logger.warning(f"Dropping {node.title}: no video file for entry {entry_id}")
                node.detach()
                return

            logger.info(f"Detecting slides in {node.title} (entry {entry_id})")

            # The service failing on one video costs that video rather than
            # the whole run
            try:
                timestamps = self.detector.detect(video_url, self.language)
            except GraphAIError as error:
                logger.warning(f"Dropping {node.title}: {error}")
                node.detach()
                return

            logger.info(f"Entry {entry_id}: {len(timestamps)} slide changes detected")

            frames = self._extract(cache, video_url, timestamps, work_dir)
            cache.write_node(key, [(md_path.name, hash_bytes(frame.read_bytes())) for frame, md_path in frames])
            md_paths = [md_path for _, md_path in frames]

        self._build_slides(node, entry_id, md_paths)

        if not any(child.mime_type == mt.MARKDOWN for child in node.children):
            logger.warning(f"Dropping {node.title}: no frames were extracted from it")
            node.detach()

    def _extract(self, cache: Cache, video_url: str, timestamps: list[int], work_dir: Path) -> list[tuple[Path, Path]]:
        """Extract one frame per interval and read it, refilling the leaves.

        The last reported timestamp is the end of the video rather than a
        slide, so it only bounds the final interval. The frames are scratch:
        read once here, never cached — the transcripts outlive them.
        """

        kept = filter_close_timestamps(timestamps, self.min_slide_seconds)

        if len(kept) < 2:
            logger.warning(f"Only {len(kept)} slide timestamps, nothing to cut into intervals")
            return []

        work_dir.mkdir(parents=True, exist_ok=True)

        frames: list[Path] = []
        for position in range(len(kept) - 1):
            start = kept[position]
            end = kept[position + 1]

            # Each frame is named after the second its interval starts at,
            # which is where the slides and their deep links are read back
            # from. Keeping it in the name means the node record is enough
            # to adopt the slides, with no extra file listing which frame
            # belongs to which moment. Six digits so that sorting the names
            # sorts them chronologically.
            frame_path = work_dir / f"{start:06d}.jpg"
            if extract_frame(video_url, frame_time(start, end), frame_path):
                frames.append(frame_path)
            else:
                logger.warning(f"Could not extract the frame for the interval starting at {start}s")

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OCR) as executor:
            md_paths = list(
                executor.map(
                    lambda frame: ocr_cached(
                        cache, frame, SLIDE_SYSTEM_PROMPT, SLIDE_USER_PROMPT, frame.with_suffix(".md")
                    ),
                    frames,
                )
            )

        return list(zip(frames, md_paths))

    def _build_slides(self, node: Resource, entry_id: str, md_paths: list[Path]) -> None:
        """Turn the read slides into one slide child each, captioned."""

        for md_path in sorted(md_paths, key=lambda path: int(path.stem)):
            if not md_path.read_text(encoding="utf-8").strip():
                logger.debug(f"Dropping {md_path.name} of {node.title}: no readable content")
                continue

            start_seconds = int(md_path.stem)

            # Each slide carries the public watch url of its own moment, so a
            # chunk cut from it links straight into that moment of the lecture
            node.add_child(
                title=f"{start_seconds // 60}:{start_seconds % 60:02d}",
                path=md_path,
                mime_type=mt.MARKDOWN,
                url=build_entry_url(entry_id, self.mediaspace_url, start_seconds),
                cut="whole_document",
            )

        self._merge_captions(node)

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
        caption.detach()
