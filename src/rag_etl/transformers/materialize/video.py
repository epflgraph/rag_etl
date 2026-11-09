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
from ragtools.video import (
    Cue,
    SlideDetector,
    extract_frame,
    filter_close_timestamps,
    frame_time,
    parse_srt,
    text_between,
)

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
    """The video's key, for its cache node and its work folder: its entry plus the detection language."""
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

        key = _node_key(entry_id, self.language)
        work_dir = self.work_root / key

        # A crashed run can leave slides behind: they must not survive into
        # the new ones
        shutil.rmtree(work_dir, ignore_errors=True)

        cache = self.cache.scope(CACHE_VERSION)
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
        """Turn the read frames into one slide child each, with what was said over it.

        Slides carry only what is written on them, which is the scaffold of
        the lecture rather than its substance; appending the words spoken
        over each slide puts both in the same chunk, so a question answered
        out loud retrieves the slide it was answered on.
        """
        cues = self._take_subtitles(node)

        # Frames are named after the second their slide starts at; a frame
        # that reads to nothing is dropped, its neighbour's span covering it
        slides: list[tuple[int, Path, str]] = []
        for md_path in sorted(md_paths, key=lambda path: int(path.stem)):
            markdown = md_path.read_text(encoding="utf-8").strip()
            if markdown:
                slides.append((int(md_path.stem), md_path, markdown))
            else:
                logger.debug(f"Dropping {md_path.name} of {node.title}: no readable content")

        for position, (start, md_path, markdown) in enumerate(slides):
            # A slide runs until the next one starts; the last one runs to the
            # end of the recording, which text_between reads as open
            end = slides[position + 1][0] if position + 1 < len(slides) else None
            spoken = text_between(cues, start, end)
            if spoken:
                md_path.write_text(f"{markdown}\n\n{TRANSCRIPT_PREFIX}{spoken}\n", encoding="utf-8")

            # Each slide carries the public watch url of its own moment, so a
            # chunk cut from it links straight into that moment of the lecture
            node.add_child(
                title=f"{start // 60}:{start % 60:02d}",
                path=md_path,
                mime_type=mt.MARKDOWN,
                url=build_entry_url(entry_id, self.mediaspace_url, start),
                cut="whole_document",
            )

    def _take_subtitles(self, node: Resource) -> list[Cue]:
        """Detach the video's subtitles leaf and return its cues; none when absent or unreadable.

        The subtitles end up inside the slides, and their file is never
        loaded itself, so the leaf goes either way.
        """
        subtitles = next((child for child in node.children if child.mime_type == mt.SRT), None)
        if subtitles is None:
            return []

        subtitles.detach()
        if subtitles.path is None:
            return []

        try:
            return parse_srt(subtitles.path)
        except OSError as error:
            # Without the subtitles the slides keep what the vision model read
            logger.warning(f"Could not read {subtitles.path}: {error}. Leaving its slides unchanged.")
            return []
