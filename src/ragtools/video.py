"""Videos: slide-change detection, frame extraction and subtitle handling.

A recorded lecture is turned into slides by a slide detector (the
interface below; the GraphAI implementation stays in the pipeline):
slide changes are found, one frame per slide segment is extracted with
ffmpeg straight from the remote url, and the subtitles spoken over the
segment are kept with the frame's text by whoever merges them.
"""

import logging
import re
import subprocess

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class SlideDetector(ABC):
    """Finds the slide-change moments of a video, in seconds, ascending."""

    @abstractmethod
    def detect(self, video_url: str, language: str | None = None) -> list[int]:
        raise NotImplementedError


# Time a frame is taken before the end of the slide segment it represents
FRAME_OFFSET_SECONDS = 5

FFMPEG_TIMEOUT = 60


def filter_close_timestamps(timestamps: list[int], min_seconds: int) -> list[int]:
    """
    Drop slide detection timestamps that are too close to each other
    """

    kept: list[int] = []
    for timestamp in sorted(timestamps):
        if kept and timestamp - kept[-1] < min_seconds:
            continue
        kept.append(timestamp)

    return kept


def frame_time(start: int, end: int, offset_seconds: int = FRAME_OFFSET_SECONDS) -> int:
    """
    Return the moment to grab the frame representing the interval [start, end).

    The frame is taken shortly before the interval ends, so that anything the
    lecturer added to the slide while speaking over it is present. It is
    always at least one second after the start, so a short interval still
    holds a frame inside itself.
    """

    return max(start + 1, end - offset_seconds)


def extract_frame(video_url: str, seconds: int, output_path: Path) -> bool:
    """
    Extract a single frame from a video into output_path.

    Seeking is done before the input so ffmpeg issues HTTP range requests
    instead of reading the file from the start, which keeps this at well
    under a second per frame against a remote URL.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        # The system ffmpeg, found on PATH. The static binary imageio-ffmpeg
        # used to bundle segfaults on a host whose glibc is much newer than the
        # one it was linked against, which a distribution package cannot do
        "ffmpeg",
        # Never read stdin: without this ffmpeg can block waiting for an answer
        "-nostdin",
        # Report errors only, so a successful extraction stays silent
        "-loglevel",
        "error",
        # Seek before -i: ffmpeg jumps with HTTP range requests instead of
        # decoding the video from the start, which is what makes this fast
        "-ss",
        str(seconds),
        # Read the video straight from its remote URL, without downloading it
        "-i",
        video_url,
        # Write a single frame and stop
        "-frames:v",
        "1",
        # The frame keeps the resolution of the source: scaling down loses the
        # small print on a slide, and scaling up only inflates the payload
        # JPEG quality, 2 being the best and 31 the worst
        "-q:v",
        "3",
        # The frame to write, its extension deciding the format
        str(output_path),
        # Overwrite an existing file rather than prompting
        "-y",
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
    except subprocess.CalledProcessError as error:
        logger.warning(f"ffmpeg failed at {seconds}s: {error.stderr.decode('utf-8', 'replace')[:200]}")
        return False
    except subprocess.TimeoutExpired:
        logger.warning(f"ffmpeg timed out at {seconds}s")
        return False

    return output_path.exists()


@dataclass
class Cue:
    """One subtitle cue: when it is spoken and what is said."""

    index: int
    start: float
    end: float
    text: str


def parse_timestamp(text: str) -> float:
    """Convert an SRT timestamp such as 00:31:22,740 into seconds."""

    hours, minutes, rest = text.split(":")
    seconds, milliseconds = rest.replace(".", ",").split(",")

    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def parse_srt(srt_path: str | Path) -> list[Cue]:
    """
    Parse an SRT file into cues, skipping blocks that are malformed.

    Automatically generated subtitles occasionally contain blocks without a
    timing line, which are dropped rather than failing the whole file.
    """

    text = Path(srt_path).read_text(encoding="utf-8", errors="replace").strip()

    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        match = re.match(r"([\d:,.]+)\s*-->\s*([\d:,.]+)", lines[1])
        if not match:
            continue

        content = " ".join(line.strip() for line in lines[2:])

        try:
            index = int(lines[0].strip())
        except ValueError:
            index = len(cues) + 1

        cues.append(
            Cue(
                index=index,
                start=parse_timestamp(match.group(1)),
                end=parse_timestamp(match.group(2)),
                text=content,
            )
        )

    return cues


def text_between(cues: list[Cue], start: float, end: float | None) -> str:
    """
    Return what is said between start and end, as one paragraph.

    A cue counts as inside the interval when it starts before the interval
    ends and finishes after it begins, so speech straddling a boundary is
    kept with both sides rather than lost.
    """

    spoken = []
    for cue in cues:
        if end is not None and cue.start >= end:
            continue
        if cue.end <= start:
            continue
        spoken.append(cue.text)

    return " ".join(spoken).strip()
