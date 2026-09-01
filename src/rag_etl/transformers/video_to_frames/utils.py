import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

FFMPEG_TIMEOUT = 60


def filter_close_timestamps(timestamps: list[int], min_seconds: int) -> list[int]:
    """
    Drop slide detection timestamps that are too close to each other
    """

    kept = []
    for timestamp in sorted(timestamps):
        if kept and timestamp - kept[-1] < min_seconds:
            continue
        kept.append(timestamp)

    return kept


def frame_time(start: int, end: int, offset_seconds: int) -> int:
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
