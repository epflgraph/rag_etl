import re
from dataclasses import dataclass
from pathlib import Path


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

    cues = []
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
