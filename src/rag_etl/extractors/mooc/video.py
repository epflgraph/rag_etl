"""A MOOC video: the video, its subtitles in the MOOC's language as a child."""

import json
import logging

from pathlib import Path

from lxml.etree import _Element

import ragtools.mime_types as mt
from rag_etl.core import Resource
from rag_etl.extractors.mooc.export import Export

logger = logging.getLogger(__name__)


def parse_video(export: Export, element: _Element) -> Resource | None:
    """The video, titled by its own name (the unit's is its parent's title already).

    A video plays from its own source (a Mediaspace url) or else from
    YouTube. One that names neither is a broken export and stops the run.
    """

    url_name = element.get("url_name")
    xml_path = export.xml_path("video", url_name)

    block = export.load("video", url_name)
    if block is None:
        return None

    source_url = ""
    for child in block:
        if child.tag == "source":
            source_url = child.get("src")

    youtube_id = element.get("youtube_id_1_0")
    video_url = source_url or (f"https://www.youtube.com/watch?v={youtube_id}" if youtube_id else "")
    if not video_url:
        raise ValueError(f"The video {xml_path} names neither a source url nor a YouTube id")

    video = Resource(title=block.get("display_name", ""), url=video_url, path=xml_path, mime_type=mt.MP4)

    srt_path = _subtitles(block, export)
    if srt_path is not None:
        video.add_child(title="subtitles", path=srt_path, mime_type=mt.SRT)

    return video


def _subtitles(block: _Element, export: Export) -> Path | None:
    """
    The video's subtitle file in the MOOC's language, or None.

    The video descriptor names its subtitle files in its `transcripts`
    attribute (edX's word), as a JSON map of language code to file name,
    e.g. {"en": "<uuid>-en.srt", "fr": "<uuid>-fr.srt"}; the files
    themselves live in the export's static folder.
    """

    raw = block.get("transcripts")
    if not raw:
        return None

    try:
        by_language = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse transcripts attribute: '{raw}'")
        return None

    file_name = (by_language or {}).get(export.language)
    if not file_name:
        logger.debug(f"No {export.language} subtitles among {sorted(by_language or {})}, skipping them")
        return None

    srt_path = export.dir / "static" / file_name
    if not srt_path.exists():
        logger.warning(f"Subtitles {srt_path} named by the descriptor do not exist")
        return None

    return srt_path
