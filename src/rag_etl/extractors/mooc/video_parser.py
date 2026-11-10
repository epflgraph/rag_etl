from lxml.etree import _Element
from pathlib import Path

import json
import logging

from rag_etl.core import Resource
from rag_etl.extractors.mooc.utils import Export
import ragtools.mime_types as mt

logger = logging.getLogger(__name__)


class VideoParser:
    """
    Video Parser for MOOCs.
    """

    def find_transcript(self, root_video: _Element, export: Export) -> Path | None:
        """
        Return the path of the video's subtitle file in the MOOC export.

        The video descriptor names its transcripts as a JSON map of language
        code to file name, e.g. {"en": "<uuid>-en.srt", "fr": "<uuid>-fr.srt"},
        and the files themselves live in the export's static folder.
        """

        raw = root_video.get("transcripts")
        if not raw:
            return None

        try:
            transcripts = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse transcripts attribute: '{raw}'")
            return None

        if not transcripts:
            return None

        file_name = transcripts.get(export.language)

        if not file_name:
            logger.debug(f"No {export.language} transcript among {sorted(transcripts)}, skipping subtitles for this video")
            return None

        transcript_path = export.dir / "static" / file_name
        if not transcript_path.exists():
            logger.warning(f"Transcript {transcript_path} named by the descriptor does not exist")
            return None

        return transcript_path

    def parse(
        self,
        export: Export,
        elem_vertical: _Element,
        vertical_display_name: str,
        week: int | None = None,
    ) -> Resource | None:
        """Parse a MOOC video: the video itself, its subtitles as a child.

        A video plays from its own source (a Mediaspace url) or else from
        YouTube. One that names neither is a broken export and stops the run.
        """

        video_url_name = elem_vertical.get("url_name")
        video_xml_path = export.xml_path("video", video_url_name)

        root_video = export.load("video", video_url_name)
        if root_video is None:
            return None

        source_url = ""
        for elem in root_video:
            if elem.tag == "source":
                source_url = elem.get("src")

        youtube_id = elem_vertical.get("youtube_id_1_0")
        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}" if youtube_id else ""

        video_url = source_url or youtube_url
        if not video_url:
            raise ValueError(f"The video {video_xml_path} names neither a source url nor a YouTube id")

        video_title = root_video.get("display_name", "")

        # The week is inherited by every slide cut from this video, while the
        # numbering itself is not: a slide is identified by its timestamp
        video = Resource(
            # Its own name: the unit's is its parent's title already
            title=video_title,
            url=video_url,
            path=video_xml_path,
            mime_type=mt.MP4,
            week=week,
        )

        # A transcript found in the export travels beside the video as a caption leaf
        srt_path = self.find_transcript(root_video, export)
        if srt_path is not None:
            video.add_child(title="subtitles", path=srt_path, mime_type=mt.SRT, cut=None)

        return video
