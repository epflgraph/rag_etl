from lxml.etree import _Element
from pathlib import Path

import json
import logging

from rag_etl.resources.mooc_resource import MOOCResource
from rag_etl.extractors.mooc.utils import load_root_elem_from_mooc_xml
from rag_etl.utils.kaltura import extract_entry_id_from_url
import rag_etl.utils.mime_types as mt

logger = logging.getLogger(__name__)


class VideoParser:
    """
    Video Parser for MOOCs.
    """

    def get_video_platform_id(self, video_url: str, youtube_id: str) -> str:
        if video_url != "":
            video_platform = "mediaspace"
        elif youtube_id != "":
            video_platform = "youtube"
        return video_platform

    def find_transcript(self, root_video: _Element, course_path: str, language: str | None) -> str | None:
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

        file_name = transcripts.get(language)

        if not file_name:
            logger.debug(f"No {language} transcript among {sorted(transcripts)}, skipping subtitles for this video")
            return None

        transcript_path = Path(course_path) / "static" / file_name
        if not transcript_path.exists():
            logger.warning(f"Transcript {transcript_path} named by the descriptor does not exist")
            return None

        return str(transcript_path)

    def parse(
        self,
        course_path: str,
        elem_vertical: _Element,
        vertical_display_name: str,
        tag_metadata: dict,
        language: str | None = None,
    ) -> MOOCResource | None:
        """Parse a MOOC video"""

        video_url_name = elem_vertical.get("url_name")
        youtube_id = elem_vertical.get("youtube_id_1_0")
        youtube_url = ""
        if youtube_id:
            youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"

        video_filename = video_url_name + ".xml"
        video_xml_path = Path(course_path) / "video" / video_filename

        root_video = load_root_elem_from_mooc_xml(video_xml_path)
        if root_video is None:
            return None

        try:
            video_title = root_video.get("display_name")
        except Exception:
            video_title = ""

        switch_video_url = ""
        for elem in root_video:
            if elem.tag == "source":
                switch_video_url = elem.get("src")

        entry_id = extract_entry_id_from_url(switch_video_url)
        srt_path = self.find_transcript(root_video, course_path, language)

        video_platform = self.get_video_platform_id(video_url=switch_video_url, youtube_id=youtube_id)

        video_url = ""
        if video_platform == "mediaspace":
            video_url = switch_video_url
        elif video_platform == "youtube":
            video_url = youtube_url
        else:
            return None

        tag_name = "MOOC_VIDEO"
        tag_dict = tag_metadata.get(tag_name)

        mooc_resource_title = vertical_display_name + " - " + video_title
        return MOOCResource(
            source="mooc",
            title=mooc_resource_title,
            url=video_url,
            path=str(video_xml_path),
            mime_type=mt.MP4,
            is_video=True,
            is_gemini_processed_video=tag_dict.get("is_gemini_processed_video", False),
            srt_path=srt_path,
            entry_id=entry_id,
            tag=tag_name,
            type=tag_dict.get("type"),
            subtype=tag_dict.get("subtype"),
            processing_method=tag_dict.get("processing_method"),
            model=tag_dict.get("model"),
            vertical=vertical_display_name,
        )
