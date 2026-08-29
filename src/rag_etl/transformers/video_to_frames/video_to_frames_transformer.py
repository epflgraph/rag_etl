import logging
from collections.abc import Sequence
from pathlib import Path

from rag_etl.config import CONFIG
from rag_etl.resources import BaseResource
from rag_etl.transformers.base_transformer import BaseTransformer
from rag_etl.transformers.video_to_frames.utils import (
    extract_frame,
    filter_close_timestamps,
    frame_time,
)
from rag_etl.utils.graphai import detect_slides, get_access_token, retrieve_video
from rag_etl.utils.kaltura import build_entry_url, create_kaltura_session, find_slides_entry_id, get_video_download_url
import rag_etl.utils.mime_types as mt

logger = logging.getLogger(__name__)


class VideoToFramesTransformer(BaseTransformer):
    """
    Transformer that turns a video resource into one image resource per slide.

    Slide change timestamps come from GraphAI; the frames themselves are cut
    out of the video with ffmpeg, seeking over HTTP so the video is never
    downloaded. Each frame carries the public watch URL of its own moment, so
    the Markdown produced from it later links straight into that moment of the lecture.

    The video resource is replaced by its frames.
    """

    def __init__(
        self,
        type_subtypes=None,
        min_slide_seconds: int = 10,
        frame_offset_seconds: int = 5,
        language: str | None = None,
        mediaspace_url: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.type_subtypes = type_subtypes
        self.min_slide_seconds = min_slide_seconds
        self.frame_offset_seconds = frame_offset_seconds
        self.language = language
        # Any URL on the Mediaspace instance the links should point at. A MOOC
        # video's own url is a playManifest on the streaming host, which is not
        # a page a student can open, so it cannot serve as the base.
        self.mediaspace_url = mediaspace_url

    def slide_detected_timestamps(self, video_url: str) -> list[int]:
        """Return the slide change timestamps GraphAI reports for a video."""

        token = get_access_token()
        video_token = retrieve_video(video_url, token)

        return detect_slides(video_token, token, self.language)

    def write_frames(self, video_url: str, timestamps: list[int], frames_path: Path) -> None:
        """
        Extract one frame per interval into frames_path.

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

            # Each frame is named after the second its interval starts at, which
            # is where build_frame_resources reads the timestamp back from.
            # Keeping it in the name means the cached folder is enough on its own,
            # with no extra file listing which frame belongs to which moment.
            # Six digits so that sorting the names sorts them chronologically.
            output_path = frames_path / f"{start:06d}.jpg"
            if not extract_frame(video_url, frame_time(start, end, self.frame_offset_seconds), output_path):
                logger.warning(f"Could not extract the frame for the interval starting at {start}s")

    def build_frame_resources(self, resource: BaseResource, frames_path: Path) -> list[BaseResource]:
        """Turn the frames on disk into one resource each, in chronological order."""

        frame_resources = []

        for frame_path in sorted(frames_path.glob("*.jpg")):
            # The file name is the interval start in seconds, zero padded
            start_seconds = int(frame_path.stem)

            url_to_link = build_entry_url(resource.entry_id, self.mediaspace_url, start_seconds)
            # is_video set to True
            # TODO: Double check its effect on the pipeline
            frame_resources.append(
                resource.copy_with(
                    title=f"{resource.title} > {start_seconds // 60}:{start_seconds % 60:02d}",
                    path=str(frame_path),
                    mime_type=mt.JPEG,
                    url=url_to_link,
                    number=str(start_seconds),
                    sub_number=str(start_seconds),
                    processing_method="rcp",
                    model=CONFIG["RCP_VISION_MODEL"],
                    is_gemini_processed_video=False,
                    one_chunk_per_doc=True,
                )
            )

        return frame_resources

    def transform(self, resources: Sequence[BaseResource]) -> list[BaseResource]:
        """
        Replace every video resource by one image resource per detected slide.
        """

        transformed_resources: list[BaseResource] = []

        client = None

        for resource in resources:
            # Skip if resource is not in the specified list of types and subtypes
            if self.type_subtypes is not None and (resource.type, resource.subtype) not in self.type_subtypes:
                transformed_resources.append(resource)
                continue

            # Skip if resource is not a video
            if not resource.is_video:
                transformed_resources.append(resource)
                continue

            entry_id = getattr(resource, "entry_id", None)
            if not entry_id:
                logger.warning(f"Video {resource.title} has no Kaltura entry id, leaving it unchanged")
                transformed_resources.append(resource)
                continue

            # Frames live in a folder named after the resource that produced them
            frames_path = Path(resource.path).with_suffix("")

            cached = self.get_from_cache(resource.path, frames_path)
            if not cached:
                if client is None:
                    client = create_kaltura_session(
                        api_url=CONFIG["SWITCH_API_URL"],
                        kaltura_app_token_id=CONFIG["KALTURA_APP_TOKEN_ID"],
                        kaltura_user_id=CONFIG["KALTURA_USER_ID"],
                        kaltura_token=CONFIG["KALTURA_TOKEN"],
                        kaltura_partner_id=int(CONFIG["KALTURA_PARTNER_ID"]),
                    )

                slides_entry_id = find_slides_entry_id(client, entry_id)
                video_url = get_video_download_url(client, slides_entry_id)

                if not video_url:
                    logger.warning(f"No video file for entry {slides_entry_id}, leaving {resource.title} unchanged")
                    transformed_resources.append(resource)
                    continue

                logger.info(f"Detecting slides in {resource.title} (entry {slides_entry_id})")
                timestamps = self.slide_detected_timestamps(video_url)
                logger.info(f"Entry {slides_entry_id}: {len(timestamps)} slide changes detected")

                # Filter out close in time timestamps
                # and write frames
                self.write_frames(video_url, timestamps, frames_path)
                self.set_to_cache(resource.path, frames_path)

            frame_resources = self.build_frame_resources(resource, frames_path)

            if not frame_resources:
                logger.warning(f"No frames extracted from {resource.title}, leaving it unchanged")
                transformed_resources.append(resource)
                continue

            logger.info(f"{resource.title}: {len(frame_resources)} slides")
            transformed_resources.extend(frame_resources)

        return transformed_resources
