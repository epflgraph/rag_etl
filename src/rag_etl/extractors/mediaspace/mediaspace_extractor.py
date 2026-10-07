import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from rag_etl.config import CONFIG
from rag_etl.core import Resource
from rag_etl.extractors.base_extractor import Extractor, SourceUnavailable
from rag_etl.extractors.mediaspace.utils import (
    caption_matches_language,
    entry_created_after,
    safe_filename,
)
from rag_etl.utils.kaltura import (
    build_entry_url,
    caption_mime_type,
    create_kaltura_session,
    download_caption,
    extract_category_id_from_url,
    extract_playlist_id_from_url,
    get_subtitle_urls,
    is_playlist_url,
    list_channel_entries,
    list_playlist_entries,
)
import rag_etl.utils.mime_types as mt

logger = logging.getLogger(__name__)


def config_value(key: str) -> str:
    """Return a required config value, failing loud when it is missing."""
    value = CONFIG.get(key)
    if not value:
        raise SourceUnavailable(f"Missing required config value: {key}")
    return value


@dataclass(frozen=True)
class MediaspaceExtractor(Extractor):
    """
    Extractor for retrieving videos from an EPFL Mediaspace (Kaltura)
    playlist or channel.

    One resource is produced per video entry, pointing at a small descriptor
    naming the entry. The video itself is never downloaded.
    VideoToFramesTransformer reads it over HTTP

    Subtitles are downloaded (when available) and travel beside the video as
    caption leaves.

    Note that mime_types filters caption assets. The resources themselves are
    videos and carry mt.MP4
    """

    playlist_or_channel_url: str
    type: str | None = None
    subtype: str | None = None
    is_solution: bool = False
    cut: str | None = None
    language: str | None = None
    created_after: str | date | datetime | float | None = None
    mime_types: Sequence[str] = (mt.SRT,)

    def list_entries(self, client) -> list:
        """Return the entries of the configured playlist or channel."""

        if is_playlist_url(self.playlist_or_channel_url):
            playlist_id = extract_playlist_id_from_url(self.playlist_or_channel_url)
            entries = list_playlist_entries(client=client, playlist_id=playlist_id)
            logger.info(f"Playlist {playlist_id}: {len(entries)} entries")
            return entries

        category_id = extract_category_id_from_url(self.playlist_or_channel_url)
        entries = list_channel_entries(client=client, category_id=category_id)
        logger.info(f"Channel {category_id}: {len(entries)} entries")
        return entries

    def select_caption(self, captions: list[dict], entry_id: str) -> dict | None:
        """
        Return the single caption to keep for an entry, or None.

        Each filter reports only when it is the step that emptied the list,
        so an entry with no captions at all stays a single line of output.
        """

        if not captions:
            logger.warning(f"Entry {entry_id}: no caption assets")
            return None

        # An entry usually carries the same subtitles in several formats
        typed_captions = []
        for caption in captions:
            if caption_mime_type(caption) in self.mime_types:
                typed_captions.append(caption)

        if not typed_captions:
            logger.warning(f"Entry {entry_id}: no caption assets of an expected type ({self.mime_types})")
            return None

        # And in several languages, of which the course wants one
        matching_captions = []
        for caption in typed_captions:
            if caption_matches_language(caption, self.language):
                matching_captions.append(caption)

        if not matching_captions:
            logger.warning(f"Entry {entry_id}: no caption assets in language {self.language}")
            return None

        if len(matching_captions) == 1:
            return matching_captions[0]

        # More than one caption, we keep the one marked asisDefault
        selected = matching_captions[0]
        for caption in matching_captions:
            if caption.get("is_default"):
                selected = caption
                break

        # Name the discarded ones, so a wrong pick can be traced from the logs
        ignored_ids = []
        for caption in matching_captions:
            if caption is not selected:
                ignored_ids.append(caption.get("caption_asset_id"))

        logger.warning(
            f"Entry {entry_id}: {len(matching_captions)} matching caption assets, "
            f"keeping {selected.get('caption_asset_id')} and ignoring {ignored_ids}"
        )

        return selected

    def build_caption_path(self, caption_info: dict, entry_id: str, entry_name: str) -> Path:
        """Build the path the caption of an entry is downloaded to."""

        file_ext = (caption_info.get("file_ext") or "srt").lstrip(".")

        # Kaltura fills either field depending on the entry, so both are tried
        language = caption_info.get("language_code") or caption_info.get("language") or ""

        # The entry id leads, since it is the only part guaranteed unique
        name_parts = [entry_id]
        if entry_name:
            name_parts.append(safe_filename(entry_name))
        if language:
            name_parts.append(safe_filename(str(language), max_length=16))

        file_stem = "_".join(name_parts)
        file_name = f"{file_stem}.{file_ext}"
        caption_path = self.dir / file_name

        return caption_path

    def build_descriptor_path(self, entry_id: str, entry_name: str) -> Path:
        """Build the path of the file standing for an entry's video."""

        # Same shape as build_caption_path, minus the language, so an entry's
        # descriptor and its subtitle sit next to each other under one name
        name_parts = [entry_id]
        if entry_name:
            name_parts.append(safe_filename(entry_name))

        file_stem = "_".join(name_parts)
        descriptor_path = self.dir / f"{file_stem}.json"

        return descriptor_path

    def write_descriptor(self, entry_id: str, entry_name: str) -> Path:
        """
        Write the file an entry's resource points at, and return its path.

        It holds the entry id and nothing else on purpose. Transformers key
        their cache on the bytes of the file a resource points at.
        """

        descriptor_path = self.build_descriptor_path(entry_id, entry_name)
        descriptor_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor_path.write_text(json.dumps({"entry_id": entry_id}), encoding="utf-8")

        return descriptor_path

    def extract(self) -> Resource:
        """
        Extract video resources for this course from Mediaspace into a
        container of video leaves, each carrying its captions beside it.
        """

        # Reaching the source, failing loud when it cannot be reached
        try:
            client = create_kaltura_session(
                api_url=config_value("SWITCH_API_URL"),
                kaltura_app_token_id=config_value("KALTURA_APP_TOKEN_ID"),
                kaltura_user_id=config_value("KALTURA_USER_ID"),
                kaltura_token=config_value("KALTURA_TOKEN"),
                kaltura_partner_id=int(config_value("KALTURA_PARTNER_ID")),
            )
        except Exception as error:
            raise SourceUnavailable(f"Kaltura session could not be created: {error}") from error

        try:
            entries = self.list_entries(client)
        except Exception as error:
            raise SourceUnavailable(f"Could not list Mediaspace entries: {error}") from error

        # Deleting first so the folder ends up holding exactly
        # what this run extracted
        if self.dir.exists():
            shutil.rmtree(self.dir)

        resources = Resource(title=self.playlist_or_channel_url, url=self.playlist_or_channel_url)

        for entry in entries:
            entry_id = entry.id
            entry_name = getattr(entry, "name", "") or ""

            # Skip if recorded before the date defined by created_after
            # We don't want repeated videos from previous editions of the course
            if not entry_created_after(entry, self.created_after):
                logger.debug(f"Skipping entry {entry_id} because it was not created after {self.created_after}.")
                continue

            # An entry the recording system created but never filled has no
            # file behind it, so there is nothing to cut frames from
            duration = getattr(entry, "duration", 0) or 0
            if duration <= 0:
                logger.warning(f"Skipping entry {entry_id} because it has no duration")
                continue

            # Subtitles are optional: a video published without them, or
            # without them in this course's language, still uses its extracted slide frames
            caption = self.select_caption(get_subtitle_urls(client, entry_id), entry_id)

            srt_path = None

            if caption is not None:
                caption_path = self.build_caption_path(caption, entry_id, entry_name)

                # A failed download leaves the video without a transcript
                # rather than dropping it
                if download_caption(client, caption, caption_path):
                    logger.info(f"Entry {entry_id}: wrote {caption_path}")
                    srt_path = str(caption_path)

            descriptor_path = self.write_descriptor(entry_id, entry_name)

            video = resources.add_child(
                Resource(
                    title=entry_name,
                    url=build_entry_url(entry_id, self.playlist_or_channel_url),
                    path=descriptor_path,
                    mime_type=mt.MP4,
                    type=self.type,
                    subtype=self.subtype,
                    is_solution=self.is_solution,
                    cut=self.cut,
                )
            )

            # Subtitles travel beside the video as a caption leaf
            if srt_path is not None:
                video.add_child(Resource(path=Path(srt_path), mime_type=mt.SRT))

        if not resources.children:
            logger.warning(f"No videos extracted from {self.playlist_or_channel_url}")

        return resources
