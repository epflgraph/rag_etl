import logging
import shutil
from datetime import date, datetime
from pathlib import Path

from rag_etl.config import CONFIG
from rag_etl.extractors.base_extractor import BaseExtractor
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
from rag_etl.resources.mediaspace_resource import MediaspaceResource
import rag_etl.utils.mime_types as mt

logger = logging.getLogger(__name__)

# The tag is only for defining the metadata fields in each course
TAG = "MEDIASPACE_VIDEO"


class MediaspaceExtractor(BaseExtractor):
    """
    Extractor for retrieving video subtitles from an EPFL Mediaspace
    (Kaltura) playlist or channel.

    One resource is produced per video entry, pointing at the subtitle
    file downloaded for it. The video itself is never downloaded: the
    subtitles already carry the text the RAG pipeline indexes.
    """

    def __init__(
        self,
        playlist_or_channel_url: str,
        mediaspace_base_path: str,
        tag_metadata: dict | None = None,
        mime_types: list[str] | None = None,
        language: str | None = None,
        created_after: str | date | datetime | float | None = None,
    ) -> None:
        self.playlist_or_channel_url = playlist_or_channel_url
        self.mediaspace_base_path = Path(mediaspace_base_path)
        self.language = language
        self.created_after = created_after

        if tag_metadata:
            self.tag_metadata = tag_metadata
        else:
            self.tag_metadata = {}

        # Kaltura publishes captions in several formats (SRT, WebVTT, DFXP)
        # We'll use SRT
        if mime_types is None:
            self.mime_types = [mt.SRT]
        else:
            self.mime_types = mime_types

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

        # Several assets can share entry and language. Mediaspace shows the
        # default one, so it is the one a student would have seen.
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
        caption_path = self.mediaspace_base_path / file_name

        return caption_path

    def extract(self) -> list[MediaspaceResource]:
        """
        Extract subtitle resources for this course from Mediaspace.

        Returns:
            list[MediaspaceResource]: List of raw Resources.
        """

        client = create_kaltura_session(
            api_url=CONFIG["SWITCH_API_URL"],
            kaltura_app_token_id=CONFIG["KALTURA_APP_TOKEN_ID"],
            kaltura_user_id=CONFIG["KALTURA_USER_ID"],
            kaltura_token=CONFIG["KALTURA_TOKEN"],
            kaltura_partner_id=int(CONFIG["KALTURA_PARTNER_ID"]),
        )

        entries = self.list_entries(client)

        # Deleting first so the folder ends up holding exactly
        # what this run extracted
        if self.mediaspace_base_path.exists():
            shutil.rmtree(self.mediaspace_base_path)

        # A channel holds one kind of material, so every resource it produces
        # carries the same metadata, declared by the course under this tag
        tag_dict = self.tag_metadata.get(TAG, {})

        resources = []
        for entry in entries:
            entry_id = entry.id
            entry_name = getattr(entry, "name", "") or ""

            # Skip if recorded before the date defined by created_after
            # We don't want repeated videos from previous editions of the course
            if not entry_created_after(entry, self.created_after):
                logger.debug(f"Skipping entry {entry_id} because it was not created after {self.created_after}.")
                continue

            # An entry without usable subtitles is ignored
            caption = self.select_caption(get_subtitle_urls(client, entry_id), entry_id)
            if caption is None:
                continue

            # A failed download skips
            caption_path = self.build_caption_path(caption, entry_id, entry_name)
            if not download_caption(client, caption, caption_path):
                continue

            logger.info(f"Entry {entry_id}: wrote {caption_path}")

            resources.append(
                MediaspaceResource(
                    title=entry_name,
                    source="mediaspace",
                    url=build_entry_url(entry_id, self.playlist_or_channel_url),
                    path=str(caption_path),
                    mime_type=caption_mime_type(caption) or mt.SRT,
                    srt_path=str(caption_path),
                    is_video=True,
                    is_gemini_processed_video=False,
                    entry_id=entry_id,
                    caption_asset_id=caption.get("caption_asset_id"),
                    tag=TAG,
                    type=tag_dict.get("type"),
                    subtype=tag_dict.get("subtype"),
                    is_solution=tag_dict.get("is_solution", False),
                    one_chunk_per_page=tag_dict.get("one_chunk_per_page", False),
                    one_chunk_per_doc=tag_dict.get("one_chunk_per_doc", False),
                )
            )

        if not resources:
            logger.warning(f"No subtitles extracted from {self.playlist_or_channel_url}")

        return resources
