"""Video plumbing for materialize: reaching the video behind a leaf.

Everything here is pipeline plumbing — Kaltura sessions, GraphAI calls,
config values — not business logic, which lives in ragtools
(`ragtools.video`). The transformer composes the two.
"""

from __future__ import annotations

import json
import logging

from rag_etl.config import config_value
from rag_etl.core import Resource
from ragtools.graphai import GraphAIError, detect_slides, get_access_token, retrieve_video
from ragtools.kaltura import (
    create_kaltura_session,
    extract_entry_id_from_url,
    find_slides_entry_id,
    get_video_download_url,
)
from ragtools.video import SlideDetector

logger = logging.getLogger(__name__)


class GraphAISlideDetector(SlideDetector):
    """Slide-change detection over GraphAI, behind the swappable interface."""

    def detect(self, video_url: str, language: str | None = None) -> list[int]:
        token = get_access_token()
        video_token = retrieve_video(video_url, token)
        if video_token is None:
            raise GraphAIError(f"GraphAI found no video at {video_url}")
        return detect_slides(video_token, token, language)


class KalturaVideoResolver:
    """Resolves the file url of a video entry, one session for all of them."""

    def __init__(self) -> None:
        self._client = None

    def url(self, entry_id: str) -> str | None:
        if self._client is None:
            self._client = create_kaltura_session(
                api_url=config_value("SWITCH_API_URL"),
                kaltura_app_token_id=config_value("KALTURA_APP_TOKEN_ID"),
                kaltura_user_id=config_value("KALTURA_USER_ID"),
                kaltura_token=config_value("KALTURA_TOKEN"),
                kaltura_partner_id=int(config_value("KALTURA_PARTNER_ID")),
            )

        slides_entry_id = find_slides_entry_id(self._client, entry_id)
        return get_video_download_url(self._client, slides_entry_id)


def entry_id_of(node: Resource) -> str | None:
    """
    Return the Kaltura entry id a video node stands for.

    Mediaspace videos point at a descriptor naming the entry; MOOC videos
    carry the entry id in their playManifest url.
    """

    if node.path is not None and node.path.suffix == ".json":
        try:
            descriptor = json.loads(node.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return descriptor.get("entry_id")

    if node.url:
        return extract_entry_id_from_url(node.url)

    return None
