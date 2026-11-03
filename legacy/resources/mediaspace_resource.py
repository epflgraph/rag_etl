from dataclasses import dataclass

from rag_etl.resources.base_resource import BaseResource


@dataclass
class MediaspaceResource(BaseResource):
    source = "mediaspace"

    # Kaltura id of the video this came from, as in "0_072xwsyi"
    # It is the handle later stages use to reach the video again
    entry_id: str | None = None

    # Kaltura id of the subtitle file that was downloaded, for tracing which
    # of an entry's several caption assets was picked
    caption_asset_id: str | None = None

    # Key into the course's tag_metadata that supplied this resource's fields
    tag: str | None = None
