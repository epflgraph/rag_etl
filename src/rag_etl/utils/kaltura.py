import hashlib
import logging
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
from KalturaClient import KalturaClient, KalturaConfiguration
from KalturaClient.exceptions import KalturaException
from KalturaClient.Plugins.Caption import KalturaCaptionAssetFilter, KalturaCaptionAssetStatus, KalturaCaptionType

from rag_etl.config import CONFIG
import rag_etl.utils.mime_types as mt
from KalturaClient.Plugins.Core import (
    KalturaBaseEntryFilter,
    KalturaFilterPager,
    KalturaFlavorAssetFilter,
    KalturaFlavorAssetStatus,
    KalturaMediaEntryFilter,
    KalturaSessionType,
)

logger = logging.getLogger(__name__)

# Lifetime in seconds of the Kaltura session token.
# It has to outlive the whole extraction run of a whole Mediaspace channel
# Set to 8 hours
SESSION_EXPIRY = 28800

# How many entries we ask Kaltura per request
PAGE_SIZE = 500
DOWNLOAD_SRT_TIMEOUT = 60


def is_kaltura_id(text: str) -> bool:
    """
    True when text has the shape of a Kaltura entry/playlist id.

    An id is the partner id, an underscore and a short hash, as in
    "0_5odi1mzf", so it splits in two at the only underscore it contains.
    """

    partner_id, separator, hash_part = text.partition("_")

    has_separator = bool(separator)
    has_valid_partner_id = partner_id.isdecimal()
    has_valid_hash = hash_part.isalnum()

    return has_separator and has_valid_partner_id and has_valid_hash


def enum_value(value):
    """
    Unwrap a Kaltura enum object to its plain value.

    Fields such as language, languageCode and format come back as wrapper
    objects (KalturaLanguage, KalturaCaptionType, ...) rather than strings,
    so comparing or formatting them directly yields the repr of the object.
    """

    if hasattr(value, "getValue"):
        return value.getValue()

    return value


def is_playlist_url(url_or_id: str) -> bool:
    """True when the given URL points at a Mediaspace playlist rather than a channel."""

    is_playlist = "/playlist/" in str(url_or_id)

    return is_playlist


def extract_playlist_id_from_url(url_or_id: str) -> str:
    """
    Pull the Kaltura playlist id out of a Mediaspace playlist URL.

    A playlist URL carries the partner id, the playlist id and the entry the
    player opens on:
      https://mediaspace.epfl.ch/playlist/dedicated/30437/0_5odi1mzf/0_qvj7hkcu

    Kaltura ids look like "0_5odi1mzf" (partner prefix, underscore, hash), so
    the first segment of that shape is the playlist; the trailing one is the
    entry. Also accepts a ?playlistId=... query parameter and a bare id.
    """

    text = str(url_or_id).strip()

    if is_kaltura_id(text):
        return text

    parsed = urlparse(text)
    query = parse_qs(parsed.query)

    for key in ("playlistId", "playlist_id", "playlistid"):
        if query.get(key):
            return query[key][0]

    segments = []
    for segment in parsed.path.split("/"):
        if segment:
            segments.append(segment)

    for segment in segments:
        if is_kaltura_id(segment):
            return segment

    raise ValueError(f"Could not find a playlist id in '{text}'")


def extract_category_id_from_url(url_or_id: str) -> str:
    """
    Pull the Kaltura category id out of a Mediaspace channel URL.

    A channel URL ends with the category id, after the (possibly
    percent-encoded) channel name:
      https://mediaspace.epfl.ch/channel/MATH-310%2BAlgebra/30044

    Also accepts /channel/30044, a ?categoryId=... query parameter, and a
    bare id, so a caller can pass either form.
    """

    text = str(url_or_id).strip()

    if text.isdigit():
        return text

    parsed = urlparse(text)
    query = parse_qs(parsed.query)

    for key in ("categoryId", "category_id", "categoryid"):
        if query.get(key):
            return query[key][0]

    # The id is the last purely numeric path segment; the channel name can
    # itself contain digits, so trailing-segment order matters here.
    segments = []
    for segment in parsed.path.split("/"):
        if segment:
            segments.append(segment)

    for segment in reversed(segments):
        if segment.isdigit():
            return segment

    raise ValueError(f"Could not find a category id in '{text}'")


def extract_entry_id_from_url(url: str) -> str | None:
    """
    Pull the Kaltura entry id out of any URL that embeds one.

    MOOC video descriptors point at a playManifest such as
      https://api.cast.switch.ch/p/113/sp/11300/playManifest/entryId/0_0caer0s6/...
    so the id follows an "entryId" path segment.
    """

    segments = str(url).split("/")

    # The id is the segment right after the one reading "entryId", so the last
    # segment is never a candidate and is left out of the walk
    for position, segment in enumerate(segments[:-1]):
        next_segment = segments[position + 1]
        if segment == "entryId" and is_kaltura_id(next_segment):
            return next_segment

    return None


def build_entry_url(entry_id: str, base_url: str | None = None, start_seconds: int | None = None) -> str:
    """
    Build the public Mediaspace watch URL of an entry, optionally starting at
    a given moment.

    The /media/t/<entryId> route is the page the player renders
    base_url may be any URL on Mediaspace (a channel or playlist URL), and
    DEFAULT_MEDIASPACE_BASE_URL is used when it names no host
    """

    parsed = urlparse(str(base_url or "").strip())

    if parsed.scheme and parsed.netloc:
        instance_url = f"{parsed.scheme}://{parsed.netloc}"
    else:
        instance_url = CONFIG["DEFAULT_MEDIASPACE_BASE_URL"]

    url = f"{instance_url}/media/t/{entry_id}"

    if start_seconds is not None:
        url = f"{url}?kalturaStartTime={int(start_seconds)}"

    return url


def url_with_ks(url: str, ks: str) -> str:
    """
    Append the Kaltura session to a caption serve URL.

    Restricted entries return 403 on the bare URL, and captionAsset.getUrl
    does not embed the KS itself.
    """

    if not ks:
        return url

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    if "ks" in query:
        return url

    query["ks"] = [ks]

    encoded_query = urlencode(query, doseq=True)
    signed_url = urlunparse(parsed._replace(query=encoded_query))

    return signed_url


def caption_mime_type(caption_info: dict) -> str | None:
    """
    Return the mime type of a caption dict from get_subtitle_urls.

    Kaltura reports the format as a string enum ("1" for SRT), but some
    assets only carry a meaningful fileExt, so both are consulted. Returns
    None for the formats Kaltura also publishes (WebVTT, DFXP), which no
    course asks for, and which callers therefore filter out.
    """

    by_format = {str(KalturaCaptionType.SRT): mt.SRT}

    mime_type = by_format.get(str(caption_info.get("format")))
    if mime_type:
        return mime_type

    by_extension = {"srt": mt.SRT}

    file_ext = (caption_info.get("file_ext") or "").lower().lstrip(".")

    return by_extension.get(file_ext)


def create_kaltura_session(
    api_url: str,
    kaltura_app_token_id: str,
    kaltura_user_id: str,
    kaltura_token: str,
    kaltura_partner_id: int,
) -> KalturaClient:
    """
    Open an admin Kaltura session from an application token.

    The app token cannot be used directly: it is hashed together with an
    anonymous widget session, and that hash buys the admin session that the
    rest of the calls ride on.
    """

    config = KalturaConfiguration(kaltura_partner_id)
    config.serviceUrl = api_url
    client = KalturaClient(config)

    widget_id = f"_{kaltura_partner_id}"
    widget_session = client.session.startWidgetSession(widget_id, SESSION_EXPIRY)
    client.setKs(widget_session.ks)

    token_hash = hashlib.sha256(widget_session.ks.encode("ascii") + kaltura_token.encode("ascii")).hexdigest()

    session = client.appToken.startSession(
        kaltura_app_token_id,
        token_hash,
        kaltura_user_id,
        KalturaSessionType.ADMIN,
        SESSION_EXPIRY,
    )
    client.setKs(session.ks)

    return client


def list_playlist_entries(client: KalturaClient, playlist_id: str) -> list:
    """
    Return all Kaltura entries in a playlist.

    playlist.execute returns an array, not a list response, so we page until
    a page returns fewer than PAGE_SIZE items.
    """

    entries = []
    page_index = 1

    while True:
        pager = KalturaFilterPager()
        pager.pageSize = PAGE_SIZE
        pager.pageIndex = page_index

        page_entries = client.playlist.execute(playlist_id, "1", pager=pager)

        if not page_entries:
            break

        entries.extend(page_entries)

        if len(page_entries) < PAGE_SIZE:
            break

        page_index += 1

    return entries


def list_channel_entries(client: KalturaClient, category_id: str) -> list:
    """
    Return all media entries belonging to a Mediaspace channel (category).

    media.list is a real list response with a totalCount, but it is paged the
    same way as the playlist helper for consistency.
    """

    entries = []
    page_index = 1

    while True:
        entry_filter = KalturaMediaEntryFilter()
        entry_filter.categoriesIdsMatchOr = str(category_id)

        pager = KalturaFilterPager()
        pager.pageSize = PAGE_SIZE
        pager.pageIndex = page_index

        result = client.media.list(entry_filter, pager)
        page_entries = result.objects or []

        if not page_entries:
            break

        entries.extend(page_entries)

        if len(page_entries) < PAGE_SIZE or len(entries) >= result.totalCount:
            break

        page_index += 1

    return entries


def get_subtitle_urls(client: KalturaClient, entry_id: str) -> list[dict]:
    """
    Return the ready caption assets attached to a Kaltura entry, as plain
    dicts with their download URL resolved.

    A failure to reach the caption service is reported and treated as "no
    captions", so a single unreachable entry does not cost the whole run.
    """

    caption_filter = KalturaCaptionAssetFilter()
    caption_filter.entryIdEqual = entry_id
    caption_filter.statusEqual = KalturaCaptionAssetStatus.READY

    try:
        caption_list = client.caption.captionAsset.list(caption_filter, None)
    except Exception as e:
        logger.warning(f"Could not list caption assets for entry {entry_id}: {e}")
        return []

    captions = []
    for caption in caption_list.objects or []:
        try:
            url = client.caption.captionAsset.getUrl(caption.id)
        except Exception as e:
            logger.warning(f"Could not get URL for caption asset {caption.id}: {e}")
            url = None

        captions.append(
            {
                "caption_asset_id": caption.id,
                "entry_id": caption.entryId,
                "label": getattr(caption, "label", None),
                "language": enum_value(getattr(caption, "language", None)),
                "language_code": enum_value(getattr(caption, "languageCode", None)),
                "format": enum_value(getattr(caption, "format", None)),
                "file_ext": getattr(caption, "fileExt", None),
                "is_default": getattr(caption, "isDefault", None),
                "status": enum_value(getattr(caption, "status", None)),
                "url": url,
            }
        )

    return captions


def download_caption(client: KalturaClient, caption_info: dict, output_path: Path) -> bool:
    """
    Download one caption asset to output_path, creating parent folders.

    Returns True when the file was written, False when the asset has no URL
    or the download failed.
    """

    url = caption_info.get("url")
    if not url:
        logger.warning(f"Caption asset {caption_info.get('caption_asset_id')} has no URL")
        return False

    try:
        response = requests.get(url_with_ks(url, client.getKs()), timeout=DOWNLOAD_SRT_TIMEOUT)
        response.raise_for_status()
    except Exception as e:
        logger.warning(f"Could not download caption {caption_info.get('caption_asset_id')}: {e}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    return True


def original_flavor(client: KalturaClient, entry_id: str):
    """
    Return the entry's original (source) flavor asset, or None.

    A flavor is Kaltura's name for one encoded version of an entry, so a
    single recording carries several: the file as uploaded, plus the lower
    resolution transcodes made from it for streaming. The original is the one
    flagged isOriginal, and it is the one worth reading, since text projected
    on a slide survives it best.

    Falls back to the best transcode when the entry was published without its
    original upload.
    """

    flavor_filter = KalturaFlavorAssetFilter()
    flavor_filter.entryIdEqual = entry_id

    ready = []
    for asset in client.flavorAsset.list(flavor_filter, None).objects or []:
        if enum_value(getattr(asset, "status", None)) == KalturaFlavorAssetStatus.READY:
            ready.append(asset)

    if not ready:
        return None

    for asset in ready:
        if getattr(asset, "isOriginal", False):
            return asset

    # This entry was published without its original upload, so the best
    # available transcode is used instead: the one with the highest bitrate,
    # and on a tie the largest file
    best_asset = None
    best_quality = (0, 0)

    for asset in ready:
        bitrate = getattr(asset, "bitrate", 0) or 0
        size = getattr(asset, "size", 0) or 0
        quality = (bitrate, size)

        if best_asset is None or quality > best_quality:
            best_asset = asset
            best_quality = quality

    return best_asset


def bits_per_pixel(asset) -> float:
    """
    Bitrate normalised by frame area, used to differentiate a slide from a
    camera auditorium shot within one recording

    A slide capture is almost static, so bitrate is lower for the screen recording
    stream than for the camera auditorium stream

    The value is only meaningful relative to the other stream of the same
    recording: a slide can sit as high as a camera in absolute terms
    """

    width = getattr(asset, "width", 0) or 0
    height = getattr(asset, "height", 0) or 0
    bitrate = getattr(asset, "bitrate", 0) or 0

    if not width or not height:
        return float("inf")

    bits_per_second = bitrate * 1000
    pixels_per_frame = width * height

    return bits_per_second / pixels_per_frame


def find_slides_entry_id(client: KalturaClient, entry_id: str) -> str:
    """
    Return the entry that carries the slides for a dual-stream recording.

    An auditorium recording is published as two Kaltura entries, one filming
    the room and one capturing the projected slides. Which of the two is the
    parent is not consistent. The streams are separated by bits per pixel instead,
    the one with lower bitrate is the one that contains the slides

    Returns the entry id itself when the recording is single-stream, which is
    the case for MOOC screencasts.
    """

    entry_filter = KalturaBaseEntryFilter()
    entry_filter.parentEntryIdEqual = entry_id

    children = client.baseEntry.list(entry_filter, None).objects or []

    if not children:
        return entry_id

    candidates = [entry_id]
    for child in children:
        candidates.append(child.id)

    best_id = None
    best_score = float("inf")
    for candidate in candidates:
        asset = original_flavor(client, candidate)
        if asset is None:
            logger.warning(f"Entry {candidate} has no usable flavor, skipping")
            continue

        score = bits_per_pixel(asset)
        logger.debug(f"Entry {candidate}: {score:.3f} bits/pixel")

        if score < best_score:
            best_score = score
            best_id = candidate

    if best_id is None:
        return entry_id

    logger.info(f"Entry {entry_id} is dual-stream, using {best_id} ({best_score:.3f} bits/pixel) for slides")

    return best_id


def get_video_download_url(client: KalturaClient, entry_id: str) -> str | None:
    """
    Return a direct serveFlavor URL for the entry's video file.

    entry.downloadUrl is a playManifest that only redirects, so callers that
    need to read the file itself (frame extraction, slide detection) need the
    concrete rendition this returns.
    """

    asset = original_flavor(client, entry_id)

    if asset is None:
        logger.error(f"Entry {entry_id} has no ready video flavor")
        return None

    # getUrl raises instead of returning nothing when the flavor has no file
    try:
        download_url = client.flavorAsset.getUrl(asset.id)
    except KalturaException as error:
        logger.error(f"Entry {entry_id} has no file for flavor {asset.id}: {error}")
        return None

    return download_url
