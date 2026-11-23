"""A MOOC page (an html item): the page as the export holds it, and the documents it links."""

import logging
import unicodedata

from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from bs4 import BeautifulSoup
from lxml.etree import _Element

import ragtools.mime_types as mt
from rag_etl.core import Resource
from rag_etl.extractors.markers import NO_BOT
from rag_etl.extractors.mooc.export import Export, get_filename_via_assets, url_exists

logger = logging.getLogger(__name__)

# The documents a page links that are extracted beside it
LINKED_DOCUMENTS = (".pdf", ".txt", ".zip", ".md")


def parse_page(export: Export, element: _Element, vertical_title: str, vertical_has_video: bool) -> list[Resource]:
    """The page and the documents it links, as facts: titles, files, urls.

    What the material is, the judge decides. A page marked [NO_BOT] (in its
    name, its unit's or its text) is left out with everything it links; a
    link whose paragraph is marked is left out alone. The page goes as the
    export holds it: normalize turns HTML into Markdown.
    """

    url_name = element.get("url_name", "")
    block = export.load("html", url_name)
    if block is None:
        return []

    title = block.get("display_name", "")
    html_path = export.dir / "html" / f"{url_name}.html"

    try:
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    except OSError as error:
        logger.warning(f"Could not read the page {html_path}: {error}")
        return []

    if NO_BOT.search(f"{vertical_title} {title}") or NO_BOT.search(soup.get_text()):
        logger.info(f"Skipping the page {title!r} because it is marked NO_BOT")
        return []

    page = Resource(title=title, path=html_path, mime_type=mt.HTML)

    # The documents a page holding a video links are left out on purpose:
    # they are that video's slide deck, which the video already brings in
    # slide by slide, each linking to its own moment, so the deck would be
    # indexed twice. A choice about duplicates, not a fact of the export:
    # revisit it if a course links other material beside its videos
    if vertical_has_video:
        return [page]

    return [page, *_linked_documents(export, soup)]


def _linked_documents(export: Export, soup: BeautifulSoup) -> list[Resource]:
    documents = []
    for href in _links(soup):
        path, url = _resolve(href, export)
        if not path.exists():
            # A resource pointing at a file that is not there only fails
            # later, in whichever transformer opens it first
            logger.warning(f"Missing asset: href={href!r} resolved={path}")
            continue

        documents.append(
            Resource(
                title=unicodedata.normalize("NFC", path.name),
                url=url,
                path=path,
                mime_type=mt.guess_mime_type(path),
            )
        )
    return documents


def _links(soup: BeautifulSoup) -> list[str]:
    """Every document the page links, in page order, once each, but those in a NO_BOT paragraph.

    The same href appears twice in a paragraph whenever an icon and a
    label were made into two links.
    """
    hrefs: dict[str, None] = {}
    for paragraph in soup.find_all("p"):
        excluded = bool(NO_BOT.search(paragraph.get_text()))
        for anchor in paragraph.find_all("a"):
            href = anchor.get("href")
            if not isinstance(href, str) or not href.lower().split("?")[0].endswith(LINKED_DOCUMENTS):
                continue
            if excluded:
                logger.info(f"Skipping {href} because it is marked NO_BOT")
                continue
            hrefs.setdefault(href, None)
    return list(hrefs)


def _resolve(href: str, export: Export) -> tuple[Path, str | None]:
    """Where a linked file is in the export, and the url it is published at."""

    url = _asset_url(href, export.asset_base_url)

    linked = "/static/" + href.split("block@")[-1] if "block@" in href else href

    # Some pages encode their links twice, so unquoting is repeated until
    # it stops changing anything
    previous = None
    while linked != previous:
        previous, linked = linked, unquote(linked)

    path = export.dir / Path(linked).relative_to("/") if linked.startswith("/") else None
    if path is None or not path.exists():
        path = get_filename_via_assets(export.dir, linked, export.assets_map)

    return path, url


def _asset_url(href: str, asset_base_url: str | None) -> str | None:
    """
    Return the public url of a linked file, or None.

    Pages link their files both ways: some write the full published url,
    others only "/static/name". The second kind is published under the same
    prefix, so the name is enough to name it.
    """

    # An archive is a way of handing files over rather than something to
    # read, and everything unpacked from it inherits this url, so linking
    # to it would send a reader a whole download instead of the file
    if href.lower().split("?")[0].endswith(".zip"):
        return None

    # A url the page states is taken at its word
    if "block@" in href:
        return href

    if not asset_base_url:
        return None

    # One unquote undoes the double encoding some pages carry, and leaves a
    # name that is still encoded once, which is what a url needs
    inferred_url = f"{asset_base_url}block@{PurePosixPath(unquote(href)).name}"

    # The name a file has in the export is not always the name it was
    # published under, and some files were never published at all, so an
    # inferred url is only kept once it is known to resolve
    return inferred_url if url_exists(inferred_url) else None
