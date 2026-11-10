import logging
from pathlib import Path
from lxml.etree import _Element
from pathlib import PurePosixPath
from urllib.parse import unquote
import re
import unicodedata

from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString


from rag_etl.core import Resource
from rag_etl.extractors.mooc.utils import (
    Export,
    clean_text,
    get_filename_via_assets,
    url_exists,
)

from ragtools.encoding import sanitize_for_filename
from rag_etl.extractors.markers import NO_BOT, split_tag_number_text, split_tag_text

import ragtools.mime_types as mt


logger = logging.getLogger(__name__)

# A tag as it is written into a page, as in "[MOOC_EXERCICE_6_SOLUTION]": a
# capital and at least one more capital, digit or underscore, so citations
# ("[1]") and single letters ("[A]") are not taken for one. Spaces are allowed
# because a tag split across two HTML elements gains one when they are joined
TAG_MARKER = re.compile(r"\[[A-Z][A-Z_0-9 ]+\]")


class HtmlParser:
    """
    HTML Parser for MOOCs.
    """

    def convert_html_text_to_markdown(self, soup: BeautifulSoup) -> str:
        """Convert HTML text to Markdown with BeautifulSoup"""

        self.remove_noise(soup)
        self.remove_empty_paragraphs(soup)

        out: list[str] = []
        root = soup.body or soup
        self.walk(root, out)

        return "\n".join(out).strip()

    def remove_noise(self, soup: BeautifulSoup) -> None:
        """Remove HTML noise with BeautifulSoup"""

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        for tag in soup.select("nav#menu, nav#im"):
            tag.decompose()

    def remove_empty_paragraphs(self, soup: BeautifulSoup) -> None:
        """Remove Empty paragraphs with BeautifulSoup"""

        for p in soup.find_all("p"):
            text = p.get_text().replace("\xa0", " ").strip()
            if not text:
                p.decompose()

    def inline_text(self, node: Tag) -> str:
        """
        From Node to text.
        """

        parts: list[str] = []

        for child in node.children:
            # If child is plain text (NavigableString in BeautifulSoup)
            if isinstance(child, NavigableString):
                # Clean text
                t = clean_text(str(child))

                # If it's not empty, append
                if t:
                    parts.append(t)

            # If child is another HTML tag
            elif isinstance(child, Tag):
                # Clean it
                t = clean_text(child.get_text(" ", strip=True))
                if not t:
                    continue

                # If Bold HTML, convert to Markdown
                if child.name in {"strong", "b"}:
                    parts.append(f"**{t}**")
                else:
                    parts.append(t)

        # Join all parts and clean any starting and ending white spaces
        return " ".join(parts).strip()

    def walk(self, node, out: list[str]) -> None:
        """Walk through the HTML structure and try to convert it to Markdown"""

        # Ignore plain text
        if isinstance(node, NavigableString):
            return

        # Ignore non-HTML tags
        if not isinstance(node, Tag):
            return

        # HTML Tag name
        name = node.name

        # HTML Tags to Markdown
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            text = clean_text(node.get_text(" ", strip=True))
            if text:
                out.append("#" * level + " " + text)
                out.append("")
            return

        if name == "p":
            text = self.inline_text(node)
            if text:
                out.append(text)
                out.append("")
            return

        if name == "br":
            out.append("")
            return

        if name == "ul":
            for li in node.find_all("li", recursive=False):
                self.walk(li, out)
            out.append("")
            return

        if name == "ol":
            i = 1
            for li in node.find_all("li", recursive=False):
                text = clean_text(li.get_text(" ", strip=True))
                if text:
                    out.append(f"{i}. {text}")
                    i += 1
            out.append("")
            return

        if name == "li":
            text = clean_text(node.get_text(" ", strip=True))
            if text:
                out.append(f"- {text}")
            return

        # For any child HTML tag not handled above
        for child in node.children:
            self.walk(child, out)

    def tagged_links(self, soup: BeautifulSoup, extensions: tuple[str, ...]) -> list[tuple[str, str | None]]:
        """
        Return every document linked from the page, with the tag written beside it.

        A page lists several files under one heading and tags each of them
        separately, so the tag is read from the paragraph the link sits in
        rather than from the page. The same href appears twice in a paragraph
        whenever the tag and the label were made into two links, so it is
        returned once.

        Returns:
            list[tuple[str, str | None]]: (href, raw tag) in page order, the
            tag being None for a link its paragraph does not tag.
        """

        tagged: dict[str, str | None] = {}

        for paragraph in soup.find_all("p"):
            tag, _ = split_tag_text(paragraph.get_text())

            for anchor in paragraph.find_all("a"):
                href = anchor.get("href")
                if not isinstance(href, str) or not href:
                    continue

                if not href.lower().split("?")[0].endswith(extensions):
                    continue

                if href not in tagged:
                    tagged[href] = tag

        return list(tagged.items())

    def asset_url(self, href: str, asset_base_url: str | None) -> str | None:
        """
        Return the public url of a linked file, or None.

        Pages link their files both ways: some write the full published url,
        others only "/static/name". The second kind is published under the same
        prefix, so the name is enough to name it.

        An archive never gets one, whichever way it was linked.
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
        file_name = PurePosixPath(unquote(href)).name
        inferred_url = f"{asset_base_url}block@{file_name}"

        # The name a file has in the export is not always the name it was
        # published under, and some files were never published at all, so an
        # inferred url is only kept once it is known to resolve
        if not url_exists(inferred_url):
            return None

        return inferred_url

    def strip_tag_markers(self, md_text: str) -> str:
        """
        Remove the tag markers a page carries from the text taken out of it.

        A marker says how to file the thing it sits next to; it is not something
        a student reads, and left in place it is indexed as course content and
        read back by later stages as if it were a heading. A marker is capitals,
        digits and underscores, so citations ([1]) and bracketed prose survive.
        """

        for marker in set(TAG_MARKER.findall(md_text)):
            md_text = md_text.replace(marker, "")

        # Taking a marker out of a line can leave the space it sat in
        return "\n".join(line.strip() for line in md_text.splitlines()).strip()

    def resolve_link(self, linked: str, export: Export) -> tuple[Path, str | None]:
        """Return where a linked file is in the export, and the url it is published at."""

        resource_url = self.asset_url(linked, export.asset_base_url)

        if "block@" in linked:
            linked = "/static/" + linked.split("block@")[-1]

        # Some pages encode their links twice, so unquoting is repeated until
        # it stops changing anything
        previous = None
        while linked != previous:
            previous, linked = linked, unquote(linked)

        relative_path = Path(linked)
        resource_path = export.dir / relative_path.relative_to("/") if linked.startswith("/") else None
        if resource_path is None or not resource_path.exists():
            resource_path = get_filename_via_assets(export.dir, linked, export.assets_map)

        return resource_path, resource_url

    def parse(
        self,
        export: Export,
        elem_vertical: _Element,
        vertical_display_name: str,
        vertical_has_video: bool = False,
        week: int | None = None,
    ) -> list[Resource]:
        """Parse a MOOC HTML page: the page itself and the documents it links.

        Only facts are recorded: titles, files, urls, the week and the number
        a tag marker carries. What the material is (type, subtype, cut) is the
        judge's to decide, corrected by the project's overrides. A page or a
        link marked [NO_BOT] is left out.
        """

        mooc_resources: list[Resource] = []

        html_url_name = elem_vertical.get("url_name", "")
        html_path = export.dir / "html" / f"{html_url_name}.html"

        root_html = export.load("html", html_url_name)
        if root_html is None:
            return []

        html_display_name = root_html.get("display_name", "")

        # The page is titled by its own name; its unit's name is its parent's
        # title already. A marker, though, may sit in either name, so the two
        # are read together for it
        _, _, page_title = split_tag_number_text(html_display_name)
        module_tag, module_number, names = split_tag_number_text(vertical_display_name + " - " + html_display_name)

        # Extract text from HTML
        try:
            html_text = html_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Exception when reading HTML file %s: %s", html_path, e)
            return []

        soup = BeautifulSoup(html_text, "html.parser")

        if not module_tag:
            plain_text = soup.get_text()

            # Look for tags inside the HTML
            # Only the tag is taken from the text. html_text stays the raw
            # HTML, because the links below live in its href attributes and
            # the plain text has none of them
            module_tag, module_number, _ = split_tag_number_text(plain_text)

            logger.debug(f"inside html module_tag={module_tag}")
            logger.debug(f"inside html module_number={module_number}")

        # A page marked NO_BOT is not for the bots, nor are the documents it links
        if NO_BOT.search(names) or NO_BOT.search(soup.get_text()):
            logger.info(f"Skipping the page {html_display_name!r} because it is marked NO_BOT")
            return []

        # HTML to MarkDown
        md_text = self.convert_html_text_to_markdown(soup)
        md_text = self.strip_tag_markers(md_text)

        # Write MarkDown to file in the same HTML folder, under a name the filesystem takes
        markdown_path = export.dir / "html" / sanitize_for_filename(f"{html_url_name}.md")

        Path(markdown_path).write_text(md_text, encoding="utf-8")
        logger.debug(
            "markdown_path exists? %s (%s)",
            Path(markdown_path).exists(),
            repr(markdown_path),
        )

        mooc_resources.append(
            Resource(
                title=page_title,
                path=markdown_path,
                mime_type=mt.guess_mime_type(str(markdown_path)),
                number=str(module_number) if module_number is not None else None,
                week=week,
            )
        )

        # Note: the documents a page holding a video links are left out on
        # purpose. They are that video's slide deck, which the video already
        # brings in slide by slide, each linking to its own moment, so the
        # deck would be indexed twice. This is a choice about duplicates, not
        # a fact of the export: revisit it if a course links other material
        # beside its videos
        if vertical_has_video:
            return mooc_resources

        # Every linked document, each with the tag marker of its own paragraph
        for linked, link_tag in self.tagged_links(soup, (".pdf", ".txt", ".zip", ".md")):
            if link_tag and NO_BOT.search(f"[{link_tag}]"):
                logger.info(f"Skipping {linked} because it is marked NO_BOT")
                continue

            resource_path, resource_url = self.resolve_link(linked, export)
            if not resource_path.exists():
                # A resource pointing at a file that is not there only fails
                # later, in whichever transformer opens it first
                logger.warning(f"Missing asset: href={linked!r} resolved={resource_path}")
                continue

            # A link's own marker carries its number, so one page can list
            # several numbered documents; otherwise the page's number holds
            _, link_number, _ = split_tag_number_text(f"[{link_tag}]") if link_tag else (None, module_number, None)

            mooc_resources.append(
                Resource(
                    title=unicodedata.normalize("NFC", resource_path.name),
                    url=resource_url,
                    path=resource_path,
                    mime_type=mt.guess_mime_type(resource_path),
                    number=str(link_number) if link_number is not None else None,
                    week=week,
                )
            )

        return mooc_resources
