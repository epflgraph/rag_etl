import logging
from pathlib import Path
from lxml.etree import _Element
from pathlib import PurePosixPath
from urllib.parse import unquote, urljoin
import re
import unicodedata
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString


from rag_etl.resources.mooc_resource import MOOCResource
from rag_etl.extractors.mooc.utils import (
    clean_text,
    load_root_elem_from_mooc_xml,
    get_filename_via_assets,
    UntaggedDocuments,
    pdf_is_slides,
    url_exists,
)

from rag_etl.utils.tags import split_tag_number_text, split_tag_text
from rag_etl.utils import sanitize_for_filename

import rag_etl.utils.mime_types as mt


logger = logging.getLogger(__name__)


class HtmlParser:
    """
    HTML Parser for MOOCs.
    """

    pdf_default_processing_method = "rcp"
    pdf_default_model = "Qwen/Qwen3-VL-235B-A22B-Thinking-fp8"

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

    def generate_title_for_found_resource(self, html_title: str, found_resource_path: str) -> str:
        """Generate resource title making use of its parent title."""

        normalized_path = unicodedata.normalize("NFC", found_resource_path)
        resource_title = html_title + " " + Path(normalized_path).name
        return resource_title

    def find_document_links_local_regex(self, html_text: str, extension: str) -> list[str]:
        """Find documents links in a local HTML file using regex"""

        # Ensure the extension starts with a dot
        if not extension.startswith("."):
            extension = "." + extension

        link_pattern = re.compile(
            rf'href=[\'"]([^\'"]+{re.escape(extension)}(?:\?[^\s\'"]*)?)[\'"]',
            re.IGNORECASE,
        )
        base_url = ""
        links = [urljoin(base_url, match) for match in link_pattern.findall(html_text)]
        return links

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
                if not href:
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

    def parse_untagged_documents(
        self,
        soup: BeautifulSoup,
        course_path: str,
        html_display_name: str,
        assets_map: dict[str, str],
        asset_base_url: str | None,
        tag_metadata: dict,
        untagged_documents: UntaggedDocuments,
        vertical_has_video: bool,
    ) -> list[MOOCResource]:
        """
        Build a resource for every document a page links without tagging it.

        Which tag a PDF gets is decided by looking at its first page, since a
        slide deck and a reading want to be chunked differently. Anything that
        is not a PDF cannot be a deck, so it is filed as theory.
        """

        # The slides of a page that holds a video are already indexed from the
        # video itself, one frame per slide, each linking to its own moment
        if vertical_has_video:
            return []

        mooc_resources: list[MOOCResource] = []

        for linked, link_tag in self.tagged_links(soup, untagged_documents.extensions):
            # A tagged link is not this method's business
            if link_tag:
                continue

            resource_path, resource_url = self.resolve_link(linked, course_path, assets_map, asset_base_url)
            if not resource_path.exists():
                logger.warning(f"Missing asset: href={linked!r} resolved={resource_path}")
                continue

            mime_type = mt.guess_mime_type(resource_path)

            tag = untagged_documents.theory_tag
            if mime_type == mt.PDF:
                try:
                    if pdf_is_slides(resource_path):
                        tag = untagged_documents.theory_slides_tag
                except Exception as error:
                    # A document that cannot be looked at is still worth having,
                    # so it is filed as theory and the file is named in the log
                    logger.warning(f"Could not classify {resource_path.name}, filing it as {tag}: {error}")

            tag_dict = tag_metadata.get(tag)
            if tag_dict is None:
                logger.info(f"Skipping {linked} because its tag ({tag}) is unexpected")
                continue

            logger.info(f"Untagged document {resource_path.name} filed as {tag}")

            processing_method = None
            model = None
            if mime_type == mt.PDF:
                processing_method = self.pdf_default_processing_method
                model = self.pdf_default_model

            mooc_resources.append(
                MOOCResource(
                    title=self.generate_title_for_found_resource(html_display_name, str(linked)),
                    source="mooc",
                    url=resource_url,
                    path=str(resource_path),
                    mime_type=mime_type,
                    type=tag_dict.get("type"),
                    subtype=tag_dict.get("subtype"),
                    is_solution=tag_dict.get("is_solution", False),
                    one_chunk_per_page=tag_dict.get("one_chunk_per_page"),
                    one_chunk_per_doc=tag_dict.get("one_chunk_per_doc"),
                    processing_method=processing_method,
                    model=model,
                    is_video=False,
                    is_gemini_processed_video=False,
                )
            )

        return mooc_resources

    def resolve_link(
        self, linked: str, course_path: str, assets_map: dict[str, str], asset_base_url: str | None
    ) -> tuple[Path, str | None]:
        """Return where a linked file is in the export, and the url it is published at."""

        resource_url = self.asset_url(linked, asset_base_url)

        if "block@" in linked:
            linked = "/static/" + linked.split("block@")[-1]

        # Some pages encode their links twice, so unquoting is repeated until
        # it stops changing anything
        previous = None
        while linked != previous:
            previous, linked = linked, unquote(linked)

        relative_path = Path(linked)
        resource_path = Path(course_path) / relative_path.relative_to("/") if linked.startswith("/") else None
        if resource_path is None or not resource_path.exists():
            resource_path = get_filename_via_assets(course_path, linked, assets_map)

        return resource_path, resource_url

    def parse(
        self,
        course_path: str,
        elem_vertical: _Element,
        vertical_display_name: str,
        assets_map: dict[str, str],
        tag_metadata: dict,
        asset_base_url: str | None = None,
        untagged_documents: UntaggedDocuments | None = None,
        vertical_has_video: bool = False,
    ) -> list[MOOCResource]:
        """Parse a MOOC HTML file"""

        mooc_resources: list[MOOCResource] = []

        html_url_name = elem_vertical.get("url_name", "")
        html_filename = html_url_name + ".html"
        html_xml_filename = html_url_name + ".xml"
        markdown_filename = html_url_name + ".md"

        logging.debug(f"course_path={course_path}")
        html_xml_path = Path(course_path) / "html" / html_xml_filename

        html_path = Path(course_path) / "html" / html_filename

        root_html = load_root_elem_from_mooc_xml(html_xml_path)
        if root_html is None:
            return []

        html_display_name = root_html.get("display_name", "")
        mooc_resource_title = vertical_display_name + " - " + html_display_name
        logger.debug(f"mooc_resource_title={mooc_resource_title}")

        # Extract tag
        module_number = None

        module_tag, module_number, mooc_resource_title = split_tag_number_text(mooc_resource_title)
        logger.debug(f"title module_tag={module_tag}")
        logger.debug(f"title module_number={module_number}")
        logger.debug(f"title mooc_resource_title={mooc_resource_title}")

        # Extract text from HTML
        try:
            html_text = html_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Exception when reading HTML file %s: %s", html_path, e)
            return []

        soup = BeautifulSoup(html_text, "html.parser")

        if untagged_documents is None:
            untagged_documents = UntaggedDocuments()

        if not module_tag:
            plain_text = soup.get_text()

            # Look for tags inside the HTML
            # Only the tag is taken from the text. html_text stays the raw
            # HTML, because the links below live in its href attributes and
            # the plain text has none of them
            module_tag, module_number, _ = split_tag_number_text(plain_text)

            logger.debug(f"inside html module_tag={module_tag}")
            logger.debug(f"inside html module_number={module_number}")

        # A page carrying no tag the course declares has no metadata of its
        # own, but the documents it links can still be classified one by one
        if module_tag not in tag_metadata:
            if not untagged_documents.include:
                return []

            return self.parse_untagged_documents(
                soup=soup,
                course_path=course_path,
                html_display_name=html_display_name,
                assets_map=assets_map,
                asset_base_url=asset_base_url,
                tag_metadata=tag_metadata,
                untagged_documents=untagged_documents,
                vertical_has_video=vertical_has_video,
            )

        tag_dict = tag_metadata.get(module_tag)

        # HTML to MarkDown
        md_text = self.convert_html_text_to_markdown(soup)

        # We sanitize the filename of the MarkDown file we create
        markdown_filename = sanitize_for_filename(markdown_filename)

        # Write MarkDown to file in the same HTML folder
        markdown_path = Path(course_path) / "html" / markdown_filename

        Path(markdown_path).write_text(md_text, encoding="utf-8")
        logger.debug(
            "markdown_path exists? %s (%s)",
            Path(markdown_path).exists(),
            repr(markdown_path),
        )

        mime_type = mt.guess_mime_type(str(markdown_path))
        if module_number is not None:
            module_number = str(module_number)

        # Create resource
        html_resource: MOOCResource = MOOCResource(
            source="mooc",
            url=None,
            title=mooc_resource_title,
            path=str(markdown_path),
            mime_type=mime_type,
            type=tag_dict.get("type"),
            subtype=tag_dict.get("subtype"),
            number=module_number,
            one_chunk_per_page=tag_dict.get("one_chunk_per_page"),
            one_chunk_per_doc=tag_dict.get("one_chunk_per_doc"),
            processing_method=tag_dict.get("processing_method"),
            model=tag_dict.get("model"),
        )
        mooc_resources.append(html_resource)

        # For all supported linked files, each with the tag of its own paragraph
        for linked, link_tag in self.tagged_links(soup, (".pdf", ".txt", ".zip", ".md")):
            # A link's own tag wins over the page's, so one page can hold
            # the exercises and their solutions and name each correctly
            if link_tag:
                link_module_tag, link_module_number, _ = split_tag_number_text(f"[{link_tag}]")
            else:
                link_module_tag, link_module_number = module_tag, module_number

            logger.debug(f"linked={linked}")

            # A page either writes the published url or a path under
            # /static, and the file itself is read from the export either way
            linked_url = self.asset_url(linked, asset_base_url)
            if "block@" in linked:
                linked = "/static/" + linked.split("block@")[-1]

            # Some pages encode their links twice, so unquoting is repeated
            # until it stops changing anything
            previous = None
            while linked != previous:
                previous, linked = linked, unquote(linked)

            linked_path = Path(linked)
            directory = linked_path.parent
            filename = linked_path.name
            filename = sanitize_for_filename(filename)

            relative_path = directory / filename

            # We remove the leading '/' in '/static/'
            relative_path = relative_path.relative_to("/")
            logger.debug(f"course_path={course_path}")
            logger.debug(f"relative_path={relative_path}")
            logger.debug(f"filename={filename}")

            resource_path = Path(course_path) / relative_path

            resource_path_exists = resource_path.exists()
            logger.debug(
                "resource_path exists? %s (%s)",
                resource_path_exists,
                repr(resource_path),
            )
            if not resource_path_exists:
                resource_path = get_filename_via_assets(course_path, linked, assets_map)
                resource_path_exists = resource_path.exists()
                if not resource_path_exists:
                    logger.warning("Missing asset: href=%r resolved=%s", linked, resource_path)
                    # A resource pointing at a file that is not there only
                    # fails later, in whichever transformer opens it first
                    continue

                logger.debug(
                    "resource_path resolved exists? %s (%s)",
                    resource_path_exists,
                    repr(resource_path),
                )

            mime_type = mt.guess_mime_type(resource_path)

            resource_title = self.generate_title_for_found_resource(
                html_title=html_display_name,
                found_resource_path=str(linked),
            )

            # We don't extract tags from the PDF files, we use the one
            # written next to the link, or the page's as a fallback
            if not link_module_tag:
                continue

            logger.debug(f"link_module_tag={link_module_tag}")
            tag_dict = tag_metadata.get(link_module_tag)

            # A tag the course does not declare carries no metadata, so the
            # file is skipped rather than indexed unclassified
            if tag_dict is None:
                logger.info(f"Skipping {linked} because its tag ({link_module_tag}) is unexpected")
                continue

            if link_module_number is not None:
                link_module_number = str(link_module_number)

            # Set processing method and model for PDFs. Both are reset each
            # time, so a PDF's settings cannot leak onto the next file
            processing_method = None
            model = None
            if mime_type == mt.PDF:
                processing_method = self.pdf_default_processing_method
                model = self.pdf_default_model

            # Create resource and append it
            mooc_resource: MOOCResource = MOOCResource(
                title=resource_title,
                source="mooc",
                url=linked_url,
                path=str(resource_path),
                mime_type=mime_type,
                type=tag_dict.get("type"),
                subtype=tag_dict.get("subtype"),
                number=link_module_number,
                is_solution=tag_dict.get("is_solution", False),
                one_chunk_per_page=tag_dict.get("one_chunk_per_page"),
                one_chunk_per_doc=tag_dict.get("one_chunk_per_doc"),
                processing_method=processing_method,
                model=model,
                is_video=False,
                is_gemini_processed_video=False,
            )

            mooc_resources.append(mooc_resource)

        return mooc_resources
