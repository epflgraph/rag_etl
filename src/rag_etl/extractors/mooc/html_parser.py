import logging
from pathlib import Path
from lxml.etree import _Element
from urllib.parse import urljoin, urlparse
import re
import unicodedata
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString


from rag_etl.resources.mooc_resource import MOOCResource
from rag_etl.extractors.mooc.utils import load_root_elem_from_mooc_xml, clean_text

from rag_etl.utils.tags import split_tag_number_text
from rag_etl.utils import resolve_path

import rag_etl.utils.mime_types as mt


logger = logging.getLogger(__name__)


class HtmlParser:
    """
    HTML Parser for MOOCs.
    """

    def convert_html_text_to_markdown(self, html_text: str) -> str:
        """Convert HTML content to Markdown with BeautifulSoup"""

        soup = BeautifulSoup(html_text, "html.parser")

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

    def generate_title_for_found_resource(
        self, html_title: str, found_resource_path: str
    ) -> str:
        """Generate resource title making use of its parent title."""

        normalized_path = unicodedata.normalize("NFC", found_resource_path)
        resource_title = html_title + " " + Path(normalized_path).name
        return resource_title

    def find_document_links_local_regex(
        self, html_text: str, extension: str
    ) -> list[str]:
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

    def parse(
        self,
        course_path: str,
        elem_vertical: _Element,
        vertical_display_name: str,
        tag_metadata: dict,
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

        # Extract tag
        module_number = None
        # module_tag = self.extract_mooc_tag(mooc_resource_title)
        logger.debug(f"mooc_resource_title={mooc_resource_title}")
        # module_tag = extract_tag(mooc_resource_title)

        module_tag, module_number, mooc_resource_title = split_tag_number_text(mooc_resource_title)
        logger.debug(f"title module_tag={module_tag}")
        logger.debug(f"title module_number={module_number}")
        logger.debug(f"title mooc_resource_title={mooc_resource_title}")

        # Extract text from HTML
        html_text = html_path.read_text(encoding="utf-8")

        if not module_tag:
            # Look for tags inside the HTML
            module_tag, module_number, html_text = split_tag_number_text(html_text)

            if not module_tag:
                logger.warning(
                    "HtmlParser: no module tag found in title nor html content:"
                )
                return []

            logger.debug(f"inside html module_tag={module_tag}")
            logger.debug(f"inside html module_number={module_number}")
            logger.debug(f"inside html html_text={html_text}")

        if module_tag not in tag_metadata.keys():
            return []

        tag_dict = tag_metadata.get(module_tag)

        # HTML to MarkDown
        md_text = self.convert_html_text_to_markdown(html_text=html_text)

        # Write MarkDown to file in the same HTML folder
        markdown_path = Path(course_path) / "html" / markdown_filename
        markdown_path.write_text(md_text, encoding="utf-8")

        mime_type = mt.guess_mime_type(str(markdown_path))

        # Create resource
        html_resource: MOOCResource = MOOCResource(
            source="mooc",
            title=mooc_resource_title,
            url="",
            path=markdown_path,
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

        # For all supported linked files
        for ext in ("pdf", "txt", "zip", "md"):
            links = self.find_document_links_local_regex(
                html_text=html_text, extension=ext
            )
            for linked in links:
                # If it's an URL skip
                if "http" in linked:
                    continue

                parsed = urlparse(linked)
                resource_path = Path(course_path) / parsed.path.lstrip("/")
                logger.info(
                    "resource_path exists? %s (%s)",
                    Path(resource_path).exists(),
                    repr(str(resource_path)),
                )
                resource_path = resolve_path(resource_path)
                logger.debug("resource_path= %s", str(resource_path))
                mime_type = mt.guess_mime_type(str(resource_path))

                resource_title = self.generate_title_for_found_resource(
                    html_title=html_display_name,
                    found_resource_path=str(linked),
                )

                # We don't extract tags from the PDF files, we use the one extracted from the HTML title or content
                if not module_tag:
                    logger.warning(f"HtmlParser: no module tag found in title: {resource_title}")
                    return []

                logger.debug(f"module_tag={module_tag}")
                tag_dict = tag_metadata.get(module_tag)

                if module_number is not None:
                    module_number = str(module_number)

                # Create resource and append it
                mooc_resource: MOOCResource = MOOCResource(
                    title=resource_title,
                    source="mooc",
                    url="",
                    path=str(resource_path),
                    mime_type=mime_type,
                    type=tag_dict.get("type"),
                    subtype=tag_dict.get("subtype"),
                    number=module_number,
                    one_chunk_per_page=tag_dict.get("one_chunk_per_page"),
                    one_chunk_per_doc=tag_dict.get("one_chunk_per_doc"),
                    processing_method=tag_dict.get("processing_method"),
                    model=tag_dict.get("model"),
                    is_video=False,
                    is_gemini_processed_video=False,
                )

                mooc_resources.append(mooc_resource)

        return mooc_resources
