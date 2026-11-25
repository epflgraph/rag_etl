"""Images: every image read by a vision model, wherever it is.

An image file is read in place, its transcript taking its place. A
markdown file's local images (`![alt](figure.png)`, `![alt](<path with
spaces.png>)`, relative to the file or absolute) are each replaced by
their transcript, where they were, in a scratch copy. One cache for both:
an image is read once, wherever it is shown. The step runs before pdf and
video, so the markdown it reads is only ever the sources', never pages
or slides read later.
"""

import logging
import re

from dataclasses import dataclass
from urllib.parse import unquote

import ragtools.mime_types as mt
from ragtools.images import transcribe_slide

from rag_etl.core import Resource
from rag_etl.core.resource import hash_file, hash_text
from rag_etl.transformers.base import Transformer
from rag_etl.transformers.ocr import cached_transcript, file_to_read

logger = logging.getLogger(__name__)

# The images that are read, as files of their own or shown in markdown
IMAGES = (mt.JPEG, mt.PNG)

# ![alt](target) or ![alt](<target>), an optional "title" after the target
IMAGE_REFERENCE = re.compile(r'!\[[^\]]*\]\((?:<([^>]+)>|([^)\s]+))(?:\s+"[^"]*")?\)')


@dataclass(frozen=True)
class ImagesTransformer(Transformer):
    """Reads image files in place, and the local images markdown files show."""

    name = "images"
    cache_version = "image-ocr-1"  # the image reads (prompts, model)

    def transform(self, roots: list[Resource]) -> list[Resource]:
        for root in roots:
            for node in list(root.walk()):
                if node.mime_type in IMAGES:
                    self._read_image(node)
                elif node.mime_type == mt.MARKDOWN:
                    self._read_shown_images(node)
        return roots

    def _read_image(self, node: Resource) -> None:
        image_path = file_to_read(node)
        work_dir = self.work_root / hash_file(image_path)
        md_path = cached_transcript(self.cache, image_path, transcribe_slide, work_dir / f"{image_path.stem}.md")

        if not md_path.read_text(encoding="utf-8").strip():
            logger.info(f"Dropping {node.title}: no readable content on it")
            node.detach()
            return

        node.path = md_path
        node.mime_type = mt.MARKDOWN

    def _read_shown_images(self, node: Resource) -> None:
        md_path = file_to_read(node)
        text = md_path.read_text(encoding="utf-8")
        work_dir = self.work_root / hash_text(text)
        read_any = False

        def transcript(match: re.Match[str]) -> str:
            nonlocal read_any
            target = match.group(1) or match.group(2)

            # Web images and embedded data are not files the material ships
            if re.match(r"[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
                return match.group(0)

            image_path = (md_path.parent / unquote(target)).resolve()
            if mt.guess_mime_type(str(image_path)) not in IMAGES:
                return match.group(0)

            # A folder may ship a markdown file without the images it shows:
            # the text is still worth indexing, the reference stays as written
            if not image_path.is_file():
                logger.warning(f"{node.title}: image {target} not found, left as a reference")
                return match.group(0)

            read_any = True
            image_md = cached_transcript(
                self.cache, image_path, transcribe_slide, work_dir / "images" / f"{image_path.stem}.md"
            )
            return image_md.read_text(encoding="utf-8").strip()

        read = IMAGE_REFERENCE.sub(transcript, text)
        if not read_any:
            return

        # The source file stays as it is: the text with its images read is scratch
        out_path = work_dir / md_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(read, encoding="utf-8")
        node.path = out_path
