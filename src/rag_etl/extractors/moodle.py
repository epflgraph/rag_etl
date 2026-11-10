from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

import ragtools.mime_types as mt
from rag_etl.core import Resource
from rag_etl.extractors.base import Extractor, SourceUnavailable
from rag_etl.extractors.markers import NO_BOT, split_tag_text
from ragtools.encoding import sanitize_for_filename

logger = logging.getLogger(__name__)

# Seconds a request may go without receiving data, so a stalled Moodle fails
# the run loud instead of hanging it; it does not cap a whole download
REQUEST_TIMEOUT = 60

# EPFL's time: release dates read as the professor set them, offset included
LOCAL_TIME = ZoneInfo("Europe/Zurich")


def local_iso(epoch: int) -> str:
    """An epoch as an ISO date in Lausanne's time, with its offset, so it names one instant."""
    return datetime.fromtimestamp(epoch, LOCAL_TIME).isoformat(timespec="microseconds")


def display(name: str) -> str:
    """Return the name with its legacy tag marker stripped, as a student reads it."""
    return split_tag_text(name)[1]


def extract_url(module, module_contents):
    # If resource is hidden from students, do not fill in url
    if not module["visible"]:
        return None

    # If module is visible, use module contents url if any or default to module url
    if module_contents["fileurl"]:
        url = module_contents["fileurl"]
        url = url.replace("https://moodle.epfl.ch/webservice", "https://moodle.epfl.ch")
        url = url.replace("?forcedownload=1", "")
    else:
        url = module["url"]
        url = f"{url}?redirect=1"

    return url


def extract_from_and_until(module) -> tuple[str | None, str | None]:
    # If not specified availability, return
    if not module["availability"]:
        return (None, None)

    # If availability is not parsable, return
    try:
        availability = json.loads(module["availability"])
    except Exception:
        return (None, None)

    # Initialise from_ and until
    from_ = None
    until = None

    # Keep only date restrictions
    restrictions = [restriction for restriction in availability.get("c", []) if restriction.get("type") == "date"]

    # From field
    gt_restrictions = [restriction for restriction in restrictions if restriction.get("d") in (">=", ">")]
    if gt_restrictions:
        gt_epochs = [restriction.get("t") for restriction in gt_restrictions]
        from_ = local_iso(max(gt_epochs))

    # Until field
    lt_restrictions = [restriction for restriction in restrictions if restriction.get("d") in ("<=", "<")]
    if lt_restrictions:
        lt_epochs = [restriction.get("t") for restriction in lt_restrictions]
        until = local_iso(min(lt_epochs))

    return from_, until


@dataclass(frozen=True)
class MoodleExtractor(Extractor):
    """
    Extractor for retrieving course materials from Moodle.

    One container per section, one leaf per downloaded file. Leaves carry
    facts only: title, url, path, mime type and the availability dates.
    Everything not marked NO_BOT is extracted; the judge labels it later.
    """

    course_id: int
    mime_types: Sequence[str] = tuple(mt.DEFAULT_MIME_TYPES)
    sections: int | None = None    # when set, only the first N sections are extracted

    def extract(self) -> Resource:
        """
        Extract the course's material into one container per section.
        """

        # Build Moodle endpoint and parameters
        moodle_endpoint = f"{os.environ['MOODLE_URL']}/webservice/rest/server.php"

        params = {
            "wstoken": os.environ["MOODLE_TOKEN"],
            "wsfunction": "core_course_get_contents",
            "moodlewsrestformat": "json",
            "courseid": self.course_id,
        }

        # Retrieve course contents from Moodle API, failing loud when unreachable
        try:
            sections = requests.get(moodle_endpoint, params=params, timeout=REQUEST_TIMEOUT).json()
        except requests.RequestException as error:
            raise SourceUnavailable(f"Moodle API unreachable: {error}") from error

        # An error (a bad token among them) arrives as an error object, not a list
        if not isinstance(sections, list):
            raise SourceUnavailable(f"Moodle API did not answer with course contents: {sections}")

        if self.sections is not None:
            sections = sections[: self.sections]

        logger.info(f"Moodle course {self.course_id}: {len(sections)} sections")

        course_url = f"{os.environ['MOODLE_URL']}/course/view.php?id={self.course_id}"
        course = Resource(title=f"Moodle course {self.course_id}", url=course_url)

        # Empty dir if it exists
        if self.dir.exists():
            shutil.rmtree(self.dir)

        extracted = 0

        # Iterate over sections, modules and module contents
        for section in sections:
            section_node: Resource | None = None
            for module in section.get("modules", []):
                # Skip if not a 'resource' (filter Forum modules, URL modules, etc.)
                if module["modname"] not in ("resource", "folder"):
                    logger.debug(f"Skipping module {module['name']} because of modname {module['modname']}")
                    continue

                if NO_BOT.search(module["name"]):
                    logger.debug(f"Skipping module {module['name']} because it is marked NO_BOT")
                    continue

                module_title = display(module["name"])
                from_, until = extract_from_and_until(module)

                # One folder per module, named by its kind (resource, folder), name and id
                module_unique_name = f"{module['modname']}.{module['name'].replace(':', '')}.{module['id']}"
                module_unique_name = sanitize_for_filename(module_unique_name)
                module_path = self.dir / module_unique_name / "content"

                for module_contents in module.get("contents", []):
                    mime_type = module_contents["mimetype"]

                    # Moodle reports "document/unknown" for extensions it does not know (e.g. .s).
                    # Fall back to the filename. The list check below still decides what is kept
                    if mime_type == mt.MOODLE_UNKNOWN:
                        guessed = mt.guess_mime_type(module_contents["filename"])
                        if guessed:
                            mime_type = guessed

                    # Skip if mime type not in list
                    if mime_type not in self.mime_types:
                        logger.debug(f"Skipping file {module_contents['filename']} because of mime type {mime_type}")
                        continue

                    # A file marked NO_BOT is not for the bots
                    if NO_BOT.search(module_contents["filename"]):
                        logger.debug(f"Skipping file {module_contents['filename']} because it is marked NO_BOT")
                        continue

                    title = display(module_contents["filename"])

                    # Download file from url, failing loud when it fails
                    try:
                        response = requests.get(
                            f"{module_contents['fileurl']}&token={os.environ['MOODLE_TOKEN']}", timeout=REQUEST_TIMEOUT
                        )
                        response.raise_for_status()
                    except requests.RequestException as error:
                        raise SourceUnavailable(
                            f"Download failed for file {module['name']} > {module_contents['filename']}: {error}"
                        ) from error

                    # Build download path for file
                    module_contents_path = (
                        module_path
                        / Path(module_contents["filepath"]).relative_to("/")
                        / sanitize_for_filename(module_contents["filename"])
                    )

                    # Add .pdf extension if not there
                    if not module_contents_path.suffix and mime_type == mt.PDF:
                        module_contents_path = module_contents_path.with_suffix(".pdf")

                    # Save file to disk
                    module_contents_path.parent.mkdir(parents=True, exist_ok=True)
                    module_contents_path.write_bytes(response.content)
                    extracted += 1
                    logger.info(f"Downloaded {module_title} > {title}")

                    if section_node is None:
                        section_node = course.add_child(title=section.get("name") or "")

                    section_node.add_child(
                        title=f"{module_title} > {title}",
                        url=extract_url(module, module_contents),
                        path=module_contents_path,
                        mime_type=mime_type,
                        from_=from_,
                        until=until,
                    )

        logger.info(f"Moodle course {self.course_id}: extracted {extracted} files")

        return course
