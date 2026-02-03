from __future__ import annotations

from datetime import datetime

import shutil
from pathlib import Path
import requests

import re
import json

from typing import List, Optional

import logging

from rag_etl.resources import MoodleResource
from rag_etl.extractors import BaseExtractor

import rag_etl.utils.mime_types as mt
from rag_etl.config import CONFIG


def extract_moodle_tag(text):
    match = re.search(r"\[(.*?)\]", text)

    if match:
        return match.group(1)

    return None


def extract_tag_number(tag):
    match = re.search(r'_(\d+)(?:_|$)', tag)

    if not match:
        return (None, tag)

    number = match.group(1)
    rest = tag.replace(f'_{number}', '')

    return (int(number), rest)


def extract_url(module, module_contents):
    # If resource is hidden from students, do not fill in url
    if not module['visible']:
        return None

    # If module is visible, use module contents url if any or default to module url
    if module_contents['fileurl']:
        url = module_contents['fileurl']
        url = url.replace('https://moodle.epfl.ch/webservice', 'https://moodle.epfl.ch')
        url = url.replace('?forcedownload=1', '')
    else:
        url = module['url']
        url = f'{url}?redirect=1'

    return url


def extract_from_and_until(module):
    # If not specified availability, return
    if not module['availability']:
        return (None, None)

    # If availability is not parsable, return
    try:
        availability = json.loads(module['availability'])
    except Exception:
        return (None, None)

    # Initialise from_ and until
    from_ = None
    until = None

    # Keep only date restrictions
    restrictions = [restriction for restriction in availability.get('c', []) if restriction.get('type') == 'date']

    # From field
    gt_restrictions = [restriction for restriction in restrictions if restriction.get('d') in ('>=', '>')]
    if gt_restrictions:
        gt_epochs = [restriction.get('t') for restriction in gt_restrictions]
        from_ = datetime.fromtimestamp(max(gt_epochs)).strftime("%Y-%m-%dT%H:%M:%S.%f")

    # Until field
    lt_restrictions = [restriction for restriction in restrictions if restriction.get('d') in ('<=', '<')]
    if lt_restrictions:
        lt_epochs = [restriction.get('t') for restriction in lt_restrictions]
        until = datetime.fromtimestamp(min(lt_epochs)).strftime("%Y-%m-%dT%H:%M:%S.%f")

    return from_, until


class MoodleExtractor(BaseExtractor):
    """
    Extractor for retrieving course materials from Moodle.
    """

    def __init__(
        self,
        moodle_course_id: int,
        moodle_base_path: str,
        tag_metadata: Optional[dict] = None,
        mime_types: Optional[list[str]] = None,
    ) -> None:
        self.moodle_course_id = moodle_course_id
        self.moodle_base_path = Path(moodle_base_path)
        self.tag_metadata = tag_metadata
        if mime_types is None:
            self.mime_types = mt.DEFAULT_MIME_TYPES
        else:
            self.mime_types = mime_types

    def extract(self) -> List[MoodleResource]:
        """
        Extract resources for this course from Moodle.

        Returns:
            List[MoodleResource]: List of raw Resources.
        """

        # Build Moodle endpoint and parameters
        moodle_endpoint = f"{CONFIG['MOODLE_URL']}/webservice/rest/server.php"

        params = {
            "wstoken": CONFIG['MOODLE_TOKEN'],
            "wsfunction": "core_course_get_contents",
            "moodlewsrestformat": "json",
            "courseid": self.moodle_course_id
        }

        # Retrieve course contents from Moodle API
        sections = requests.get(moodle_endpoint, params=params).json()

        # Empty moodle_base_path if it exists
        if self.moodle_base_path.exists():
            shutil.rmtree(self.moodle_base_path)

        # Iterate over sections, modules and module contents
        resources = []
        for section in sections:
            for module in section.get('modules', []):
                # Skip if not a 'resource' (filter Forum modules, URL modules, etc.)
                if module['modname'] not in ('resource', 'folder'):
                    logging.debug(f"Skipping module {module['name']} because of modname {module['modname']}")
                    continue

                # Extract module tag
                module_tag = extract_moodle_tag(module['name'])

                # Build module unique name
                module_unique_name = f"{module['modplural'][:-1]}.{module['name'].replace(':', '')}.{module['id']}"
                module_path = self.moodle_base_path / module_unique_name / 'content'

                for module_contents in module.get('contents', []):
                    # Skip if mime type not in list
                    if module_contents['mimetype'] not in self.mime_types:
                        continue

                    # Extract module contents tag, default to module tag
                    tag = extract_moodle_tag(module_contents['filename'])
                    if not tag:
                        tag = module_tag

                    # Skip if no tag
                    if not tag:
                        continue

                    # Split number and tag
                    number, tag = extract_tag_number(tag)

                    # Skip if unrecognised tag
                    if tag not in self.tag_metadata:
                        continue

                    # Extract metadata from tag
                    type_ = self.tag_metadata.get(tag, {}).get('type')
                    subtype = self.tag_metadata.get(tag, {}).get('subtype')
                    is_solution = self.tag_metadata.get(tag, {}).get('is_solution', False)
                    one_chunk_per_page = self.tag_metadata.get(tag, {}).get('one_chunk_per_page', False)
                    one_chunk_per_doc = self.tag_metadata.get(tag, {}).get('one_chunk_per_doc', False)

                    # Download file from url
                    url = f"{module_contents['fileurl']}&token={CONFIG['MOODLE_TOKEN']}"
                    response = requests.get(url)

                    # Skip file if download fails
                    try:
                        response.raise_for_status()  # Raises an error if download fails
                    except requests.HTTPError as e:
                        logging.debug(f"Download failed for file {module['name']} > {module_contents['filename']}. Ignoring...")
                        continue

                    # Save file to disk
                    module_contents_path = module_path / Path(module_contents['filepath']).relative_to('/') / module_contents['filename']
                    module_contents_unique_name = str(module_contents_path.relative_to(module_path))
                    module_contents_path.parent.mkdir(parents=True, exist_ok=True)
                    module_contents_path.write_bytes(response.content)

                    # Extract url
                    url = extract_url(module, module_contents)

                    # Extract availability date if specified
                    from_, until = extract_from_and_until(module)

                    # Processing method and model
                    if module_contents['mimetype'] == mt.PDF:
                        processing_method = 'rcp'
                        model = CONFIG['RCP_VISION_MODEL']
                    else:
                        processing_method = None
                        model = None

                    # Append resource
                    resources.append(MoodleResource(
                        section_title=section['name'],
                        section_text=section['summary'],
                        tag=tag,
                        title=f"{module['name']} > {module_contents_unique_name}",
                        url=url,
                        path=str(module_contents_path),
                        source='moodle',
                        mime_type=module_contents['mimetype'],
                        type=type_,
                        subtype=subtype,
                        number=number,
                        is_solution=is_solution,
                        processing_method=processing_method,
                        model=model,
                        one_chunk_per_page=one_chunk_per_page,
                        one_chunk_per_doc=one_chunk_per_doc,
                        from_=from_,
                        until=until,
                    ))

        return resources
