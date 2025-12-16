from __future__ import annotations

import shutil
from pathlib import Path
import requests

from typing import List

import logging

from rag_etl.resources import MoodleResource
from rag_etl.extractors import BaseExtractor

from rag_etl.extractors.moodle.utils import extract_moodle_tag

from rag_etl.config import CONFIG


class MoodleExtractor(BaseExtractor):
    """
    Extractor for retrieving course materials from Moodle.
    """

    def __init__(
        self,
        moodle_course_id: int,
        moodle_base_path: str,
    ) -> None:
        self.moodle_course_id = moodle_course_id
        self.moodle_base_path = Path(moodle_base_path)

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

                    # Extract module contents tag, default to module tag
                    module_contents_tag = extract_moodle_tag(module_contents['filename'])
                    if not module_contents_tag:
                        module_contents_tag = module_tag

                    # Append resource
                    resources.append(MoodleResource(
                        section_title=section['name'],
                        section_text=section['summary'],
                        tag=module_contents_tag,
                        title=f"{module['name']} > {module_contents_unique_name}",
                        url=module['url'],
                        path=str(module_contents_path),
                        source='moodle',
                        mime_type=module_contents['mimetype'],
                    ))

        return resources
