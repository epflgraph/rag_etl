from typing import List, Tuple
from pathlib import Path

import mimetypes

from bs4 import BeautifulSoup


mimetypes.add_type("application/x-ipynb+json", ".ipynb")


def parse_index(moodle_dump_path: Path, allowed_href_prefixes: Tuple[str]) -> List[dict]:
    """Parse sections and local links from index.html. Returns a list of dictionaries with the resource information."""

    # Open and read the base HTML file
    index = moodle_dump_path / 'index.html'
    html = index.read_text(encoding="utf-8")

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Initialise list of resources
    resources = []

    # Iterate over all headers
    for h in soup.find_all('h3'):
        section_title = h.get_text(strip=True)

        texts = []
        section_resources = []

        # Iterate over all siblings of the header until the next one
        for sib in h.find_next_siblings():
            # Finish if sibling is next header
            if sib.name == 'h3':
                break

            # Skip irrelevant siblings
            if sib.name in {'script', 'style'}:
                continue

            # Gather partial texts for each sibling
            text = sib.get_text(separator='\n', strip=True)
            if text:
                texts.append(text)

            # Gather all links in sibling
            for a in sib.find_all('a', href=True):
                href = a['href']

                # Keep only allowed href prefixes
                if not href.startswith(allowed_href_prefixes):
                    continue

                section_resources.append({'url': href, 'title': a.get_text(strip=True)})

        # Now we have the full section text
        section_text = '\n'.join(texts).strip()

        # Iterate over the resources found in this section and add the section details
        for resource in section_resources:
            resources.append({
                'section_title': section_title,
                'section_text': section_text,
                'url': resource['url'],
                'title': resource['title'],
            })

    return resources


def resolve_resource(resource: dict, moodle_dump_path: Path) -> dict:
    """Resolve the Moodle URL and local file path for a single resource, represented by a dictionary."""

    if not resource['url'].startswith('./'):
        return resource

    # Extract resource index path. Fail if not exactly './<folder>/<filename>.html'
    url_parts = resource['url'].split('/')
    if len(url_parts) != 3:
        raise ValueError(f"Unexpected URL shape {resource['url']}; expected './<folder>/<filename>.html'")
    _, folder, filename = url_parts

    # Open and read the HTML file for the particular resource
    resource_index = moodle_dump_path / folder / filename
    resource_html = resource_index.read_text(encoding="utf-8")

    # Parse with BeautifulSoup
    soup = BeautifulSoup(resource_html, "html.parser")

    # Extract the <main> element from the HTML, fail if not present
    main = soup.find('main')
    if main is None:
        raise RuntimeError(f"<main> not found in {resource_index}")

    # Find the correct links (to Moodle and to the local file)
    for a in main.find_all('a', href=True):
        href = a['href']

        if href.startswith('https://moodle.epfl.ch'):
            # The correct url for the file is the actual Moodle link
            resource['url'] = href
        elif href.startswith('content/'):
            # The path of the file is the one under content/
            resource['path'] = f"{moodle_dump_path}/{folder}/{href}"

    # Clean up the link text (remove Moodle resource type): e.g. "examen 2018 (File)" -> "examen 2018"
    moodle_resource_type = folder.split('_')[0]
    resource['title'] = resource['title'].replace(f'({moodle_resource_type})', '').strip()

    # Guess resource mime type
    mime_type, _ = mimetypes.guess_type(resource['path'])
    resource['mime_type'] = mime_type

    return resource
