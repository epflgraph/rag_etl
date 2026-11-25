"""Content-hash disk cache: the one cache every transform shares.

Each project keeps its own, in its folder's cache/, one folder per
recipe version (`<project>/cache/pdf-ocr-1/`): bumping a step's version
starts a fresh folder, so nothing made under the old recipe is ever served
again. An entry is a key, most often the sha256 of the content the
transform would recompute from, so identical content within one project
shares it, holding named text values (`<key>/<name>`). A per-run
`--refresh STEP` flag bypasses lookups for the named step; there is no
eviction — the cache grows until cleaned.

Documents made of parts are cached as Merkle trees, parts and documents
side by side in the one folder: each part (a page's transcript) is an
ordinary entry under its own key, and the document's entry holds an index
of its ordered parts. A document found by its index costs nothing; one
whose index misses falls back to its parts, so only what changed is redone.
"""

import json

from collections.abc import Sequence
from pathlib import Path

# The file a document's index is stored under, next to nothing else in its entry
INDEX_FILE = "index.json"


class Cache:
    """Disk cache of one recipe version for one project: text values by key and name."""

    def __init__(self, cache_path: Path, refresh: bool = False):
        self.cache_path = cache_path
        self.refresh = refresh

    def read(self, key: str, name: str) -> str | None:
        """Return the text value cached under `key`, or None on miss or refresh."""
        if self.refresh:
            return None

        entry = self.cache_path / key / name
        if not entry.is_file():
            return None

        return entry.read_text(encoding="utf-8")

    def write(self, key: str, name: str, text: str) -> None:
        """Store a text value under `key`."""
        entry = self.cache_path / key / name
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(text, encoding="utf-8")

    def write_index(self, key: str, parts: Sequence[tuple[str, str]]) -> None:
        """Record a document's ordered parts under `key`: each one's name and the key its text is cached under."""
        self.write(key, INDEX_FILE, json.dumps([[name, part_key] for name, part_key in parts]))

    def read_index(self, key: str) -> list[tuple[str, str]] | None:
        """Return the (name, text) parts of the document indexed under `key`, in order.

        None — so the caller redoes the document — when it has no index,
        the index is malformed, or any part it names is missing (refresh
        included).
        """
        index = self.read(key, INDEX_FILE)
        if index is None:
            return None

        try:
            named_keys = [(str(name), str(part_key)) for name, part_key in json.loads(index)]
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

        parts: list[tuple[str, str]] = []
        for name, part_key in named_keys:
            text = self.read(part_key, name)
            if text is None:
                return None
            parts.append((name, text))

        return parts
