"""Content-hash disk cache: the one cache every transform shares.

The scope is the transform's name; the key is the sha256 of the content
the transform would recompute from, so identical content shares an entry
across projects. Values are files or whole directories (nested), copied
in and out as they are. A per-run `--refresh TRANSFORM` flag bypasses
lookups for the named transform; there is no eviction — the cache grows
until cleaned.
"""

from __future__ import annotations

import shutil

from pathlib import Path

from rag_etl.config import CONFIG


class Cache:
    """Disk cache for one transform, keyed by content hash only."""

    def __init__(self, transform: str, refresh: bool = False):
        self.transform = transform
        self.refresh = refresh

        cache_path = Path(CONFIG["CACHE_DIR"])
        if not cache_path.exists():
            raise ValueError(f"Cache path {cache_path} does not exist.")
        self.cache_path = cache_path / transform

    def _entry_path(self, content_hash: str, value: Path) -> Path:
        return self.cache_path / content_hash / value.name

    def get(self, content_hash: str, value: Path) -> bool:
        """Copy the cached output for `content_hash` to `value`; False on miss or refresh."""
        value = Path(value)

        if self.refresh:
            return False

        entry = self._entry_path(content_hash, value)
        if not entry.is_file() and not entry.is_dir():
            return False

        if entry.is_dir():
            value.mkdir(parents=True, exist_ok=True)
            shutil.copytree(entry, value, dirs_exist_ok=True)
        else:
            value.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry, value)

        return True

    def set(self, content_hash: str, value: Path) -> None:
        """Store the output produced for `content_hash` in the cache."""
        value = Path(value)
        entry = self._entry_path(content_hash, value)

        if value.is_dir():
            entry.mkdir(parents=True, exist_ok=True)
            shutil.copytree(value, entry, dirs_exist_ok=True)
        else:
            entry.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(value, entry)
