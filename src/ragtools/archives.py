"""Archives: zip and tar.gz archives turned into plain files.

An archive handed over by a source is extracted next to itself into a
folder named after it, so the material that was one file becomes a
directory of files. Names are made readable and written in one normal
form, since course material travels between systems with different
opinions about accents and encodings.
"""

from __future__ import annotations

import logging
import tarfile
import unicodedata
import zipfile

from collections.abc import Iterator
from pathlib import Path

from ragtools.encoding import zip_entry_filename

logger = logging.getLogger(__name__)


def _extract_dir(archive_path: Path) -> Path:
    """The folder an archive is extracted into, named after it, composed."""
    name = archive_path.name
    for suffix in (".tar.gz", ".tgz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    else:
        name = archive_path.stem

    return archive_path.with_name(unicodedata.normalize("NFC", name))


def unzip_file(zip_path: str | Path) -> Path:
    """Extract a zip archive next to itself and return the folder."""
    zip_path = Path(zip_path)
    extract_dir = _extract_dir(zip_path)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        entries = zip_ref.infolist()
        for entry in entries:
            entry.filename = zip_entry_filename(entry)

        # The entries have to be handed over as objects: given names, extractall
        # looks each one up in an index still keyed by the names they had before
        # this, and raises for every one of them
        zip_ref.extractall(extract_dir, members=entries)

    return extract_dir


def untar_file(tar_path: str | Path) -> Path:
    """Extract a tar archive (any compression) next to itself and return the folder."""
    tar_path = Path(tar_path)
    extract_dir = _extract_dir(tar_path)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:*") as tar_ref:
        # `data` refuses entries pointing outside the folder or at devices
        tar_ref.extractall(extract_dir, filter="data")

    return extract_dir


def iter_files(root: str | Path) -> Iterator[Path]:
    """Yield every file under root, skipping macOS metadata."""
    root = Path(root)
    for path in root.rglob("*"):
        if path.is_file():
            # Skip macOS metadata
            parts = set(path.parts)
            if "__MACOSX" in parts or ".ipynb_checkpoints" in parts or path.name == ".DS_Store":
                continue
            yield path
