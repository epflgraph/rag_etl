"""Files: walking a folder the way every source and archive is walked."""

from __future__ import annotations

import os

from collections.abc import Iterator
from pathlib import Path

# Folders that only ever hold metadata written by an operating system
SKIPPED_FOLDERS = ("__MACOSX",)


def is_hidden(name: str) -> bool:
    """Whether a file or folder name is hidden (.git, .DS_Store, .ipynb_checkpoints, ...)."""
    return name.startswith(".")


def iter_files(root: str | Path) -> Iterator[Path]:
    """Yield every file under root, in the order the filesystem lists them.

    Hidden files and folders are skipped, and so are folders that only hold
    operating-system metadata (__MACOSX); nothing inside a skipped folder is
    visited.
    """
    for folder, folder_names, file_names in os.walk(root, topdown=True):
        folder_names[:] = [name for name in folder_names if not is_hidden(name) and name not in SKIPPED_FOLDERS]

        for file_name in file_names:
            if not is_hidden(file_name):
                yield Path(folder) / file_name
