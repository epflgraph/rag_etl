"""Git extractor: a GitLab/GitHub repository cloned shallow into a resource tree."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import rag_etl.utils.mime_types as mt
from rag_etl.core import Resource
from rag_etl.extractors.base_extractor import Extractor, SourceUnavailable

logger = logging.getLogger(__name__)

# A branch a repository url can point at, as in GitHub "/tree/main" or GitLab "/-/tree/main"
BRANCH_IN_URL = re.compile(r"/(?:-/)?tree/([^/?#]+)")


def branch_from_url(repo_url: str) -> str | None:
    """Return the branch a repository url points at, or None."""
    match = BRANCH_IN_URL.search(repo_url)
    return match.group(1) if match else None


def repo_from_url(repo_url: str) -> str:
    """Return the clone url of the repository, without any branch it points at."""
    return BRANCH_IN_URL.sub("", repo_url).rstrip("/")


@dataclass(frozen=True)
class GitExtractor(Extractor):
    """Extracts the files of a Git repository into a resource tree.

    The url may point at the repository or at a branch: a branch url extracts
    only that branch, otherwise the repository's default branch is cloned.
    Everything indexed by default: one leaf per non-hidden file, left to the
    later stages to make sense of.
    """

    repo_url: str
    branch: str | None = None

    def requested_branch(self) -> str | None:
        """Return the branch to check out: the explicit one, or the one the url points at."""
        return self.branch or branch_from_url(self.repo_url)

    def clone(self, clone_url: str, branch: str | None) -> None:
        """Clone the repository shallow, failing loud when the clone fails."""

        command = ["git", "clone", "--depth", "1"]
        if branch:
            command += ["--branch", branch]
        command += [clone_url, str(self.dir)]

        try:
            result = subprocess.run(command, capture_output=True, text=True)
        except OSError as error:
            raise SourceUnavailable(f"Clone of {clone_url} failed: {error}") from error

        if result.returncode != 0:
            raise SourceUnavailable(f"Clone of {clone_url} failed: {result.stderr.strip()}")

    def checkout(self) -> str:
        """Return the branch the clone has checked out."""

        result = subprocess.run(
            ["git", "-C", str(self.dir), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
        )
        branch = result.stdout.strip()
        return branch if branch and branch != "HEAD" else "detached"

    def extract(self) -> Resource:
        """Extract the repository's files into one container of leaves."""

        clone_url = repo_from_url(self.repo_url)
        requested = self.requested_branch()

        # Deleting first so the folder ends up holding exactly what this run extracted
        if self.dir.exists():
            shutil.rmtree(self.dir)

        self.clone(clone_url, requested)
        branch = self.checkout()

        repo = Path(clone_url).stem
        base = Resource(title=f"{repo}@{branch}", url=self.repo_url)

        for dir_path, dir_names, file_names in os.walk(self.dir, topdown=True):
            dir_names[:] = [d for d in dir_names if not d.startswith(".")]

            for file_name in file_names:
                file_path = Path(dir_path) / file_name

                if file_path.name.startswith("."):
                    logger.debug(f"Skipping file {str(file_path)} because it is hidden.")
                    continue

                base.add_child(
                    Resource(
                        title=str(file_path.relative_to(self.dir)),
                        path=file_path,
                        mime_type=mt.guess_mime_type(str(file_path)),
                    )
                )

        return base
