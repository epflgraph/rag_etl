"""Environment plumbing for ragtools: the one place secrets are read.

A standalone user sets the variables (or a .env next to where they run);
nothing here imports the pipeline.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def required(key: str) -> str:
    """Return a required environment variable, failing loud when missing."""
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value
