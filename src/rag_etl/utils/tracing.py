from __future__ import annotations

from langfuse import get_client

from rag_etl.config import CONFIG

_REQUIRED_ENV_VARS = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
)


def is_tracing_enabled() -> bool:
    """Return True when all required Langfuse configuration values are present."""
    return all(CONFIG.get(key) for key in _REQUIRED_ENV_VARS)


def get_langfuse():
    """Return the configured Langfuse singleton client, or None if tracing is disabled."""
    if not is_tracing_enabled():
        return None
    return get_client()
