"""Tests run on their own settings: nothing of the machine's environment or its .env reaches them."""

import os

import pytest

# Every setting the code reads from the environment, plus Langfuse's own
SETTINGS = (
    "BASE_PATH",
    "DEFAULT_MEDIASPACE_BASE_URL",
    "ES_CAFILE",
    "ES_HOST",
    "ES_PASSWORD",
    "ES_PORT",
    "ES_USERNAME",
    "GRAPH_AI_PASSWORD",
    "GRAPH_AI_URL",
    "GRAPH_AI_USERNAME",
    "KALTURA_APP_TOKEN_ID",
    "KALTURA_PARTNER_ID",
    "KALTURA_TOKEN",
    "KALTURA_USER_ID",
    "MOODLE_TOKEN",
    "MOODLE_URL",
    "RCP_API_KEY",
    "RCP_BASE_URL",
    "SWITCH_API_URL",
)


@pytest.fixture(autouse=True)
def no_real_settings(monkeypatch):
    """A test that needs a setting sets it: one that leans on the real ones fails here as it would in CI."""
    for name in (*SETTINGS, *(name for name in os.environ if name.startswith("LANGFUSE_"))):
        monkeypatch.delenv(name, raising=False)

    # The CLI fills the environment from a .env: never the developer's own, in a test
    monkeypatch.setattr("rag_etl.cli.load_dotenv", lambda *args, **kwargs: False)
