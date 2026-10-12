from pathlib import Path

from dotenv import dotenv_values, load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)
CONFIG = dotenv_values(_ENV_PATH)


def config_value(key: str) -> str:
    """Return a required config value, failing loud when it is missing."""
    value = CONFIG.get(key)
    if not value:
        raise ValueError(f"Missing required config value: {key}")
    return value
