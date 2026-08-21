from pathlib import Path
from dotenv import dotenv_values, load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)
CONFIG = dotenv_values(_ENV_PATH)
