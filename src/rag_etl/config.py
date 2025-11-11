from pathlib import Path
from dotenv import dotenv_values

CONFIG = dotenv_values(Path(__file__).resolve().parents[2] / '.env')
