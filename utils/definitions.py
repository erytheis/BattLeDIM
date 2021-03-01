import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # This is your Project Root
UTILS_DIR = Path(os.path.dirname(__file__))

RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
