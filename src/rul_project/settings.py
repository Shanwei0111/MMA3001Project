"""Runtime paths resolved from CLI environment settings.

DATA_DIR defaults to data/raw and OUTPUT_DIR to outputs under the current
working directory. RUL_DATA_DIR and RUL_OUTPUT_DIR override these defaults.
No directory is created merely by importing this module.
"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("RUL_DATA_DIR", "data/raw")).resolve()
OUTPUT_DIR = Path(os.environ.get("RUL_OUTPUT_DIR", "outputs")).resolve()
