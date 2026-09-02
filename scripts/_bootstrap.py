"""Shared script bootstrap helpers.

Keeps standalone script entrypoints consistent and avoids repeating
project-root discovery, src path injection, and default config lookup.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "project_config.yml"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
