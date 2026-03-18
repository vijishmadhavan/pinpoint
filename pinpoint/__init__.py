from __future__ import annotations

from pathlib import Path

__version__ = "1.0.4"


def user_data_dir() -> Path:
    return Path.home() / ".pinpoint"
