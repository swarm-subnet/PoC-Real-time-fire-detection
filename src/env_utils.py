"""Small environment-file helpers used by command-line scripts."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_if_present(path: Path) -> None:
    """Load simple KEY=VALUE lines from a .env file without overriding env vars."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or "=" not in clean:
            continue

        key, value = clean.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
