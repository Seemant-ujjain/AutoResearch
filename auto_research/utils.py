"""Utility helpers for the AutoResearch project."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def slugify(value: str, max_length: int = 60) -> str:
    """Convert a question or title into a filesystem-safe slug."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    if not normalized:
        return "untitled"
    return normalized[:max_length].rstrip("-")


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
