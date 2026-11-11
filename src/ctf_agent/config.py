"""Load settings from the local .env file (never printed or logged)."""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def update_env(updates: dict[str, str], path: str | Path = ".env") -> None:
    """Set key=value pairs in the .env file, preserving everything else.

    The first uncommented occurrence of each key is replaced in place; later
    duplicate uncommented occurrences of that key are dropped; unknown keys
    are appended. Values are written verbatim — callers must never log them.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = p.read_text().splitlines() if p.exists() else []
    out: list[str] = []
    written: set[str] = set()
    for line in lines:
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip() if "=" in stripped else None
        if key not in updates:
            out.append(line)
        elif key not in written:
            out.append(f"{key}={updates[key]}")  # first occurrence: replace
            written.add(key)
        # else: later uncommented duplicate of a set key — drop it
    for k, v in updates.items():
        if k not in written:
            out.append(f"{k}={v}")
    p.write_text("\n".join(out) + "\n")
