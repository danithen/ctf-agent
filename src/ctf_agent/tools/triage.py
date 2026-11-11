"""Deterministic, zero-LLM triage facts for the seed message.

Trace evidence: every attempt spends 5-8 steps re-deriving the same static
facts (file inventory, file type, architecture, first needle-string hits)
before analysis starts - identical work repeated across every attempt in a
sweep. This module aggregates exactly those facts from the files staged into
the workspace - the same bytes the agent could list/read itself - so the
brief saves steps without granting any information advantage: no
interpretation, no flag material (staging guarantees none is in the
workspace), deterministically derived, cheap to test.
"""
from __future__ import annotations

import re
from pathlib import Path

_NEEDLES = re.compile(rb"(?i)(flag|correct|wrong|key|secret|password)")
_PRINTABLE = re.compile(rb"[\x20-\x7e]{4,}")
_ELF_MACHINES = {3: "i386", 0x28: "ARM", 0x3E: "x86-64", 0xB7: "AArch64"}
_SCAN_LIMIT = 262_144   # max bytes strings-scanned per file
_MAX_HITS = 6           # needle-print hits per file
_MAX_FILES = 8          # files described in the brief


def _elf_desc(data: bytes) -> str | None:
    """One-line ELF identification from the header. None if not an ELF."""
    if data[:4] != b"\x7fELF":
        return None
    is64 = data[4] == 2
    width = 8 if is64 else 4
    endian = "big" if data[5] == 2 else "little"
    mach_i = int.from_bytes(data[18:20], endian)
    mach = _ELF_MACHINES.get(mach_i, f"machine 0x{mach_i:x}")
    entry = int.from_bytes(data[24:24 + width], endian)
    return f"ELF{64 if is64 else 32} {mach}, entry 0x{entry:x}"


def _string_hits(data: bytes) -> list[str]:
    """Printable needle-string hits with offsets (deterministic, capped)."""
    out: list[str] = []
    for m in _PRINTABLE.finditer(data[:_SCAN_LIMIT]):
        s = m.group(0)
        if _NEEDLES.search(s):
            out.append(f'    0x{m.start():x}: "{s.decode("ascii", "replace")[:72]}"')
            if len(out) >= _MAX_HITS:
                break
    return out


def triage_brief(workspace: Path) -> str:
    """Static facts about the staged files. Best-effort: returns '' on error."""
    try:
        files = sorted(p for p in workspace.iterdir() if p.is_file())[:_MAX_FILES]
        if not files:
            return ""
        lines = ["", "TRIAGE (deterministic facts; verify anything you build on):"]
        for p in files:
            raw = p.read_bytes()
            desc = _elf_desc(raw[:64]) or "data"
            lines.append(f"- {p.name} ({len(raw)} bytes, {desc})")
            lines.extend(_string_hits(raw))
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — brief is best-effort seed sugar; never fail a run
        return ""
