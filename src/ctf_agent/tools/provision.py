"""Deterministic workspace-derived pre-flight: provision packages the handout
needs before the agent takes a single step.

Per-challenge provisioning done by hand is whack-a-mole and risks leaking
challenge knowledge into the setup. This module instead derives requirements
from exactly what the agent can see — the staged workspace — through a small
whitelisted signature map. Zero information advantage by construction: the
environment reflects only artifacts the agent could read anyway.

Correctness detail: the import probe runs THROUGH the bwrap jail (run_sandboxed
is the same path run_python uses), so "is this importable?" is answered by the
exact interpreter, PATH, and environment the agent will get — never the harness
venv. Installation is a host-level action (the harness has a network; the jailed
agent does not) targeting that same interpreter.

Security: package names come only from _SIGNATURES — text found in the
workspace can never name an arbitrary package to install.

Failures are recorded, never raised: a challenge whose requirement cannot be
satisfied (offline host, no pip, huge wheel timing out) still runs; the trace
records `provisioning: unavailable` so the report stays honest.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ctf_agent.tools.sandbox import run_sandboxed

# agent-visible signature -> (pip package, module the probe imports)
_PT_PACKAGES: dict[str, tuple[str, str]] = {
    ".pt": ("torch", "torch"),  # torch checkpoints/state dicts
}
_TEXT_PACKAGES: tuple[tuple[str, str, str], ...] = (
    # (text needle, pip package, probe module)
    ("import Crypto", "pycryptodome", "Crypto"),
    ("from Crypto", "pycryptodome", "Crypto"),
    ("import numpy", "numpy", "numpy"),
    ("from numpy", "numpy", "numpy"),
    ("import sympy", "sympy", "sympy"),
    ("from sympy", "sympy", "sympy"),
    ("import z3", "z3-solver", "z3"),
    ("import gmpy2", "gmpy2", "gmpy2"),
    ("import pwn", "pwntools", "pwn"),
    ("from pwn", "pwntools", "pwn"),
)
_TEXT_SUFFIXES = {".py", ".txt", ".md"}
_MAX_TEXT_CHARS = 100_000

_INSTALL_TIMEOUT_S = 600
_PROBE_TIMEOUT_S = 20

# The jail fixes PATH (see sandbox.py); resolve python3 exactly like it does.
_JAIL_PATH = "/usr/local/bin:/usr/bin:/bin"


@dataclass
class Preflight:
    """What the pre-flight found and what it did about it."""

    packages: dict[str, str] = field(default_factory=dict)  # pip pkg -> status
    python: str = ""  # interpreter the jail resolves (provisioning target)

    @property
    def summary(self) -> str:
        if not self.packages:
            return ""
        joined = ", ".join(f"{k}: {v}" for k, v in self.packages.items())
        return f"pre-flight provisioning ({joined})"


def scan_requirements(workspace: Path) -> list[tuple[str, str]]:
    """Whitelisted (pip package, probe module) pairs suggested by staged files.

    Only agent-visible artifacts are considered: handout files and, where they
    decode as text, their contents. Description files count — a description
    that tells the player to install a library is a real requirement signal.
    """
    found: dict[str, str] = {}
    for p in sorted(workspace.rglob("*")):
        if not p.is_file():
            continue
        need = _PT_PACKAGES.get(p.suffix.lower())
        if need:
            found[need[0]] = need[1]
        elif p.suffix.lower() in _TEXT_SUFFIXES:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")[
                    :_MAX_TEXT_CHARS
                ]
            except OSError:
                continue
            for needle, pip_pkg, mod in _TEXT_PACKAGES:
                if needle in text:
                    found[pip_pkg] = mod
    return sorted(found.items())


def preflight(workspace: Path) -> Preflight:
    """Probe (through the jail) and install (through the harness) requirements.

    Idempotent and cheap when everything is already importable: the scan is a
    directory walk and each probe one jailed `python3 -c` call.
    """
    result = Preflight()
    py = shutil.which("python3", path=_JAIL_PATH)
    result.python = py or "(not found)"
    reqs = scan_requirements(workspace)
    if not reqs:
        return result
    for pip_pkg, mod in reqs:
        status = _ensure_installed(pip_pkg, mod, workspace, py)
        result.packages[pip_pkg] = status
    return result


def _probe_import(mod: str, workspace: Path) -> bool:
    """Ask the jail itself whether the module is importable — the same
    interpreter and environment run_python will use."""
    out = run_sandboxed(
        ["python3", "-c", f"import {mod}"],
        workspace,
        timeout_s=_PROBE_TIMEOUT_S,
        output_chars=1000,
    )
    return out.exit_code == 0


def _ensure_installed(pip_pkg: str, mod: str, workspace: Path, py: str) -> str:
    """'present' | 'installed' | 'unavailable (<reason|stderr tail>)'."""
    if _probe_import(mod, workspace):
        return "present"
    if py is None:
        return "unavailable (no python3 on the jail PATH)"
    try:
        r = subprocess.run(
            [py, "-m", "pip", "install", "--quiet",
             "--disable-pip-version-check", pip_pkg],
            capture_output=True,
            text=True,
            timeout=_INSTALL_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError:
        return "unavailable (pip not present for the sandbox python)"
    except subprocess.TimeoutExpired:
        return f"unavailable (pip install exceeded {_INSTALL_TIMEOUT_S}s)"
    if _probe_import(mod, workspace):
        return "installed"
    tail = (r.stderr or "").strip().splitlines()
    reason = tail[-1][:200] if tail else "probe still failing after install"
    return f"unavailable ({reason})"
