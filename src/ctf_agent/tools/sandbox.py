"""Sandboxed execution for model-written code.

Runs commands with cwd fixed to the workspace, an empty environment, a
throwaway HOME, a hard timeout, and truncated output. Commands are executed in
a new session so the whole process group is killed on timeout.

This is NOT a container. It limits accidental reach, not a determined escape.
Docker is the stronger option once it is available to WSL.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


def _clip(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]", True


def run_sandboxed(
    argv: list[str],
    workspace: Path,
    *,
    timeout_s: float,
    output_chars: int,
    stdin: str | None = None,
) -> ExecResult:
    home = Path(tempfile.mkdtemp(prefix="ctf-home-", dir=workspace.parent))
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.Popen(
        argv,
        cwd=workspace,
        env=env,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        out, err = proc.communicate(
            input=stdin.encode() if stdin is not None else None, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
    out_s, t1 = _clip(out.decode("utf-8", "replace"), output_chars)
    err_s, t2 = _clip(err.decode("utf-8", "replace"), output_chars)
    return ExecResult(
        exit_code=None if timed_out else proc.returncode,
        stdout=out_s,
        stderr=err_s,
        timed_out=timed_out,
        truncated=t1 or t2,
    )
