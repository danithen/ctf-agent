"""Sandboxed execution for model-written code.

Runs commands inside bubblewrap (bwrap): only system binaries (/usr, /lib,
/lib64, /bin, /etc — read-only), /dev, and the workspace (read-write) are
visible inside the container. User, network, pid, ipc and uts namespaces are
all isolated, the environment is minimal, HOME is a throwaway bound next to
the workspace, and output is truncated.

Consequently the agent cannot read anything outside its workspace — including
the benchmark dataset and its per-challenge challenge.json (which holds the
official flag). Submitting the flag to `Target.check` is the only way a run
can pass, and `Target.check` is reachable only through the agent's
`submit_flag` tool.

bubblewrap is a hard requirement; there is no weaker silent fallback
(a fallback would quietly invalidate the guarantee above). Install with
`sudo apt install bubblewrap` if it is missing.
"""

from __future__ import annotations

import os
import shutil
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


def _bwrap_argv(workspace: Path, home: Path) -> list[str]:
    """bubblewrap wrapper: system libs/bins ro, workspace+HOME rw, rest hidden."""
    cmd: list[str] = ["bwrap", "--unshare-all", "--die-with-parent"]
    for ro in ("/usr", "/lib", "/lib64", "/bin", "/etc"):
        cmd += ["--ro-bind-try", ro, ro]
    cmd += [
        "--dev-bind", "/dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--tmpfs", "/var",
        "--tmpfs", "/run",
        "--bind", str(workspace), str(workspace),
        "--bind", str(home), str(home),
        "--chdir", str(workspace),
    ]
    return cmd


def run_sandboxed(
    argv: list[str],
    workspace: Path,
    *,
    timeout_s: float,
    output_chars: int,
    stdin: str | None = None,
) -> ExecResult:
    if shutil.which("bwrap") is None:
        raise RuntimeError(
            "bubblewrap (bwrap) is required to run the agent sandbox — "
            "install it with 'sudo apt install bubblewrap'."
        )
    home = Path(tempfile.mkdtemp(prefix="ctf-home-", dir=workspace.parent))
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.Popen(
        _bwrap_argv(workspace, home) + argv,
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
