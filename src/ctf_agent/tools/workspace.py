"""File and execution tools, all confined to the workspace directory."""

from __future__ import annotations

from pathlib import Path

from ctf_agent.tools.sandbox import run_sandboxed


class ToolError(ValueError):
    pass


class Workspace:
    def __init__(self, root: Path, *, timeout_s: float, output_chars: int) -> None:
        self.root = root.resolve()
        self.timeout_s = timeout_s
        self.output_chars = output_chars

    def _resolve(self, rel: str) -> Path:
        if not rel or Path(rel).is_absolute():
            raise ToolError("path must be a relative path inside the workspace")
        p = (self.root / rel).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise ToolError("path escapes the workspace") from None
        return p

    def list_files(self) -> str:
        files = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file())
        return "\n".join(files) or "(empty)"

    def read_file(self, path: str, max_chars: int) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError("not a file")
        text = p.read_text(errors="replace")
        return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"

    def write_file(self, path: str, content: str) -> str:
        if len(content) > 20000:
            raise ToolError("content too large (max 20000 chars)")
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {p.relative_to(self.root)} ({len(content)} chars)"

    def run_python(self, path: str, stdin: str | None = None) -> str:
        p = self._resolve(path)
        if not p.is_file() or p.suffix != ".py":
            raise ToolError("run_python needs a .py file in the workspace")
        r = run_sandboxed(
            ["python3", "-I", str(p.relative_to(self.root))],
            self.root,
            timeout_s=self.timeout_s,
            output_chars=self.output_chars,
            stdin=stdin,
        )
        status = "timed out" if r.timed_out else f"exit {r.exit_code}"
        return f"[{status}]\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
