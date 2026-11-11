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

    def read_file(self, path: str, max_chars: int, offset: int = 0) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError("not a file")
        raw = p.read_bytes()
        if b"\\x00" in raw[:8192]:
            return (f"[binary file, {len(raw)} bytes] Not shown. Use run_python to analyze it "
                    f"(e.g. subprocess 'file', 'strings', or 'objdump -d') and read only the "
                    f"parts you need. Offset 0 to {len(raw)}.")
        text = raw.decode("utf-8", "replace")
        start = max(0, offset)
        end = start + max_chars
        chunk = text[start:end]
        suffix = "...[truncated]" if len(text) > end else ""
        prefix = f"...[starts at {start}]" if start > 0 else ""
        return f"{prefix}{chunk}{suffix}"

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
            # -E -s (not -I): the env is already curated by the jail (no
            # PYTHON* vars set), but -I implies -P, which drops the script's
            # directory from sys.path — breaking sibling imports of modules
            # the challenge handout ships (e.g. Collision-Course's my_aes.py).
            ["python3", "-E", "-s", str(p.relative_to(self.root))],
            self.root,
            timeout_s=self.timeout_s,
            output_chars=self.output_chars,
            stdin=stdin,
        )
        status = "timed out" if r.timed_out else f"exit {r.exit_code}"
        return f"[{status}]\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    def run_command(self, cmd: str) -> str:
        """Run a shell command in the workspace. Restricted to basic analysis tools."""
        # Basic safety check to prevent accidental deletion or system-wide impact
        forbidden = [";", "&&", "||", ">", ">>", "2>", "pip", "apt", "rm -rf", "chmod"]
        # We allow pipes because they are essential for analysis
        if any(f in cmd for f in forbidden if f != "|"):
            raise ToolError(f"Command contains forbidden characters or tools: {forbidden}")

        r = run_sandboxed(
            ["/bin/sh", "-c", cmd],
            self.root,
            timeout_s=self.timeout_s,
            output_chars=self.output_chars,
        )
        status = "timed out" if r.timed_out else f"exit {r.exit_code}"
        return f"[{status}]\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
