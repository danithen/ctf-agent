"""ctf-agent: tool-using LLM agent for the NYU CTF Tiny benchmark (CS 590.007)."""

from __future__ import annotations

from pathlib import Path


def _load_env() -> None:
    """Load .env from the project root if python-dotenv is available.

    Called with override=False so real environment variables always win.
    All entry points (CLI, batch, GUI, tests) import this package first, which
    makes .env the single source of truth for the gateway token, model, and
    run limits; a changed .env takes effect on the next process start.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (Path(__file__).resolve().parent.parent.parent / ".env",
                      Path.cwd() / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


_load_env()
