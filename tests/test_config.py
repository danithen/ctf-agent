"""Tests for ctf_agent.config: .env load/save helpers."""

from __future__ import annotations

import os
from pathlib import Path

from ctf_agent.config import load_env, update_env


def test_load_env_uses_setdefault(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("A=1\nB=2\n# comment\nC=3\n")
    monkeypatch.setenv("B", "existing")
    load_env(env_file)
    assert os.environ["A"] == "1"
    assert os.environ["B"] == "existing"   # real environment wins
    assert os.environ["C"] == "3"


def test_update_env_replaces_and_appends(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("# header\nCTF_MODEL=old\nLITELLM_TOKEN=sk-stale\nCTF_MODEL=dup\n")
    update_env({"CTF_MODEL": "gpt-oss:120b", "NEW_KEY": "v"}, env_file)
    text = env_file.read_text()
    assert "# header" in text                      # untouched lines preserved
    assert "CTF_MODEL=old" not in text
    assert "CTF_MODEL=dup" not in text             # later duplicates dropped
    assert text.count("CTF_MODEL=gpt-oss:120b") == 1
    assert "CTF_MODEL=gpt-oss:120b\nLITELLM_TOKEN=sk-stale" in text
    assert "NEW_KEY=v" in text


def test_update_env_creates_file(tmp_path: Path):
    env_file = tmp_path / "nested" / ".env"
    update_env({"K": "1"}, env_file)
    assert env_file.read_text() == "K=1\n"
