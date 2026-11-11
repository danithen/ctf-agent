"""Unit tests for the controls the scored evaluation depends on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctf_agent.aggregate import pass_at_k, summarize
from ctf_agent.tools.sandbox import run_sandboxed
from ctf_agent.tools.workspace import ToolError, Workspace

# --- path confinement (solver must not reach files outside its workspace) ---

def test_workspace_rejects_parent_escape(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws", timeout_s=5, output_chars=1000)
    (tmp_path / "ws").mkdir()
    with pytest.raises(ToolError):
        ws._resolve("../secret.txt")


def test_workspace_rejects_absolute_path(tmp_path: Path) -> None:
    ws = Workspace(tmp_path, timeout_s=5, output_chars=1000)
    with pytest.raises(ToolError):
        ws._resolve("/etc/passwd")


def test_workspace_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")
    (root / "link").symlink_to(outside)
    ws = Workspace(root, timeout_s=5, output_chars=1000)
    with pytest.raises(ToolError):
        ws.read_file("link", 1000)


def test_workspace_allows_inside_file(tmp_path: Path) -> None:
    ws = Workspace(tmp_path, timeout_s=5, output_chars=1000)
    (tmp_path / "a.txt").write_text("hi")
    assert ws.read_file("a.txt", 1000) == "hi"


# --- sandboxed execution ---

def test_sandbox_kills_on_timeout(tmp_path: Path) -> None:
    r = run_sandboxed(["python3", "-c", "import time; time.sleep(30)"], tmp_path,
                      timeout_s=1, output_chars=1000)
    assert r.timed_out
    assert r.exit_code is None


def test_sandbox_truncates_output(tmp_path: Path) -> None:
    r = run_sandboxed(["python3", "-c", "print('x' * 5000)"], tmp_path,
                      timeout_s=5, output_chars=100)
    assert r.truncated
    assert len(r.stdout) < 200


def test_sandbox_has_empty_environment(tmp_path: Path) -> None:
    r = run_sandboxed(["python3", "-c", "import os; print(sorted(os.environ))"], tmp_path,
                      timeout_s=5, output_chars=2000)
    assert "HOME" in r.stdout  # only the fixed set, no inherited secrets
    assert "LITELLM_TOKEN" not in r.stdout


# --- flag verification ---

def test_target_check_exact_match() -> None:
    from ctf_agent.runner import Target

    t = Target.__new__(Target)
    t.expected_flag = "flag{abc}"
    assert t.check("flag{abc}")
    assert t.check("  flag{abc}\n")
    assert not t.check("flag{abd}")
    assert not t.check("")


# --- aggregation math (from the implementation guide's worked example) ---

def test_pass_at_k_bounds() -> None:
    assert pass_at_k(3, 0, 1) == 0.0
    assert pass_at_k(3, 3, 1) == 1.0
    assert pass_at_k(3, 2, 2) == 1.0
    assert abs(pass_at_k(3, 1, 1) - 1 / 3) < 1e-9


def test_summarize_matches_guide_example(tmp_path: Path) -> None:
    rows = [{"challenge_id": "a", "verified": v, "input_tokens": 100, "output_tokens": 10,
             "elapsed_s": 5.0, "status": "solved" if v else "no_flag"}
            for v in [True, False, True]]
    p = tmp_path / "attempts.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    s = summarize(p)["challenges"]["a"]
    assert round(s["success_rate_pct"], 1) == 66.7
    assert round(s["sample_sd_pp"], 1) == 57.7
    assert s["mean_total_tokens"] == 110


def test_summarize_unavailable_tokens_stay_none(tmp_path: Path) -> None:
    rows = [{"challenge_id": "a", "verified": False, "input_tokens": None, "output_tokens": 5,
             "elapsed_s": 1.0, "status": "tool_error"}]
    p = tmp_path / "attempts.jsonl"
    p.write_text(json.dumps(rows[0]))
    s = summarize(p)["challenges"]["a"]
    assert s["mean_total_tokens"] is None
