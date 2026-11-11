"""Tests for the Flask GUI (selection / runtime / report tabs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctf_agent import flask_gui
from ctf_agent.flask_gui import app


class DummyResult:
    """Mimics ctf_agent.agent.Result for fast tests."""

    def __init__(self) -> None:
        self.verified = False
        self.status = "no_flag"
        self.steps = 2
        self.input_tokens = 100
        self.output_tokens = 10
        self.model_calls = 1
        self.tool_calls = 1
        self.estimated_cost_usd = 0.0001
        self.elapsed_s = 0.5


@pytest.fixture
def env(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    monkeypatch.setattr(flask_gui, "RUNS_ROOT", runs)
    monkeypatch.setattr(flask_gui, "RESULTS_ROOT", results)
    monkeypatch.setattr(flask_gui, "new_run_id", lambda cid: f"{cid}-test")
    monkeypatch.setattr(flask_gui, "run_agent", lambda *a, **k: DummyResult())

    def fake_target(dataset, cid):
        return {"id": cid}

    def fake_stage(target, ws_dir):
        Path(ws_dir).mkdir(parents=True, exist_ok=True)
        return Path(ws_dir)

    monkeypatch.setattr(flask_gui, "load_target", fake_target)
    monkeypatch.setattr(flask_gui, "stage_workspace", fake_stage)
    monkeypatch.setenv("CTF_DATASET_JSON", "benchmark/ctftiny.json")
    return runs, results


def test_selection_lists_challenges(env):
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "challenge_ids" in html          # checkbox form present
    assert "Run selected" in html


def test_run_start_creates_attempt_and_summary(env):
    runs, results = env
    challenge = flask_gui.load_challenges()[0] if flask_gui.load_challenges() else "demo"
    client = app.test_client()
    resp = client.post("/run/start", data={"challenge_ids": challenge})
    # Background job queued; poll the API until it finishes.
    job_id = next(iter(flask_gui.JOBS))
    for _ in range(100):
        payload = client.get(f"/api/jobs?ids={job_id}").get_json()["jobs"][0]
        if payload["status"] in {"done", "error"}:
            break
        import time

        time.sleep(0.05)
    assert payload["status"] == "done"
    # attempt record must be written for the Report tab
    assert (results / "attempts.jsonl").exists()
    rows = [json.loads(l) for l in (results / "attempts.jsonl").read_text().splitlines() if l]
    assert rows and rows[0]["challenge_id"] == challenge
    assert (flask_gui.RUNS_ROOT / f"{challenge}-test-t1" / "summary.json").name == "summary.json"


def test_report_renders_metrics(env):
    _, results = env
    results.mkdir(parents=True, exist_ok=True)
    (results / "attempts.jsonl").write_text(json.dumps({
        "run_id": "x-t1", "challenge_id": "demo", "trial": 1, "verified": True,
        "status": "solved", "steps": 3, "input_tokens": 500, "output_tokens": 50,
        "model_calls": 5, "tool_calls": 4, "estimated_cost_usd": 0.001,
        "elapsed_s": 12.0,
    }) + "\n")
    client = app.test_client()
    resp = client.get("/report")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    for needle in ["Pass@k", "pass@1", "Solved", "demo"]:
        assert needle in html


def test_report_run_detail(env):
    runs, _ = env
    run_dir = flask_gui.RUNS_ROOT / "demo-test-t1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({"challenge_id": "demo"}))
    (run_dir / "trace.jsonl").write_text(
        json.dumps({"t": 0.1, "kind": "tool", "name": "list_files", "result": "a.txt"}) + "\n")
    client = app.test_client()
    resp = client.get("/report/demo-test-t1")
    assert resp.status_code == 200
    assert "list_files" in resp.get_data(as_text=True)
