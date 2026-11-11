"""Tests for the Flask GUI (selection / runtime / report / config tabs)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ctf_agent import flask_gui
from ctf_agent.flask_gui import app

ENV_KEYS = ["LITELLM_TOKEN", "LITELLM_BASE_URL", "CTF_MODEL", "CTF_MAX_STEPS",
            "CTF_MAX_MINUTES", "CTF_MAX_OUTPUT_TOKENS", "CTF_TOOL_OUTPUT_CHARS",
            "CTF_TOOL_TIMEOUT_S", "CTF_MAX_TOTAL_TOKENS"]


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
    monkeypatch.setattr(flask_gui, "DOTENV_PATH", tmp_path / ".env")
    flask_gui.SESSION_ATTEMPTS.clear()          # fresh app session per test
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
    _runs, results = env
    challenge = flask_gui.load_challenges()[0] if flask_gui.load_challenges() else "demo"
    client = app.test_client()
    client.post("/run/start", data={"challenge_ids": challenge})
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
    """?scope=history shows every attempt persisted under the results root."""
    _, results = env
    results.mkdir(parents=True, exist_ok=True)
    (results / "attempts.jsonl").write_text(json.dumps({
        "run_id": "x-t1", "challenge_id": "demo", "trial": 1, "verified": True,
        "status": "solved", "steps": 3, "input_tokens": 500, "output_tokens": 50,
        "model_calls": 5, "tool_calls": 4, "estimated_cost_usd": 0.001,
        "elapsed_s": 12.0,
    }) + "\n")
    client = app.test_client()
    resp = client.get("/report?scope=history")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    for needle in ["Pass@k", "pass@1", "Solved", "demo"]:
        assert needle in html
    # Default scope (session) stays empty when nothing ran in this session.
    resp = client.get("/report")
    assert "No attempts this session yet" in resp.get_data(as_text=True)


def test_report_is_session_scoped_and_clears_after_restart(env):
    """Default report shows only this session's runs; disk history remains."""
    _, results = env
    results.mkdir(parents=True, exist_ok=True)
    (results / "attempts.jsonl").write_text(json.dumps({
        "run_id": "old-t1", "challenge_id": "from-an-earlier-session", "trial": 1,
        "verified": False, "status": "no_flag",
    }) + "\n")
    challenge = flask_gui.load_challenges()[0] if flask_gui.load_challenges() else "demo"
    client = app.test_client()
    client.post("/run/start", data={"challenge_ids": challenge})
    job_id = next(iter(flask_gui.JOBS))
    for _ in range(100):
        payload = client.get(f"/api/jobs?ids={job_id}").get_json()["jobs"][0]
        if payload["status"] in {"done", "error"}:
            break
        import time

        time.sleep(0.05)
    # Session scope: the new run appears, the old-session run does not.
    html = client.get("/report").get_data(as_text=True)
    assert challenge in html
    assert "from-an-earlier-session" not in html
    assert "4 attempts" not in html  # session banner counts disk rows separately
    assert "2 attempts on disk overall" in html  # seeded row + this session's run
    # History scope still shows both — the disk record was never cleared.
    html = client.get("/report?scope=history").get_data(as_text=True)
    assert "from-an-earlier-session" in html and challenge in html


def test_report_run_detail(env):
    _runs, _ = env
    run_dir = flask_gui.RUNS_ROOT / "demo-test-t1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({"challenge_id": "demo"}))
    (run_dir / "trace.jsonl").write_text(
        json.dumps({"t": 0.1, "kind": "tool", "name": "list_files", "result": "a.txt"}) + "\n")
    client = app.test_client()
    resp = client.get("/report/demo-test-t1")
    assert resp.status_code == 200
    assert "list_files" in resp.get_data(as_text=True)


@pytest.fixture
def env_guard(monkeypatch):
    """Snapshot os.environ keys the /config tab mutates; restore after test."""
    before = {k: os.environ.get(k) for k in ENV_KEYS}
    yield
    for k, v in before.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_config_get_masks_api_key(env, env_guard, monkeypatch):
    monkeypatch.setenv("LITELLM_TOKEN", "sk-supersecret-abc99")
    html = app.test_client().get("/config").get_data(as_text=True)
    assert "…bc99" in html                        # only the last 4 chars are shown
    assert "sk-supersecret-abc99" not in html


def test_config_post_applies_runtime_overrides(env, env_guard):
    client = app.test_client()
    resp = client.post("/config", data={
        "model": "test-model-9", "max_steps": "12", "max_minutes": "9",
        "max_output_tokens": "256", "tool_output_chars": "500",
        "tool_timeout_s": "45", "max_total_tokens": "40000",
    })
    assert resp.status_code == 302
    assert os.environ["CTF_MODEL"] == "test-model-9"
    assert os.environ["CTF_MAX_STEPS"] == "12"
    assert os.environ["CTF_MAX_TOTAL_TOKENS"] == "40000"
    lim = flask_gui.make_limits()
    assert (lim.max_steps, lim.max_output_tokens, lim.max_total_tokens) == (12, 256, 40000)
    assert lim.max_seconds == 9 * 60
    assert not flask_gui.DOTENV_PATH.exists()     # not persisted without “remember”


def test_config_post_persists_with_remember(env, env_guard, tmp_path: Path):
    env_file = flask_gui.DOTENV_PATH
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("# keep this comment\nCTF_MODEL=old-model\n")
    client = app.test_client()
    client.post("/config", data={
        "model": "new-model", "api_key": "sk-fresh", "max_total_tokens": "none",
        "remember": "1",
    })
    text = env_file.read_text()
    assert "# keep this comment" in text
    assert "CTF_MODEL=new-model" in text and "old-model" not in text
    assert "LITELLM_TOKEN=sk-fresh" in text
    assert "CTF_MAX_TOTAL_TOKENS=0" in text       # “none” persists as 0
    assert os.environ["CTF_MAX_TOTAL_TOKENS"].lower() in {"0", "none"}
    assert flask_gui.make_limits().max_total_tokens is None


def test_config_post_rejects_bad_int(env, env_guard, monkeypatch):
    monkeypatch.setenv("LITELLM_TOKEN", "sk-keepme")
    before_steps = os.environ.get("CTF_MAX_STEPS")   # may come from the real .env
    resp = app.test_client().post("/config", data={"max_steps": "abc",
                                                   "api_key": "sk-change"})
    assert resp.status_code == 302
    assert os.environ.get("LITELLM_TOKEN") == "sk-keepme"   # all-or-nothing apply
    assert os.environ.get("CTF_MAX_STEPS") == before_steps  # nothing applied


def test_config_test_connection(env, env_guard, monkeypatch):
    monkeypatch.setattr(flask_gui, "_probe_gateway", lambda b, t: (True, "23 models listed"))
    html = app.test_client().post("/config/test", follow_redirects=True).get_data(as_text=True)
    assert "Gateway reachable" in html and "23 models listed" in html


def test_config_model_dropdown_lists_gateway_models(env, env_guard, monkeypatch):
    monkeypatch.setenv("CTF_MODEL", "gpt-oss:120b")
    monkeypatch.setattr(flask_gui, "_fetch_models",
                        lambda b, t: (["glm-5.3", "gpt-oss:120b", "kimi-k2.7-code"], ""))
    html = app.test_client().get("/config").get_data(as_text=True)
    assert "<select" in html and 'name="model"' in html
    assert '<option value="gpt-oss:120b" selected>' in html
    assert 'value="kimi-k2.7-code"' in html
    assert "not in catalog" not in html


def test_config_model_dropdown_marks_uncatalogued_current(env, env_guard, monkeypatch):
    monkeypatch.setenv("CTF_MODEL", "my-local-model")
    monkeypatch.setattr(flask_gui, "_fetch_models",
                        lambda b, t: (["gpt-oss:120b", "kimi-k2.7-code"], ""))
    html = app.test_client().get("/config").get_data(as_text=True)
    assert 'value="my-local-model" selected>' in html and "not in catalog" in html


def test_config_model_falls_back_to_text_input(env, env_guard, monkeypatch):
    monkeypatch.setenv("CTF_MODEL", "fallback-model")
    monkeypatch.setattr(flask_gui, "_fetch_models", lambda b, t: (None, "offline"))
    html = app.test_client().get("/config").get_data(as_text=True)
    assert "<select" not in html
    assert 'name="model"' in html and 'value="fallback-model"' in html
