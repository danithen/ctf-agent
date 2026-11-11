"""Tests for the Flask GUI (selection / runtime / report / config tabs)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from ctf_agent import flask_gui
from ctf_agent.flask_gui import app

ENV_KEYS = ["LITELLM_TOKEN", "LITELLM_BASE_URL", "CTF_MODEL", "CTF_MAX_STEPS",
            "CTF_MAX_MINUTES", "CTF_MAX_OUTPUT_TOKENS", "CTF_TOOL_OUTPUT_CHARS",
            "CTF_TOOL_TIMEOUT_S", "CTF_MAX_TOTAL_TOKENS", "CTF_MAX_RETRIES"]


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
    # Deterministic retries for every GUI test: the developer's real .env may
    # persist CTF_MAX_RETRIES (Config tab “Remember”), which must not leak in.
    monkeypatch.setenv("CTF_MAX_RETRIES", "0")
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


# ---------------------------------------------------------------------------
#  Retries — clean-slate re-runs until solved (or attempts exhausted)
# ---------------------------------------------------------------------------
def _drain(timeout: float = 15.0) -> None:
    """Wait until every enqueued job reached a terminal status."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with flask_gui._lock:
            jobs = list(flask_gui.JOBS.values())
        if jobs and all(j.status in {"done", "error"} for j in jobs):
            return
        time.sleep(0.02)
    pytest.fail(f"worker did not finish within {timeout}s")


def _dummy_run(solved: bool, status: str):
    r = DummyResult()
    r.verified = solved
    r.status = status
    return r


def test_start_jobs_precreates_retry_family(env, env_guard, monkeypatch):
    monkeypatch.setenv("CTF_MAX_RETRIES", "2")
    flask_gui.JOBS.clear()
    flask_gui.SESSION_ATTEMPTS.clear()
    app.test_client().post("/run/start", data={"challenge_ids": ["rev-x"]})
    ids = sorted(flask_gui.JOBS)
    assert ids == ["rev-x-test-t1", "rev-x-test-t2", "rev-x-test-t3"]
    assert [flask_gui.JOBS[i].attempt for i in ids] == [1, 2, 3]
    assert all(j.max_attempts == 3 and j.status == "queued"
               for j in flask_gui.JOBS.values())
    _drain()


def test_retry_runs_until_solved(env, env_guard, monkeypatch):
    """Two failures, then success: three real attempts recorded, last solved."""
    results = [_dummy_run(False, "no_flag"), _dummy_run(False, "token_budget"),
               _dummy_run(True, "solved")]
    got_meta: list[dict] = []

    def ra(target, workspace, limits, trace, *, metadata=None):
        got_meta.append(metadata)
        return results.pop(0)

    monkeypatch.setattr(flask_gui, "run_agent", ra)
    monkeypatch.setenv("CTF_MAX_RETRIES", "2")
    flask_gui.JOBS.clear()
    flask_gui.SESSION_ATTEMPTS.clear()
    app.test_client().post("/run/start", data={"challenge_ids": ["rev-x"]})
    _drain()
    order = ["rev-x-test-t1", "rev-x-test-t2", "rev-x-test-t3"]
    assert [flask_gui.JOBS[i].verdict for i in order] == ["no_flag", "token_budget", "solved"]
    assert [flask_gui.JOBS[i].solved for i in order] == [False, False, True]
    rows = (flask_gui.RESULTS_ROOT / "attempts.jsonl").read_text().splitlines()
    recs = [json.loads(r) for r in rows if r.strip()]
    assert [r["trial"] for r in recs] == [1, 2, 3]
    assert [r["verified"] for r in recs] == [False, False, True]
    assert len(flask_gui.SESSION_ATTEMPTS) == 3
    # metadata flows to the agent (challenge/run_id/trial/tags), attempt-indexed
    assert len(got_meta) == 3
    assert [m["trial"] for m in got_meta] == [1, 2, 3]
    assert all(m["challenge"] == "rev-x" for m in got_meta)
    assert got_meta[2]["run_id"] == "rev-x-test-t3"
    assert [m["max_attempts"] for m in got_meta] == [3, 3, 3]


def test_retry_skips_remaining_after_success(env, env_guard, monkeypatch):
    calls: list[int] = []

    def ra(*a, **k):
        calls.append(1)
        return _dummy_run(True, "solved")

    monkeypatch.setattr(flask_gui, "run_agent", ra)
    monkeypatch.setenv("CTF_MAX_RETRIES", "2")
    flask_gui.JOBS.clear()
    flask_gui.SESSION_ATTEMPTS.clear()
    app.test_client().post("/run/start", data={"challenge_ids": ["rev-x"]})
    _drain()
    assert len(calls) == 1                                   # solved on attempt 1
    t2, t3 = flask_gui.JOBS["rev-x-test-t2"], flask_gui.JOBS["rev-x-test-t3"]
    assert t2.status == "done" and t2.verdict == "skipped"
    assert t3.status == "done" and t3.verdict == "skipped"
    assert [j.solved for j in (t2, t3)] == [False, False]
    rows = [r for r in (flask_gui.RESULTS_ROOT / "attempts.jsonl").read_text().splitlines() if r]
    assert len(rows) == 1                                    # skipped attempts record nothing


def test_zero_retries_runs_once(env, env_guard, monkeypatch):
    monkeypatch.delenv("CTF_MAX_RETRIES", raising=False)
    flask_gui.JOBS.clear()
    flask_gui.SESSION_ATTEMPTS.clear()
    app.test_client().post("/run/start", data={"challenge_ids": ["rev-x"]})
    _drain()
    assert sorted(flask_gui.JOBS) == ["rev-x-test-t1"]
    assert flask_gui.JOBS["rev-x-test-t1"].verdict == "no_flag"   # dummy never solves
    rows = [r for r in (flask_gui.RESULTS_ROOT / "attempts.jsonl").read_text().splitlines() if r]
    assert len(rows) == 1


def test_config_retries_field(env, env_guard, monkeypatch):
    client = app.test_client()
    resp = client.post("/config", data={"retries": "2"})
    assert resp.status_code == 302
    assert os.environ.get("CTF_MAX_RETRIES") == "2"
    resp = client.post("/config", data={"retries": "-1"})
    html = client.get("/config").get_data(as_text=True)      # error flashed
    assert "must be 0 or a positive integer" in html
    client.post("/config", data={"retries": "3", "remember": "on"})
    assert "CTF_MAX_RETRIES=3" in flask_gui.DOTENV_PATH.read_text()


# ---------------------------------------------------------------------------
#  Trace metadata events (meta header / done footer) flow to the runtime feed
# ---------------------------------------------------------------------------
def test_job_payload_includes_meta_and_done_events(env, env_guard):
    job = flask_gui.Job(cid="rev-x", run_id="rev-x-test-t1",
                        run_dir=flask_gui.RUNS_ROOT / "rev-x-test-t1")
    with flask_gui._lock:
        job.events = [
            {"kind": "meta", "t": 0, "challenge": "rev-x", "run_id": "rev-x-test-t1",
             "trial": 1, "max_attempts": 3, "model": "fake"},
            {"kind": "done", "t": 12.5, "status": "solved", "verified": True, "steps": 7},
        ]
    evs = flask_gui._job_payload(job)["events"]
    meta, done = evs[0], evs[-1]
    assert meta["kind"] == "meta" and meta["challenge"] == "rev-x"
    assert meta["trial"] == 1 and meta["max_attempts"] == 3 and meta["model"] == "fake"
    assert done["kind"] == "done" and done["status"] == "solved"
    assert done["verified"] is True and done["steps"] == 7


def test_retry_stops_on_infra_error_not_model_failure(env, env_guard, monkeypatch):
    """Only MODEL failures repeat. An errored attempt (infra) stops the chain:
    no retry, later attempts cancelled, and nothing recorded for them."""
    calls: list[int] = []

    def ra(*a, **k):
        calls.append(1)
        raise RuntimeError("gateway unreachable")

    monkeypatch.setattr(flask_gui, "run_agent", ra)
    monkeypatch.setenv("CTF_MAX_RETRIES", "2")
    flask_gui.JOBS.clear()
    flask_gui.SESSION_ATTEMPTS.clear()
    app.test_client().post("/run/start", data={"challenge_ids": ["rev-x"]})
    _drain()
    assert len(calls) == 1                          # errored attempt never retried
    t2, t3 = flask_gui.JOBS["rev-x-test-t2"], flask_gui.JOBS["rev-x-test-t3"]
    for t in (t2, t3):
        assert t.status == "done" and not t.solved
        assert t.verdict.startswith("cancelled — an earlier attempt errored")
    rows = [r for r in (flask_gui.RESULTS_ROOT / "attempts.jsonl").read_text().splitlines() if r]
    assert len(rows) == 1                           # just the errored attempt, no retries


def test_model_failure_verdicts_still_retry(env, env_guard, monkeypatch):
    """Budget/step-limit verdicts are the model failing → the chain continues."""
    results = [_dummy_run(False, "token_budget"), _dummy_run(True, "solved")]
    monkeypatch.setattr(flask_gui, "run_agent", lambda *a, **k: results.pop(0))
    monkeypatch.setenv("CTF_MAX_RETRIES", "1")
    flask_gui.JOBS.clear()
    flask_gui.SESSION_ATTEMPTS.clear()
    app.test_client().post("/run/start", data={"challenge_ids": ["rev-x"]})
    _drain()
    assert [flask_gui.JOBS[i].verdict
            for i in ("rev-x-test-t1", "rev-x-test-t2")] == ["token_budget", "solved"]
    assert len(flask_gui.JOBS) == 2                 # t2 ran, nothing skipped/cancelled
