"""Flask GUI for the CTF agent: Selection -> Runtime -> Report.

The UI is a small server-rendered app with three tabs:

* **Selection** — pick challenges, launch background runs.
* **Runtime** — watch the agent work: model messages, tool calls, verdicts.
* **Report** — aggregated pass@k / solve-rate / cost metrics over all runs.

Runs execute in a background worker so the browser is never blocked; the
Runtime tab polls a JSON endpoint that tails each run's ``trace.jsonl``.

Usage:
    $ uv run ctf-gui            # or: uv run python -m ctf_agent.flask_gui
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from ctf_agent.agent import Limits, run_agent
from ctf_agent.config import load_env
from ctf_agent.metrics import estimate_cost, load_attempts, summarize_attempts
from ctf_agent.runner import load_target, new_run_id, stage_workspace

# Read .env once at import so `flask run` works too (config.load_env never raises).
load_env()

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
CONFIG = Path("configs/scored_set.json")
BENCH_DATA = Path("benchmark/ctftiny.json")
RESULTS_ROOT = Path(os.environ.get("CTF_TUI_RESULTS", "results/tui"))
RUNS_ROOT = Path(os.environ.get("CTF_RUNS_DIR", "runs")) / "tui"

app = Flask(__name__)
app.secret_key = os.environ.get("CTF_FLASK_SECRET", "ctf-agent-dev-key")


# ---------------------------------------------------------------------------
#  Template filters / helpers
# ---------------------------------------------------------------------------
def _clip(value: Any, limit: int = 600) -> str:
    """Coerce to text and clip long strings (for live feeds / tables)."""
    if value is None:
        return ""
    text = str(value)
    return text if len(text) <= limit else text[:limit] + f" …[{len(text)} chars]"


app.add_template_filter(_clip, name="clip")


def load_challenges() -> list[str]:
    try:
        return json.loads(CONFIG.read_text())["challenges"]
    except Exception:  # noqa: BLE001
        return []


def bench_info(cid: str) -> dict[str, Any]:
    try:
        return json.loads(BENCH_DATA.read_text()).get(cid, {})
    except Exception:  # noqa: BLE001
        return {}


def make_limits() -> Limits:
    cfg = json.loads(CONFIG.read_text())["limits"]
    return Limits(
        max_steps=cfg["max_steps"],
        max_seconds=cfg["max_minutes"] * 60,
        tool_output_chars=cfg["tool_output_chars"],
        tool_timeout_s=cfg["tool_timeout_s"],
        max_output_tokens=cfg["max_output_tokens"],
        max_total_tokens=cfg.get("max_total_tokens"),
    )


# ---------------------------------------------------------------------------
#  Background job manager
# ---------------------------------------------------------------------------
@dataclass
class Job:
    cid: str
    run_id: str
    run_dir: Path
    status: str = "queued"          # queued | running | done | error
    verdict: str = ""
    solved: bool = False
    events: list[dict] = field(default_factory=list)   # parsed trace records
    model_calls: int = 0
    tool_calls: int = 0
    elapsed_s: float | None = None
    error: str = ""


JOBS: dict[str, Job] = {}
BATCH: list[str] = []                     # ordered ids of the most recent batch
SESSION_ATTEMPTS: list[dict[str, Any]] = []  # attempts made during this app session
_work: queue.Queue[str] = queue.Queue()
_lock = threading.Lock()
_worker_started = False


def _tail_trace(job: Job, trace: Path, stop: threading.Event) -> None:
    """Poll trace.jsonl and append parsed events to the job (live feed)."""
    pos = 0
    while not stop.is_set():
        try:
            if trace.exists():
                with trace.open() as fh:
                    fh.seek(pos)
                    while True:
                        line = fh.readline()
                        if not line.endswith("\n"):
                            break
                        pos = fh.tell()
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        with _lock:
                            job.events.append(rec)
                            if rec.get("kind") == "model":
                                job.model_calls += 1
                            elif rec.get("kind") == "tool":
                                job.tool_calls += 1
        except OSError:
            pass
        stop.wait(0.3)


def _execute(job: Job) -> None:
    """Run one attempt synchronously (called on the worker thread)."""
    job.status = "running"
    trial = 1
    dataset = os.environ.get("CTF_DATASET_JSON", "")
    run_dir = job.run_dir
    trace = run_dir / "trace.jsonl"
    started = time.monotonic()
    stop = threading.Event()
    tail = threading.Thread(target=_tail_trace, args=(job, trace, stop), daemon=True)
    tail.start()
    try:
        target = load_target(dataset, job.cid)
        workspace = stage_workspace(target, run_dir / "workspace")
        result = run_agent(target, workspace, make_limits(), trace)
        job.solved = bool(result.verified)
        job.verdict = "SOLVED" if result.verified else (result.status or "no_flag")
    except Exception as e:  # noqa: BLE001
        job.solved = False
        job.verdict = f"error: {type(e).__name__}: {e}"
        job.error = job.verdict
        result = None
    finally:
        stop.set()
        tail.join(timeout=2)

    job.elapsed_s = round(time.monotonic() - started, 2)

    # Persist attempt record (this is what makes the Report tab work).
    rec = {
        "run_id": run_dir.name,
        "challenge_id": job.cid,
        "trial": trial,
        "model": os.environ.get("CTF_MODEL"),
        "verified": job.solved,
        "status": (result.status if result else "tool_error"),
        "steps": (result.steps if result else None),
        "input_tokens": (result.input_tokens if result else None),
        "output_tokens": (result.output_tokens if result else None),
        "model_calls": (result.model_calls if result else job.model_calls),
        "tool_calls": (result.tool_calls if result else job.tool_calls),
        "estimated_cost_usd": (
            result.estimated_cost_usd if result
            else estimate_cost(
                result.input_tokens if result else None,
                result.output_tokens if result else None,
            )
        ),
        "elapsed_s": (result.elapsed_s if result else job.elapsed_s),
        "trace_file": str(trace),
    }
    # Session report: the Report tab only shows attempts made since the last
    # app start; the disk file keeps the permanent, cross-session record.
    with _lock:
        SESSION_ATTEMPTS.append(rec)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(rec, indent=2))
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        with (RESULTS_ROOT / "attempts.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass

    job.status = "error" if job.error else "done"


def _worker_loop() -> None:
    while True:
        run_id = _work.get()
        job = JOBS.get(run_id)
        if job is not None:
            try:
                _execute(job)
            except Exception as e:  # noqa: BLE001  (belt & braces)
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
        _work.task_done()


def _ensure_worker() -> None:
    global _worker_started
    if not _worker_started:
        threading.Thread(target=_worker_loop, daemon=True).start()
        _worker_started = True


def _start_jobs(cids: list[str]) -> list[Job]:
    jobs: list[Job] = []
    for cid in cids:
        run_id = f"{new_run_id(cid)}-t1"
        run_dir = RUNS_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        job = Job(cid=cid, run_id=run_id, run_dir=run_dir)
        JOBS[run_id] = job
        jobs.append(job)
        _work.put(run_id)
    with _lock:
        BATCH.clear()
        BATCH.extend(j.run_id for j in jobs)
    _ensure_worker()
    return jobs


def _job_payload(db: Job) -> dict[str, Any]:
    with _lock:
        events = list(db.events)
    out_events: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("kind") == "model":
            out_events.append({
                "kind": "model",
                "t": ev.get("t"),
                "content": _clip(ev.get("content"), 1500),
                "reasoning": _clip(ev.get("reasoning"), 1500),
                "tool_calls": [
                    {
                        "name": tc.get("function", {}).get("name", "?"),
                        "args": _clip(tc.get("function", {}).get("arguments", ""), 300),
                    }
                    for tc in (ev.get("tool_calls") or [])
                ],
                "in": ev.get("input_tokens"),
                "out": ev.get("output_tokens"),
            })
        elif ev.get("kind") == "tool":
            out_events.append({
                "kind": "tool",
                "t": ev.get("t"),
                "name": ev.get("name"),
                "result": _clip(ev.get("result"), 800),
            })
        elif ev.get("kind") == "submission":
            out_events.append({
                "kind": "submission",
                "t": ev.get("t"),
                "verified": bool(ev.get("verified")),
            })
        else:
            out_events.append({"kind": str(ev.get("kind", "?")), "t": ev.get("t"),
                               "result": _clip(ev.get("result"), 300)})
    return {
        "id": db.run_id,
        "cid": db.cid,
        "status": db.status,
        "verdict": db.verdict,
        "solved": db.solved,
        "error": db.error,
        "model_calls": db.model_calls,
        "tool_calls": db.tool_calls,
        "elapsed_s": db.elapsed_s,
        "events": out_events,
    }


# ---------------------------------------------------------------------------
#  Routes — Selection
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    challenges = load_challenges()
    infos = [(cid, bench_info(cid)) for cid in challenges]
    return render_template("selection.html", challenges=infos, active_tab="selection")


@app.post("/run/start")
def run_start():
    ids = request.form.getlist("challenge_ids") or (
        [request.form.get("cid")] if request.form.get("cid") else []
    )
    if not ids:
        flash("No challenges selected.", "warn")
        return redirect(url_for("index"))
    if not os.environ.get("CTF_DATASET_JSON"):
        flash("CTF_DATASET_JSON is not set — cannot load challenges.", "err")
        return redirect(url_for("index"))
    _start_jobs(ids)
    return redirect(url_for("runtime"))


# ---------------------------------------------------------------------------
#  Routes — Runtime
# ---------------------------------------------------------------------------
@app.route("/runtime")
def runtime():
    with _lock:
        ids = list(BATCH)
    if not ids:  # fall back to the most recent jobs (e.g. after a restart)
        ids = [j.run_id for j in sorted(
            JOBS.values(), key=lambda j: j.run_id, reverse=True)][:6]
    if not ids:
        return render_template("runtime.html", jobs=[], active_tab="runtime")
    return redirect(url_for("runtime_batch", ids=",".join(ids)))


@app.route("/runtime/<ids>")
def runtime_batch(ids: str):
    id_list = [i for i in ids.split(",") if i in JOBS]
    return render_template(
        "runtime.html",
        jobs=[_job_payload(JOBS[i]) for i in id_list],
        job_ids=id_list,
        active_tab="runtime",
    )


@app.get("/api/jobs")
def api_jobs():
    ids = request.args.get("ids", "")
    with _lock:
        id_list = [i for i in ids.split(",") if i in JOBS] or list(BATCH)
    return jsonify({"jobs": [_job_payload(JOBS[i]) for i in id_list]})


# Legacy synchronous single-run endpoint (kept for scripting/tests).
@app.route("/run/<cid>")
def run_sync(cid: str):
    dataset = os.environ.get("CTF_DATASET_JSON", "")
    if not dataset:
        flash("CTF_DATASET_JSON is not set — cannot load challenges.", "err")
        return redirect(url_for("index"))
    run_id = f"{new_run_id(cid)}-t1"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    job = Job(cid=cid, run_id=run_id, run_dir=run_dir)
    JOBS[run_id] = job
    _execute(job)  # run inline, no queueing
    return redirect(url_for("run_report", run_id=run_id))


# ---------------------------------------------------------------------------
#  Routes — Report
# ---------------------------------------------------------------------------
@app.route("/report")
def report():
    # Default scope is the current session: it starts empty every time the app
    # is launched and clears out again when the process dies. Pass
    # ?scope=history to see every attempt persisted for this results root.
    scope = "history" if request.args.get("scope") == "history" else "session"
    disk_attempts = load_attempts(RESULTS_ROOT)
    if scope == "history":
        attempts = disk_attempts
    else:
        with _lock:
            attempts = list(SESSION_ATTEMPTS)
    data = summarize_attempts(attempts) if attempts else None
    return render_template("report.html", data=data, n_attempts=len(attempts),
                           scope=scope, n_disk=len(disk_attempts),
                           results_root=RESULTS_ROOT, active_tab="report")


@app.route("/report/<run_id>")
def run_report(run_id: str):
    run_dir = RUNS_ROOT / run_id
    if not run_dir.is_dir():
        return f"Run {run_id!r} not found", 404
    try:
        summary = json.loads((run_dir / "summary.json").read_text())
    except Exception:  # noqa: BLE001
        summary = {}
    events = []
    trace = run_dir / "trace.jsonl"
    if trace.exists():
        for line in trace.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["result"] = _clip(rec.get("result"), 2000)
            rec["content"] = _clip(rec.get("content"), 4000)
            rec["reasoning"] = _clip(rec.get("reasoning"), 4000)
            events.append(rec)
    return render_template("run_detail.html", run_id=run_id, summary=summary,
                           events=events, active_tab="report")


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
