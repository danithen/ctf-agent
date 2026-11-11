"""Scored batch runner: fixed config, fresh workspace per trial, sequential.

Usage:
    uv run python -m ctf_agent.batch configs/scored_set.json --trials 3 --out results/scored
Writes one JSON record per attempt to <out>/attempts.jsonl and a per-attempt
summary under <out>/runs/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ctf_agent.agent import Limits, run_agent
from ctf_agent.config import load_env
from ctf_agent.runner import load_target, new_run_id, stage_workspace


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_one(challenge_id: str, trial: int, limits: Limits, out: Path) -> dict:
    target = load_target(os.environ["CTF_DATASET_JSON"], challenge_id)
    run_id = f"{new_run_id(challenge_id)}-t{trial}"
    run_dir = out / "runs" / run_id
    workspace = stage_workspace(target, run_dir / "workspace")
    t_start = time.time()
    try:
        result = run_agent(target, workspace, limits, run_dir / "trace.jsonl")
        status, err = result.status, None
    except Exception as e:  # noqa: BLE001 - record infra failures as attempts, don't crash batch
        result = None
        status, err = "tool_error", f"{type(e).__name__}: {e}"
    rec = {
        "run_id": run_id,
        "challenge_id": challenge_id,
        "trial": trial,
        "model": os.environ.get("CTF_MODEL"),
        "verified": bool(result and result.verified),
        "status": status,
        "steps": result.steps if result else None,
        "input_tokens": result.input_tokens if result else None,
        "output_tokens": result.output_tokens if result else None,
        "model_calls": result.model_calls if result else None,
        "tool_calls": result.tool_calls if result else None,
        "estimated_cost_usd": result.estimated_cost_usd if result else None,
        "elapsed_s": result.elapsed_s if result else round(time.time() - t_start, 2),
        "error": err,
        "trace_file": str(run_dir / "trace.jsonl"),
    }
    (run_dir / "summary.json").write_text(json.dumps(rec, indent=2))
    return rec


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser(prog="ctf-batch")
    ap.add_argument("config")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="subset of challenge IDs (pilot runs)")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    challenges = args.only or cfg["challenges"]
    lim = cfg["limits"]
    limits = Limits(
        max_steps=lim["max_steps"],
        max_seconds=lim["max_minutes"] * 60,
        tool_output_chars=lim["tool_output_chars"],
        tool_timeout_s=lim["tool_timeout_s"],
        max_output_tokens=lim["max_output_tokens"],
        max_total_tokens=lim.get("max_total_tokens"),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "config_file": args.config,
        "benchmark_commit": cfg["benchmark_commit"],
        "agent_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "model": os.environ.get("CTF_MODEL"),
        "limits": lim,
        "challenges": challenges,
        "trials": args.trials,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2))

    log = (out / "attempts.jsonl").open("a")
    for challenge_id in challenges:
        for trial in range(1, args.trials + 1):
            rec = run_one(challenge_id, trial, limits, out)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"{challenge_id} t{trial}: {rec['status']} verified={rec['verified']}",
                  file=sys.stderr)
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
