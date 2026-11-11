"""Command-line entry point: run one attempt on one challenge."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ctf_agent.agent import Limits, run_agent
from ctf_agent.config import load_env
from ctf_agent.runner import load_target, new_run_id, stage_workspace


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser(prog="ctf-agent")
    ap.add_argument("challenge_id")
    ap.add_argument("--dev", action="store_true", help="development run (not a reported attempt)")
    args = ap.parse_args(argv)

    target = load_target(os.environ["CTF_DATASET_JSON"], args.challenge_id)
    run_id = new_run_id(args.challenge_id)
    runs = Path(os.environ.get("CTF_RUNS_DIR", "./runs")) / run_id
    workspace = stage_workspace(target, runs / "workspace")
    limits = Limits(
        max_steps=int(os.environ.get("CTF_MAX_STEPS", "30")),
        max_seconds=float(os.environ.get("CTF_MAX_MINUTES", "15")) * 60,
        tool_output_chars=int(os.environ.get("CTF_TOOL_OUTPUT_CHARS", "4000")),
        tool_timeout_s=float(os.environ.get("CTF_TOOL_TIMEOUT_S", "60")),
        max_output_tokens=int(os.environ.get("CTF_MAX_OUTPUT_TOKENS", "1024")),
        max_total_tokens=int(os.environ["CTF_MAX_TOTAL_TOKENS"]) if os.environ.get("CTF_MAX_TOTAL_TOKENS") else None,
    )
    result = run_agent(
        target, workspace, limits, runs / "trace.jsonl",
        metadata={"challenge": args.challenge_id, "run_id": run_id,
                  "dev": args.dev, "model": os.environ.get("CTF_MODEL")},
    )
    summary = {
        "run_id": run_id,
        "challenge_id": args.challenge_id,
        "model": os.environ.get("CTF_MODEL"),
        "dev": args.dev,
        "verified": result.verified,
        "status": result.status,
        "steps": result.steps,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "model_calls": result.model_calls,
        "tool_calls": result.tool_calls,
        "estimated_cost_usd": result.estimated_cost_usd,
        "elapsed_s": result.elapsed_s,
        "trace_file": str(runs / "trace.jsonl"),
    }
    (runs / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
