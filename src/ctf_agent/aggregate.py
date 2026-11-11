"""Aggregate attempts.jsonl into per-challenge and overall metrics.

Usage:
    uv run python -m ctf_agent.aggregate results/scored/attempts.jsonl
Prints a Markdown table and writes <file>.summary.json next to the input.

Metrics (per challenge, over its trials):
    successes / n, success rate %, sample SD (pp) of 0/1 outcomes,
    pass@k for k = 1..n (unbiased estimator over trials),
    mean input/output/total tokens (None if any trial lacks counts),
    mean elapsed seconds, and total elapsed.
Overall:
    pass@k and average solve rate across challenges.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import mean, stdev


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k draws (without replacement) from n trials is a success,
    given c successes among n. Standard unbiased estimator."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _tokens(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows]
    return None if any(v is None for v in vals) else mean(vals)


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_chal: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_chal[r["challenge_id"]].append(r)

    per = {}
    for cid, rs in sorted(by_chal.items()):
        n = len(rs)
        outcomes = [int(r["verified"]) for r in rs]
        c = sum(outcomes)
        ins, outs = _tokens(rs, "input_tokens"), _tokens(rs, "output_tokens")
        per[cid] = {
            "trials": n,
            "successes": c,
            "success_rate_pct": 100 * c / n,
            "sample_sd_pp": 100 * stdev(outcomes) if n > 1 else None,
            "pass_at_k": {k: pass_at_k(n, c, k) for k in range(1, n + 1)},
            "mean_input_tokens": ins,
            "mean_output_tokens": outs,
            "mean_total_tokens": None if ins is None or outs is None else ins + outs,
            "mean_elapsed_s": mean(r["elapsed_s"] for r in rs),
            "total_elapsed_s": sum(r["elapsed_s"] for r in rs),
            "statuses": [r["status"] for r in rs],
        }

    total_trials = sum(p["trials"] for p in per.values())
    total_succ = sum(p["successes"] for p in per.values())
    return {
        "challenges": per,
        "overall": {
            "challenges": len(per),
            "attempts": total_trials,
            "successes": total_succ,
            "average_solve_rate_pct": 100 * total_succ / total_trials if total_trials else None,
        },
    }


def to_markdown(s: dict) -> str:
    lines = [
        "| Challenge | Successes | Rate % | SD (pp) | Mean tokens | Mean s | Statuses |",
        "|---|---|---|---|---|---|---|",
    ]
    for cid, p in s["challenges"].items():
        sd = "-" if p["sample_sd_pp"] is None else f"{p['sample_sd_pp']:.1f}"
        tok = "n/a" if p["mean_total_tokens"] is None else f"{p['mean_total_tokens']:.0f}"
        lines.append(
            f"| {cid} | {p['successes']}/{p['trials']} | {p['success_rate_pct']:.1f} | {sd} "
            f"| {tok} | {p['mean_elapsed_s']:.1f} | {', '.join(p['statuses'])} |"
        )
    o = s["overall"]
    lines.append("")
    lines.append(f"Overall: {o['successes']}/{o['attempts']} attempts solved across "
                 f"{o['challenges']} challenges; average solve rate "
                 f"{o['average_solve_rate_pct']:.1f}%.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m ctf_agent.aggregate <attempts.jsonl>", file=sys.stderr)
        return 2
    path = Path(args[0])
    s = summarize(path)
    path.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(to_markdown(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
