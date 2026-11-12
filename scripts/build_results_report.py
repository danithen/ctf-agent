"""Build results/RESULTS.md — the consolidated cross-era metrics report.

Bucket rule: every attempt is labeled by its OWN trace meta (limits string,
triage flag, dev flag) — never by ledger lore — so drifted-cap or unlabeled
runs cannot hide inside a comparable group. Ledgers carry tokens/cost; the
trace adds the ACI metrics (tool errors, error tax, oracle usage).

Regenerate with:  uv run python scripts/build_results_report.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ctf_agent.metrics import estimate_cost

RUNS = Path("runs/tui")
LEDGERS = [
    ("gpt-5.4-mini scored (frozen)", Path("results/scored/attempts.jsonl")),
    ("gpt-5.4-mini TUI dev chain", Path("results/tui/attempts.jsonl")),
    ("gpt-5.4-mini pilots (V1-V5 era)", Path("results/pilot/attempts.jsonl"))
    ,
    ("gpt-5.4-mini pilots", Path("results/pilot_checker/attempts.jsonl")),
    ("gpt-5.4-mini pilots", Path("results/pilot_baseline/attempts.jsonl")),
    ("gpt-oss:120b TUI chains", Path("results/attempts.jsonl")),
    ("gpt-oss:120b bulk", Path("results/ollama-gpt-oss-120b/attempts.jsonl")),
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        for l in path.read_text().splitlines():
            l = l.strip()
            if l:
                try:
                    rows.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return rows


def trace_meta(run_id: str) -> dict:
    """meta (limits/triage/dev/challenge) + done (status/steps/submits) + token/cost sums."""
    tp = RUNS / run_id / "trace.jsonl"
    out: dict = {"meta": None, "done": None, "in_tok": None, "out_tok": None}
    try:
        for l in tp.read_text().splitlines():
            try:
                rec = json.loads(l)
            except json.JSONDecodeError:
                continue
            k = rec.get("kind")
            if k == "meta" and out["meta"] is None:
                out["meta"] = rec
            elif k == "done" and out["done"] is None:
                out["done"] = rec
            elif k == "model":
                out["in_tok"] = (out["in_tok"] or 0) + (rec.get("input_tokens") or 0)
                out["out_tok"] = (out["out_tok"] or 0) + (rec.get("output_tokens") or 0)
            elif k == "tool":
                r = str(rec.get("result", ""))
                out["tool"] = out.get("tool", 0) + 1
                if r.startswith("error"):
                    out["errs"] = out.get("errs", 0) + 1
    except OSError:
        pass
    return out


def bucket(limits: str | None) -> str:
    """Canonical comparability buckets. Unlabeled runs stay OUT of headline tables."""
    if not limits:
        return "unlabeled (pre-meta harness)"
    l = limits.replace(" ", "")
    if l.startswith("30steps·150000"):
        return "fixed 30/150k"
    if l.startswith("100steps·150000"):
        return "loose 100/150k/10k"
    if l.startswith("100steps·300000"):
        return "drift 100/300k/10k"
    if l.startswith("100steps·1000000"):
        return "drift 100/1M/50k"
    if l.startswith("200steps·1000000"):
        return "bulk 200/1M/10k"
    return "other"


def main() -> int:
    attempts: dict[str, dict] = {}   # by run_id; ledger rows win for token/cost
    for label, path in LEDGERS:
        for r in read_jsonl(path):
            rid = r.get("run_id")
            if not rid:
                continue
            t = trace_meta(rid)
            m = t["meta"] or {}
            lim = m.get("limits")
            r["_bucket"] = bucket(lim)
            r["_label"] = label
            r["_triage"] = m.get("triage")
            r["_dev"] = bool(m.get("dev") or r.get("dev"))
            r["_trace"] = t
            attempts[rid] = r

    # Orphan traces (runs with local traces but no ledger row): synthesize rows,
    # bucketed; excluded from comparable headline tables if unlabeled/dev.
    for tp in sorted(RUNS.glob("*/trace.jsonl")):
        rid = tp.parent.name
        if rid in attempts:
            continue
        t = trace_meta(rid)
        if not t["meta"] and not t["done"]:
            continue
        m = t["meta"] or {}
        done = t["done"] or {}
        steps = int(done.get("steps", 0) or 0)
        in_tok, out_tok = int(t["in_tok"] or 0), int(t["out_tok"] or 0)
        attempts[rid] = {
            "run_id": rid, "challenge_id": m.get("challenge") or rid.rsplit("-2026", 1)[0],
            "model": m.get("model", "unknown"), "verified": bool(done.get("verified")),
            "status": done.get("status") or "interrupted (no done event)",
            "steps": steps, "submissions": done.get("submits"),
            "input_tokens": in_tok or None, "output_tokens": out_tok or None,
            "elapsed_s": None, "estimated_cost_usd": estimate_cost(in_tok, out_tok),
            "_trace": t, "_bucket": bucket(m.get("limits")), "_label": "trace-only (no ledger)",
            "_triage": m.get("triage"), "_dev": bool(m.get("dev")),
        }

    # Exclusions: dev runs, rev-checker self-test, tool_error infrastructure rows.
    for r in attempts.values():
        if r["_dev"] or r.get("challenge_id") in {"rev-checker"} or r["status"] == "tool_error":
            r["_exclude"] = "dev/infra"
        else:
            r["_exclude"] = None

    scored = [r for r in attempts.values() if not r["_exclude"]]

    def agg(rows: list[dict]) -> dict:
        n = len(rows)
        c = sum(1 for r in rows if r.get("verified"))
        steps = [float(r["steps"]) for r in rows if r.get("steps") is not None]
        toks = [float((r.get("input_tokens") or 0) + (r.get("output_tokens") or 0))
                for r in rows if r.get("input_tokens") is not None]
        errs = [r["_trace"] for r in rows if r.get("_trace")]
        err_n = sum(e.get("errs", 0) for e in errs)
        step_n = sum(r.get("steps", 0) or 0 for r in rows)
        cost = sum(float(r.get("estimated_cost_usd") or 0) for r in rows)
        return {"attempts": n, "solves": c, "rate": 100 * c / n, "steps": steps,
                "toks": toks, "err": err_n,
                "tax": (100 * err_n / step_n) if step_n else -1.0, "cost": cost}

    buckets = defaultdict(list)
    for r in scored:
        buckets[(r.get("model"), r["_bucket"])].append(r)
    # comparable = meta-labeled buckets only
    comparable = [(b, agg(rs)) for b, rs in sorted(buckets.items()) if b[1] != "unlabeled (pre-meta harness)"]
    unlabeled = agg([r for r in scored if r["_bucket"] == "unlabeled (pre-meta harness)"])

    per_chal = defaultdict(lambda: defaultdict(list))
    for r in scored:
        per_chal[r["challenge_id"]][r["_bucket"]].append(r)

    solves = sorted([r for r in scored if r.get("verified")], key=lambda r: r["run_id"])

    lines: list[str] = ["# Consolidated Results (all recorded attempts)",
                        "",
                        "Every attempt is labeled by its own trace `meta` (caps, triage flag, dev flag) —",
                        "never by which ledger it happens to sit in. Regenerate: `uv run python scripts/build_results_report.py`.",
                        ""]

    lines += ["## 1. Overall by configuration (comparable buckets)", "",
              "| Model | Caps | Attempts | Solves | Solve % | Mean steps | Mean tok | Error tax | Cost $ |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for (model, b), a in comparable:
        tax = "—" if a["tax"] < 0 else f"{a['tax']:.0f}%"
        lines.append(
            f"| {model} | {b} | {a['attempts']} | {a['solves']} | {a['rate']:.0f}% "
            f"| {sum(a['steps']) / max(len(a['steps']), 1):.1f} "
            f"| {sum(a['toks']) / max(len(a['toks']), 1):,.0f} "
            f"| {tax} | {a['cost']:.3f} |")
    lines += ["",
              ("Unlabeled pre-meta attempts (excluded from comparable tables): "
               f"{unlabeled['attempts']} attempts ({unlabeled['solves']} solves) across challenges "
               f"({', '.join(sorted({r['challenge_id'] for r in scored if r['_bucket'].startswith('unlabeled')}))})."),
              ""]

    lines += ["## 2. Per-challenge × caps matrix", "",
              "| Challenge | " + " | ".join(b for (_, b), _ in comparable) + " |",
              "|---|" + "---|" * len(comparable)]
    for cid in sorted(per_chal):
        row = [cid]
        for (_, b), _a in comparable:
            rs = per_chal[cid].get(b, [])
            row.append(f"{sum(1 for r in rs if r.get('verified'))}/{len(rs)}" if rs else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 3. Outcome mix", "",
              "| Caps | " + " | ".join(b for (_, b), _ in comparable) + " |", "|---|" + "---|" * len(comparable)]
    statuses = sorted({r["status"] for r in scored if r["status"]})
    for st in statuses:
        row = [st]
        for (_, b), _a in comparable:
            row.append(str(sum(1 for r in scored if r["_bucket"] == b and r["status"] == st)))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 4. Every solve (the dossier)", "",
              "| Run | Challenge | Caps | Steps | Tokens | Cost $ |", "|---|---|---|---|---|---|"]
    for r in solves:
        tok = (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0)
        lines.append(f"| {r['run_id']} | {r['challenge_id']} | {r['_bucket']} "
                     f"| {r.get('steps')} | {tok:,} | {float(r.get('estimated_cost_usd') or 0):.4f} |")
    n_all = sum(1 for r in scored)
    lines += ["", (f"**Totals: {sum(1 for r in scored if r.get('verified'))} solves in {n_all} "
                   "scored attempts.** Dev/rev-checker/tool_error rows are excluded and listed "
                   "in `results/full_summary.json`.")]
    Path("results/RESULTS.md").write_text("\n".join(lines) + "\n")

    # machine-readable companion
    summary = {
        "buckets": {f"{m} | {b}": a for (m, b), a in comparable},
        "unlabeled": unlabeled,
        "per_challenge": {cid: {b: agg(rs) for b, rs in sorted(bch.items())}
                          for cid, bch in sorted(per_chal.items())},
        "solves": [{k: r.get(k) for k in ("run_id", "challenge_id", "steps", "status")}
                   for r in solves],
        "excluded": [{"run_id": r["run_id"], "why": r["_exclude"]}
                     for r in attempts.values() if r["_exclude"]],
        "attempts_total_with_meta": len(attempts),
    }
    Path("results/full_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote results/RESULTS.md ({n_all} scored attempts, {len(solves)} solves) "
          f"+ results/full_summary.json ({len(attempts)} attempts incl. orphans)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
