"""Evaluator side of the feedback loop.

Between attempts of the same challenge, the orchestrator (GUI retry family)
builds a compact handoff note from the failed attempt's trace and hands it to
the next attempt as ``feedback`` (see run_agent). The note is DETERMINISTIC —
it costs zero model tokens to produce — and never contains the real flag: only
what the benchmark rejected (rejected candidates are safe to record), what was
tried, and how the attempt ended.

Batch runs deliberately do NOT use handoff notes: scored trials stay
independent so per-trial pass@k remains honest. The GUI retry chain is the
exploration surface where feedback is enabled (CTF_FEEDBACK_NOTE).
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

_MAX_NOTE_CHARS = 1800


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def build_handoff(trace_path: Path, trial: int | None = None) -> str | None:
    """Summarise one failed attempt's trace for the next attempt.

    Returns None when the trace is missing or unreadable — the next attempt
    then simply runs without feedback (never breaks a run on a note).
    """
    try:
        lines = [json.loads(l) for l in Path(trace_path).read_text().splitlines() if l.strip()]
    except Exception:  # noqa: BLE001  (missing/malformed trace)
        return None
    meta = next((e for e in lines if e.get("kind") == "meta"), {})
    done = next((e for e in reversed(lines) if e.get("kind") == "done"), {})

    tools = [e for e in lines if e.get("kind") == "tool"]
    names = collections.Counter(e.get("name", "?") for e in tools)
    errors = sum(1 for e in tools if str(e.get("result", "")).startswith("error"))
    distinct_results = {f"{e.get('name')}:{str(e.get('result'))[:120]}" for e in tools}
    repeats = len(tools) - len(distinct_results)
    rejected = [e.get("candidate", "")
                for e in lines if e.get("kind") == "submission" and not e.get("verified")]
    rejected = [r for r in rejected if r]   # legacy traces carry no candidate field
    n_subs = sum(1 for e in lines if e.get("kind") == "submission")

    which = f" (attempt {trial})" if trial is not None else ""
    outcome = done.get("status", "unknown outcome")
    steps = done.get("steps", "?")
    submits = done.get("submits", n_subs or "?")
    parts = [(f"Previous attempt{which} of {meta.get('challenge', 'this challenge')}: "
              f"ended with {outcome} after {steps} steps; {submits} flag submission(s).")]

    if names:
        mix = ", ".join(f"{n}×{c}" for n, c in names.most_common(5))
        parts.append(f"Tools used: {mix}.")
    if repeats:
        parts.append(f"{repeats} tool calls repeated earlier results — avoid re-deriving the same steps.")
    # Error ledger: the exact honest-but-deadly calls the predecessor repeated
    # (trace-evidenced retry trap class). This is the "what not to do" data:
    # mechanically derived from the trace, not advice.
    err_sigs = collections.Counter(
        _clip(str(e.get("result", "")).replace("\n", " "), 70)
        for e in tools if str(e.get("result", "")).startswith("error"))
    if err_sigs:
        sig = "; ".join(f"'{s[7:].strip() or 'error'}' ×{c}" for s, c in err_sigs.most_common(3))
        parts.append(f"Errors to stop repeating: {sig}.")
    if errors:
        parts.append(f"{errors} tool calls errored (scripts that failed last time need rewriting, not rerunning).")
    if rejected:
        rej = ", ".join(_clip(r.replace("\n", " "), 60) for r in rejected[:5])
        parts.append(f"Flags the benchmark REJECTED: {rej} — do not resubmit these.")

    # Artifact carry-over: the workspace is wiped between chain attempts, so
    # prior files exist only here. The predecessor's notes.md is the highest-
    # value distilled context available (the babycrypto solve came from such
    # a seed) — pass it verbatim (clipped), not paraphrased.
    art: dict[str, Any] | None = None
    try:
        art = json.loads((Path(trace_path).parent / "artifacts.json").read_text())
    except Exception:  # noqa: BLE001, S110 — older traces have no snapshot
        pass
    if isinstance(art, dict):
        files = art.get("files") or {}
        if files:
            listing = ", ".join(f"{n} ({s}B)" for n, s in list(files.items())[:6])
            parts.append(f"The workspace was RESET between attempts — everything from the prior "
                         f"one ({listing}) no longer exists; recreate any file you actually need "
                         "rather than assuming it exists.")
        notes = art.get("notes")
        if notes:
            parts.append("Your predecessor's notes (verbatim excerpt):\n" + _clip(str(notes).strip(), 900))

    tail = [e for e in reversed([e for e in lines if e.get("kind") == "tool"])]
    if tail:
        last: list[str] = []
        for e in tail[:3][::-1]:
            # first line(s) with actual content, ignoring runner scaffolding
            live = [s.strip() for s in str(e.get("result", "")).splitlines()
                    if s.strip() and s.strip() not in {"[exit 0]", "stdout:", "stderr:"}]
            last.append(f"{e.get('name')} → {_clip(' | '.join(live[:2]), 70)}")
        parts.append(f"Where it left off: {'; '.join(last)}")
    parts.append("Change the approach on any rejected/hypothesis path; "
                 "reuse only what verified as useful (files read, decodes that succeeded).")

    note = "\n".join(parts)
    if len(note) > _MAX_NOTE_CHARS:
        note = note[: _MAX_NOTE_CHARS - 1] + "…"
    return note
