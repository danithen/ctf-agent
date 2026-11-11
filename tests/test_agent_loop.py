"""Deterministic tests for agent-loop scaffolding: nudges, elision, budgets."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ctf_agent import agent as agent_mod
from ctf_agent.agent import Limits, run_agent
from ctf_agent.llm import ChatResult
from ctf_agent.runner import Target


def make_target(flag: str = "flag{abc}") -> Target:
    t = Target.__new__(Target)
    t.challenge = SimpleNamespace(name="unit-test")   # only .name is needed
    t.expected_flag = flag
    return t


def tool_call(i: int, name: str, args: str) -> dict:
    return {"id": f"call-{i}", "type": "function",
            "function": {"name": name, "arguments": args}}


class ScriptedChat:
    """Fake gateway: replays scripted responses, captures messages per call."""

    def __init__(self, responses: list[dict], in_tokens: int = 500, out_tokens: int = 50):
        self.responses = responses
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.seen: list[list[dict]] = []

    def __call__(self, messages, tools=None, **kw) -> ChatResult:
        self.seen.append(json.loads(json.dumps(messages)))  # snapshot
        try:
            resp = self.responses.pop(0)
        except IndexError:
            raise AssertionError("fake model got more calls than scripted")
        messages.append(resp["assistant"])
        return ChatResult(
            message=resp["assistant"],
            input_tokens=self.in_tokens,
            output_tokens=self.out_tokens,
        )


@pytest.fixture
def fake_chat(monkeypatch):
    def install(responses, in_tokens=500):
        sc = ScriptedChat(responses, in_tokens=in_tokens)
        monkeypatch.setattr(agent_mod, "chat", sc)
        return sc
    return install


@pytest.fixture
def fake_chat_spy(monkeypatch):
    """install(...) → (ScriptedChat, ...) with the whole message history per call.

    Only used by the evaluator-loop tests that assert on injected prompts."""
    holder: dict = {}
    def install(responses, in_tokens=500):
        sc = ScriptedChat(responses, in_tokens=in_tokens)
        holder.clear(); holder.update(fake=sc, seen=sc.seen)
        monkeypatch.setattr(agent_mod, "chat", sc)
        return sc
    return holder


def write_description(ws: Path, text: str = "Find the flag.") -> None:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "description.txt").write_text(text)


def basic_limits(max_steps: int, max_total_tokens: int | None = None) -> Limits:
    return Limits(max_steps=max_steps, max_seconds=120, tool_output_chars=4000,
                  tool_timeout_s=30, max_output_tokens=1024,
                  max_total_tokens=max_total_tokens)


def test_nudge_then_continue_then_solve(tmp_path: Path, fake_chat):
    ws = tmp_path / "ws"
    write_description(ws)
    sc = fake_chat([
        {"assistant": {"role": "assistant", "content": "Let me think about this...",
                       "tool_calls": []}},
        {"assistant": {"role": "assistant", "content": None,
                       "tool_calls": [tool_call(1, "list_files", "{}")]}},
        {"assistant": {"role": "assistant", "content": None,
                       "tool_calls": [tool_call(2, "submit_flag",
                                                '{"flag": "flag{abc}"}')]}},
    ])
    r = run_agent(make_target(), ws, basic_limits(4), tmp_path / "trace.jsonl")
    assert r.status == "solved" and r.verified
    # The prose reply must have produced a nudge note, not a dead run.
    assert any("No tool call" in (m.get("content") or "") and m["role"] == "user"
               for m in sc.seen[1])
    assert any(ev["kind"] == "nudge" for ev in r.events)


def test_no_flag_only_after_nudges_exhausted(tmp_path: Path, fake_chat):
    ws = tmp_path / "ws"
    write_description(ws)
    fake_chat([
        {"assistant": {"role": "assistant", "content": "hmm", "tool_calls": []}},
        {"assistant": {"role": "assistant", "content": "hmm again", "tool_calls": []}},
        {"assistant": {"role": "assistant", "content": "still thinking", "tool_calls": []}},
    ])
    r = run_agent(make_target(), ws, basic_limits(10), tmp_path / "trace.jsonl")
    assert r.status == "no_flag"
    assert r.steps == 3  # 1 original + 2 nudged retries
    assert sum(1 for ev in r.events if ev["kind"] == "nudge") == 2


def test_final_step_note_and_status_notes(tmp_path: Path, fake_chat):
    ws = tmp_path / "ws"
    write_description(ws)
    sc = fake_chat([{"assistant": {"role": "assistant", "content": None,
                                   "tool_calls": [tool_call(i, "list_files", "{}")]}}
                    for i in range(1, 4)])
    r = run_agent(make_target(), ws, basic_limits(3), tmp_path / "trace.jsonl")
    assert r.status == "step_limit"
    last = sc.seen[-1]
    assert any("FINAL step" in (m.get("content") or "") for m in last)
    assert any("Status: step" in (m.get("content") or "") for m in last)


def test_elision_stubs_old_outputs_and_allows_replay(tmp_path: Path, fake_chat):
    ws = tmp_path / "ws"
    write_description(ws)
    limits = basic_limits(2, max_total_tokens=20000)
    limits.tool_output_chars = 2000
    # Seven big distinct outputs in one step -> crossing half the budget elides
    # the oldest one; the scripted replay of its exact call must be allowed once.
    big = [tool_call(i, "run_command",
                     json.dumps({"cmd": f"python3 -c 'print(\"M{i}\" + \"A\"*2500)'"}))
           for i in range(7)]
    replay = tool_call(99, "run_command",
                       json.dumps({"cmd": "python3 -c 'print(\"M0\" + \"A\"*2500)'"}))
    sc = fake_chat([
        {"assistant": {"role": "assistant", "content": None, "tool_calls": big}},
        {"assistant": {"role": "assistant", "content": None, "tool_calls": [replay]}},
    ], in_tokens=11000)
    r = run_agent(make_target(), ws, limits, tmp_path / "trace.jsonl")
    assert any(ev["kind"] == "elide" for ev in r.events)
    # Stub visible to the second model call...
    assert any("chars elided" in (m.get("content") or "")
               for m in sc.seen[1] if m.get("role") == "tool")
    # ...and the repeated call was allowed: its result is real output, not a dedup error.
    results = [ev.get("result") or "" for ev in r.events
               if ev["kind"] == "tool" and ev["name"] == "run_command"]
    assert any("M0" in res and not res.startswith("error:") for res in results)
    assert sum(1 for ev in r.events
               if ev["kind"] == "tool" and (ev.get("result") or "").startswith("error:")) == 0


# ---------------------------------------------------------------------------
#  Trace metadata: meta header + done footer (runtime badge ⇄ trace reconciled)
# ---------------------------------------------------------------------------
def test_trace_recording_meta_and_done(tmp_path: Path, fake_chat):
    ws = tmp_path / "ws"
    write_description(ws)
    fake_chat([
        {"assistant": {"role": "assistant", "content": None,
                       "tool_calls": [tool_call(1, "submit_flag",
                                                '{"flag": "flag{abc}"}')]}},
    ])
    res = run_agent(make_target(), ws, basic_limits(10), tmp_path / "trace.jsonl",
                    metadata={"challenge": "rev-x", "run_id": "rev-x-test-t1",
                              "trial": 1, "max_attempts": 3, "model": "fake"})
    assert res.verified and res.status == "solved"
    lines = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines() if l]
    assert lines[0]["kind"] == "meta"
    assert lines[0]["challenge"] == "rev-x" and lines[0]["trial"] == 1
    assert lines[0]["run_id"] == "rev-x-test-t1" and lines[0]["model"] == "fake"
    assert lines[-1]["kind"] == "done"
    assert lines[-1]["status"] == "solved" and lines[-1]["verified"] is True
    assert lines[-1]["steps"] == res.steps


def test_trace_done_footer_reports_unsolved_status(tmp_path: Path, fake_chat):
    ws = tmp_path / "ws"
    write_description(ws)
    fake_chat([                       # prose only → nudges exhausted → no_flag
        {"assistant": {"role": "assistant", "content": "hmm", "tool_calls": []}},
        {"assistant": {"role": "assistant", "content": "hmm again", "tool_calls": []}},
        {"assistant": {"role": "assistant", "content": "still thinking", "tool_calls": []}},
    ])
    res = run_agent(make_target(), ws, basic_limits(10), tmp_path / "trace.jsonl",)
    assert res.status == "no_flag" and not res.verified
    lines = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines() if l]
    assert lines[-1]["kind"] == "done"
    assert lines[-1]["status"] == "no_flag" and lines[-1]["verified"] is False


def test_tracing_disabled_uses_no_run_tree(tmp_path: Path, fake_chat, monkeypatch):
    """With LANGSMITH_TRACING=false (conftest), _ls_attach must be a no-op."""
    ws = tmp_path / "ws"
    write_description(ws)
    fake_chat([
        {"assistant": {"role": "assistant", "content": None,
                       "tool_calls": [tool_call(1, "submit_flag",
                                                '{"flag": "flag{abc}"}')]}},
    ])
    # should neither raise nor attach anything
    res = run_agent(make_target(), ws, basic_limits(10), tmp_path / "trace.jsonl",
                    metadata={"challenge": "rev-x", "trial": 2})
    assert res.verified
# ---------------------------------------------------------------------------
#  Evaluator feedback loop: submission rejection with cap + halfway checkpoint
# ---------------------------------------------------------------------------
def test_wrong_submission_gives_feedback_and_continues(tmp_path: Path, fake_chat):
    """A rejected flag returns an evaluator-grade observation and the run
    continues — a better candidate can still land."""
    ws = tmp_path / "ws"
    write_description(ws)
    sc = fake_chat([
        {"assistant": {"role": "assistant", "content": None,
                       "tool_calls": [tool_call(1, "submit_flag", '{"flag": "flag{nope}"}')]}},
        {"assistant": {"role": "assistant", "content": None,
                       "tool_calls": [tool_call(2, "submit_flag", '{"flag": "flag{abc}"}')]}},
    ])
    res = run_agent(make_target(), ws, basic_limits(10), tmp_path / "trace.jsonl")
    assert res.verified and res.submissions == 2
    rej = next(m for m in sc.seen[-1] if isinstance(m, dict)
               and m.get("role") == "tool" and "INCORRECT" in str(m.get("content")))
    assert "flag{nope}" in rej["content"] and "1 of 3" in rej["content"]


def test_submission_cap_ends_run(tmp_path: Path, fake_chat):
    """After max_submits wrong guesses the run terminates (no guess-spam)."""
    ws = tmp_path / "ws"
    write_description(ws)
    calls = [tool_call(i + 1, "submit_flag", f'{{"flag": "flag{{wrong{i}}}"}}')
             for i in range(3)]
    fake_chat([
        {"assistant": {"role": "assistant", "content": None, "tool_calls": [c]}}
        for c in calls
    ])
    res = run_agent(make_target(), ws, basic_limits(10), tmp_path / "trace.jsonl")
    assert res.status == "no_flag" and not res.verified and res.submissions == 3
    lines = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines() if l]
    done = [e for e in lines if e.get("kind") == "done"][-1]
    assert done["submits"] == 3


def test_evaluator_checkpoint_fires_at_half_budget(tmp_path: Path, fake_chat):
    """Halfway with nothing submitted → deterministic consolidation nudge."""
    ws = tmp_path / "ws"
    write_description(ws)
    sc = fake_chat([
        {"assistant": {"role": "assistant", "content": None,
                       "tool_calls": [tool_call(1, "list_files", '{}')]}},
        {"assistant": {"role": "assistant", "content": None,
                       "tool_calls": [tool_call(2, "submit_flag", '{"flag": "flag{abc}"}')]}},
    ])
    res = run_agent(make_target(), ws, basic_limits(2), tmp_path / "trace.jsonl")
    assert res.verified
    assert any(isinstance(m, dict) and m.get("role") == "user"
               and "Evaluator checkpoint" in str(m.get("content"))
               for m in sc.seen[-1])


def test_feedback_preamble_injected(tmp_path: Path, fake_chat):
    """run_agent(feedback=…) surfaces the evaluator note before the first turn."""
    ws = tmp_path / "ws"
    write_description(ws)
    sc = fake_chat([
        {"assistant": {"role": "assistant", "content": None,
                       "tool_calls": [tool_call(1, "submit_flag", '{"flag": "flag{abc}"}')]}},
    ])
    res = run_agent(make_target(), ws, basic_limits(10), tmp_path / "trace.jsonl",
                    feedback="Previous attempt: ended with step_limit; it never submitted.")
    assert res.verified
    assert any("Previous attempt: ended with step_limit" in str(m.get("content"))
               for m in sc.seen[-1])
    lines = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines() if l]
    assert any(e.get("kind") == "feedback" for e in lines)
