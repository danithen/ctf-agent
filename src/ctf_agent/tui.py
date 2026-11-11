"""Terminal UI: selection -> run -> report, with rendered metric graphs.

Usage:
    uv run ctf-tui
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

try:
    from textual_plotext import PlotextPlot
except Exception:  # noqa: BLE001  # pragma: no cover
    class PlotextPlot:
        def __init__(self):
            self.plt = None
        def __getattr__(self, name):
            raise AttributeError
        def __repr__(self):
            return "<PlotPlaceholder>"


from ctf_agent.agent import Limits, run_agent
from ctf_agent.config import load_env
from ctf_agent.metrics import (
    discover_result_dirs,
    summarize_result_dir,
)
from ctf_agent.runner import load_target, new_run_id, stage_workspace

CONFIG = Path("configs/scored_set.json")
BENCH_DATA = Path("benchmark/ctftiny.json")
RESULTS_ROOT = Path(os.environ.get("CTF_TUI_RESULTS", "results/tui"))
RUNS_ROOT = Path(os.environ.get("CTF_RUNS_DIR", ".")) / "tui"

RUNS_ROOT = Path(os.environ.get("CTF_RUNS_DIR", "./runs")) / "tui"


def _limits() -> Limits:
    cfg = json.loads(CONFIG.read_text())["limits"]
    return Limits(
        max_steps=cfg["max_steps"],
        max_seconds=cfg["max_minutes"] * 60,
        tool_output_chars=cfg["tool_output_chars"],
        tool_timeout_s=cfg["tool_timeout_s"],
        max_output_tokens=cfg["max_output_tokens"],
        max_total_tokens=cfg.get("max_total_tokens"),
    )


def _bench_info(cid: str) -> dict[str, Any]:
    try:
        all_data = json.loads(BENCH_DATA.read_text())
        return all_data.get(cid, {})
    except Exception:  # noqa: BLE001
        return {}


class SelectionScreen(Screen):
    """Challenge selection screen."""

    CSS = """
    Screen { background: #1a1b26; color: #a9b1d6; }
    #content { padding: 1 2; }
    #title { text-align: center; margin: 1 0; color: #7aa2f7; text-style: bold; }
    ListView { border: solid #414868; background: #24283b; height: 1fr; }
    .hint { color: #565f89; text-style: italic; margin-top: 1; }
    """
    BINDINGS = [  # noqa: RUF012
        Binding("space", "toggle", "Toggle", show=True),
        Binding("enter", "run_selected", "Run selected", show=True),
        Binding("r", "run_checked", "Run checked", show=True),
        Binding("a", "check_all", "All", show=False),
        Binding("p", "open_report", "Report", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="content"):
            yield Label(" ⚡ CTF Agent Challenge Selector ", id="title")
            yield ListView(id="challenge-list")
            yield Label("space=toggle  enter=run highlighted  r=run checked  p=report  q=quit", classes="hint")
        yield Footer()

    def on_mount(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        lv = self.query_one("#challenge-list", ListView)
        cursor = lv.index
        lv.clear()
        for cid in self.app.challenges:
            info = _bench_info(cid)
            mark = "\u2713" if cid in self.app.checked else " "
            label = f"{mark} {cid}  [{info.get('category', 'N/A')}]  {info.get('event', 'N/A')}  {info.get('year', 'N/A')}"
            lv.append(ListItem(Label(label)))
        if cursor is not None:
            lv.index = min(cursor, max(0, len(self.app.challenges) - 1))

    def action_toggle(self) -> None:
        idx = self.query_one("#challenge-list", ListView).index
        if idx is None or idx >= len(self.app.challenges):
            return
        cid = self.app.challenges[idx]
        if cid in self.app.checked:
            self.app.checked.discard(cid)
        else:
            self.app.checked.add(cid)
        self._rebuild()
        self.query_one("#challenge-list", ListView).index = idx

    def action_run_selected(self) -> None:
        idx = self.query_one("#challenge-list", ListView).index
        ids: list[str] = []
        if idx is not None and idx < len(self.app.challenges):
            ids = [self.app.challenges[idx]]
        if not ids:
            ids = sorted(self.app.checked)
        if not ids:
            ids = list(self.app.challenges)
        self.app.start_run(ids)

    def action_run_checked(self) -> None:
        ids = sorted(self.app.checked)
        if not ids:
            ids = list(self.app.challenges)
        self.app.start_run(ids)

    def action_check_all(self) -> None:
        self.app.checked = set(self.app.challenges)
        self._rebuild()

    def action_open_report(self) -> None:
        self.app.show_report()


class RunScreen(Screen):
    """Live run screen."""

    CSS = """
    Screen { background: #1a1b26; color: #a9b1d6; }
    #main-container { padding: 0 1; }
    #picker { width: 30; border: solid #414868; background: #24283b; margin-right: 1; }
    #picker Label { padding: 0 1; }
    #chat-area { border: solid #414868; background: #1a1b26; padding: 1 2; overflow-y: scroll; height: 1fr; }
    #status-bar { height: 1; padding: 0 1; background: #414868; color: #c0caf5; text-style: italic; }
    .challenge-item { padding: 0 1; }
    .turn-block { margin-bottom: 1; padding: 1; border-left: solid #414868; }
    .reasoning { color: #565f89; text-style: italic; margin-bottom: 1; }
    .model-text { color: #7aa2f7; margin-bottom: 1; }
    .tool-call { color: #e0af68; text-style: bold; margin-left: 1; }
    .tool-result { color: #9ece6a; margin-left: 2; background: #24283b; padding: 0 1; }
    .verdict-solved { color: #9ece6a; text-style: bold; background: #2d334a; padding: 1; text-align: center; }
    .verdict-failed { color: #f7768e; text-style: bold; background: #2d334a; padding: 1; text-align: center; }
    """
    BINDINGS = [  # noqa: RUF012
        Binding("esc", "go_back", "Back", show=True),
        Binding("p", "open_report", "Report", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main-container"):
            with Vertical(id="picker"):
                yield Label(" \u26a1 CHALLENGES ", id="picker-title")
                yield ListView(*[ListItem(Label(self.app._mark(c) + c, id=f"lbl-{c}"), classes="challenge-item") for c in self.app.challenges], id="challenge-list")
            yield Vertical(id="chat-area")
        yield Static("idle", id="status-bar")
        yield Footer()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_open_report(self) -> None:
        self.app.show_report()

    def _log_header(self, cid: str, trial: int | None = None) -> None:
        chat = self.query_one("#chat-area", Vertical)
        suffix = f" (trial {trial})" if trial is not None else ""
        chat.mount(Static(f"--- Evaluating {cid}{suffix} ---", id="header"))
        chat.scroll_end()

    def _log_verdict(self, verdict: str, is_solved: bool) -> None:
        chat = self.query_one("#chat-area", Vertical)
        cls = "verdict-solved" if is_solved else "verdict-failed"
        icon = "\u2705" if is_solved else "\u274c"
        chat.mount(Static(f"{icon} Result: {verdict}", classes=cls))
        chat.scroll_end()

    def _render_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("kind")
        chat = self.query_one("#chat-area", Vertical)

        if kind == "model":
            self.app.tokens_in += ev.get("input_tokens") or 0
            self.app.tokens_out += ev.get("output_tokens") or 0
            self.app.steps_done += 1

            turn = Vertical(classes="turn-block")
            reasoning = (ev.get("reasoning") or "").strip()
            if reasoning:
                turn.mount(Static(f"\U0001f4ad {reasoning}", classes="reasoning"))
            text = (ev.get("content") or "").strip()
            if text:
                turn.mount(Static(f"\U0001f916 {text}", classes="model-text"))
            for c in ev.get("tool_calls") or []:
                fn = c["function"]
                turn.mount(Static(f"\U0001f6e0\ufe0f  {fn['name']} {str(fn.get('arguments', ''))[:160]}", classes="tool-call"))
            chat.mount(turn)
            chat.scroll_end()

        elif kind == "tool":
            out = str(ev.get("result", "")).replace("\n", " | ")
            res = Static(f"\U0001f4e6 {ev.get('name')}: {out[:200]}", classes="tool-result")
            chat.mount(res)
            chat.scroll_end()

        elif kind == "submission":
            verified = ev.get("verified")
            icon = "\u2705" if verified else "\u274c"
            chat.mount(Static(f"{icon} Flag submission verified={verified}", classes="verdict-solved" if verified else "verdict-failed"))
            chat.scroll_end()

        self._status_bar()

    def _status(self, text: str) -> None:
        self.query_one("#status-bar", Static).update(text)

    def _status_bar(self) -> None:
        elapsed = time.monotonic() - self.app.started_at if self.app.started_at else 0
        self._status(
            f"Step {self.app.steps_done}/{self.app.max_steps}  "
            f"Tokens In: {self.app.tokens_in} Out: {self.app.tokens_out}  "
            f"Elapsed: {elapsed:0.0f}s"
        )


class ReportScreen(Screen):
    """Report screen with rendered metric graphs."""

    CSS = """
    Screen { background: #1a1b26; color: #a9b1d6; }
    #title { color: #7aa2f7; text-style: bold; margin: 1 0; }
    #left { width: 24; margin-right: 1; }
    #body { padding: 1 2; }
    #dataset-list { border: solid #414868; background: #24283b; height: 1fr; }
    PlotextPlot { height: 12; margin: 1 0; border: solid #414868; }
    .section { color: #bb9af7; text-style: bold; margin-top: 1; }
    .hint { color: #565f89; text-style: italic; }
    """
    BINDINGS = [  # noqa: RUF012
        Binding("esc", "go_back", "Back", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("enter", "load_selected", "Load", show=True),
    ]

    def __init__(self, results_dir: Path | None = None) -> None:
        super().__init__()
        self.requested_dir = Path(results_dir) if results_dir is not None else None
        self.result_dirs: list[Path] = []
        self.selected_index: int = 0
        self.data: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            with Vertical(id="left"):
                yield Label(" Datasets ", id="title")
                yield ListView(id="dataset-list")
                yield Label("enter=load  r=refresh", classes="hint")
            with ScrollableContainer(id="body"):
                yield Label("Loading report...", id="report-title")
                yield Static("", id="report-body")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_dirs()
        self._populate_list()
        self._load_selected()

    def _refresh_dirs(self) -> None:
        # Use RESULTS_ROOT to find datasets
        dirs = discover_result_dirs(RESULTS_ROOT)
        if self.requested_dir is not None and self.requested_dir.exists() and self.requested_dir not in dirs:
            dirs = [self.requested_dir] + dirs
        self.result_dirs = dirs
        if self.requested_dir is not None:
            for i, d in enumerate(dirs):
                if d == self.requested_dir:
                    self.selected_index = i
                    break

    def _populate_list(self) -> None:
        lv = self.query_one("#dataset-list", ListView)
        lv.clear()
        for d in self.result_dirs:
            lv.append(ListItem(Label(d.name if d.name else str(d))))
        if self.result_dirs:
            lv.index = min(self.selected_index, len(self.result_dirs) - 1)

    def action_refresh(self) -> None:
        self._refresh_dirs()
        self._populate_list()
        self._load_selected()

    def action_load_selected(self) -> None:
        idx = self.query_one("#dataset-list", ListView).index
        if idx is not None:
            self.selected_index = idx
        self._load_selected()

    def _load_selected(self) -> None:
        if not self.result_dirs:
            self.query_one("#report-body", Static).update("No result datasets found.")
            return
        idx = self.selected_index
        lv = self.query_one("#dataset-list", ListView)
        if lv.index is not None:
            idx = lv.index
        self.selected_index = idx
        d = self.result_dirs[idx]
        try:
            self.data = summarize_result_dir(d)
        except Exception as e:  # noqa: BLE001
            self.query_one("#report-body", Static).update(f"Failed to load {d}: {e}")
            return
        self._render()

    def _render(self) -> None:
        data = self.data
        if not data:
            return
        title = self.query_one("#report-title", Label)
        title.update(f"Configuration: {data.get('name', '')}")

        # Clear and rebuild body with rich text + plots
        container = self.query_one("#body", ScrollableContainer)
        # remove everything after title? keep it simple: remove all children then re-add title and widgets
        # We keep title label separate; rebuild below it.
        # Remove previously mounted report widgets except title
        for child in list(container.children):
            if child is not title and child.id != "report-title":
                child.remove()

        overall = data["overall"]
        per = data["challenges"]

        container.mount(Static(
            f"Challenges: {overall['challenges']}  Attempts: {overall['attempts']}  "
            f"Solved: {overall['successes']}  Avg solve rate: {overall.get('average_solve_rate_pct') or 0:.1f}% "
            f"(SD {overall.get('solve_rate_sd_pp') or 0:.1f} pp)"
        ))

        # Pass@k plot
        pass_vals = [100.0 * float(overall.get("pass_at_k", {}).get(k, 0.0)) for k in (1, 2, 3)]
        try:
            pass_plot = PlotextPlot()
            pp = pass_plot.plt
            pp.clear_figure()
            pp.bar(["Pass@1", "Pass@2", "Pass@3"], pass_vals)
            pp.ylim(0, 100)
            pp.title("Pass@k (average across challenges)")
            container.mount(Static("Pass@k", classes="section"))
            container.mount(pass_plot)
        except Exception:  # noqa: BLE001  # pragma: no cover
            container.mount(Static("Pass@k data unavailable", classes="section"))

        # Average solve rate per challenge
        if per:
            try:
                rate_plot = PlotextPlot()
                rp = rate_plot.plt
                rp.clear_figure()
                labels = [cid.split("-")[-1][:12] for cid in per]
                rates = [float(p.get("success_rate_pct", 0.0)) for p in per.values()]
                rp.bar(labels, rates)
                rp.ylim(0, 100)
                rp.title("Average solve rate per challenge (%)")
                container.mount(Static("Average solve rate", classes="section"))
                container.mount(rate_plot)
            except Exception:  # noqa: BLE001  # pragma: no cover
                container.mount(Static("Rate plot unavailable", classes="section"))

        # Per-challenge table
        lines = [
            "Challenge          Trials Solved Rate%   SDpp  Pass@1 Pass@2 Pass@3 ModelCalls ToolCalls MeanTok   MeanS  TotalS Cost",
        ]
        for cid, m in per.items():
            p1 = 100 * m["pass_at_k"].get(1, 0)
            p2 = 100 * m["pass_at_k"].get(2, 0)
            p3 = 100 * m["pass_at_k"].get(3, 0)
            sd = m.get("sample_sd_pp")
            sd_s = "-" if sd is None else f"{sd:.1f}"
            mc = m.get("mean_model_calls")
            tc = m.get("mean_tool_calls")
            mt = m.get("mean_total_tokens")
            mt_s = "-" if mt is None else f"{mt:.0f}"
            cost = m.get("total_cost_usd")
            cost_s = "-" if cost is None else f"{cost:.4f}"
            lines.append(
                f"{cid[:18]:18s} {m['trials']:6d} {m['successes']:6d} {m['success_rate_pct']:5.1f} "
                f"{sd_s:>5s} {p1:6.1f} {p2:6.1f} {p3:6.1f} "
                f"{(mc or 0):10.1f} {(tc or 0):9.1f} {mt_s:>7s} {m.get('mean_elapsed_s') or 0:6.1f} "
                f"{m.get('total_elapsed_s') or 0:6.1f} {cost_s}"
            )
        container.mount(Static("\n".join(lines), classes="section"))

        # Time and resources per attempt
        attempt_lines = ["Per-attempt (including failures):", "Challenge          Trial Status      Solved Steps  Model Tool   InTok  OutTok  Elapsed  Cost"]
        for cid, m in per.items():
            for a in m.get("attempts", []):
                a = a or {}
                solved = "Y" if a.get("verified") else "N"
                attempt_lines.append(
                    f"{cid[:18]:18s} {a.get('trial', '-')!s:>5s} {str(a.get('status', '-'))[:11]:11s} "
                    f"{solved:6s} {a.get('steps') or '-'!s:>5s} "
                    f"{a.get('model_calls') or '-'!s:>5s} {a.get('tool_calls') or '-'!s:>4s} "
                    f"{a.get('input_tokens') or '-'!s:>6s} {a.get('output_tokens') or '-'!s:>6s} "
                    f"{float(a.get('elapsed_s') or 0):7.1f} "
                    f"{(a.get('estimated_cost_usd') if a.get('estimated_cost_usd') is not None else 0):.4f}"
                )
        container.mount(Static("\n".join(attempt_lines), classes="section"))

        totals = [
            "Totals:",
            f"  wall-clock: {overall.get('total_elapsed_s') or 0:.1f}s",
            f"  model calls: {overall.get('total_model_calls') or 0}",
            f"  tool calls: {overall.get('total_tool_calls') or 0}",
            f"  input tokens: {overall.get('total_input_tokens') or 0}",
            f"  output tokens: {overall.get('total_output_tokens') or 0}",
            f"  estimated cost: {overall.get('total_cost_usd') if overall.get('total_cost_usd') is not None else 0:.4f} USD",
        ]
        container.mount(Static("\n".join(totals), classes="section"))


class CtfTui(App):
    def __init__(self) -> None:
        super().__init__()
        load_env()
        try:
            self.challenges: list[str] = json.loads(CONFIG.read_text())["challenges"]
        except Exception:  # noqa: BLE001
            self.challenges = []
        self.checked: set[str] = set()
        self.busy = False
        self.current_trace: Path | None = None
        self.tokens_in = 0
        self.tokens_out = 0
        self.started_at = 0.0
        self.steps_done = 0
        self.max_steps = 0
        self.trials = int(os.environ.get("CTF_TUI_TRIALS", "1"))
        self.last_run_dir: Path | None = None

    def on_mount(self) -> None:
        self.push_screen(SelectionScreen())

    def _mark(self, cid: str) -> str:
        return "\u2713 " if cid in self.checked else "  "

    def show_report(self, results_dir: Path | None = None) -> None:
        self.push_screen(ReportScreen(results_dir=results_dir or self.last_run_dir))

    def start_run(self, ids: list[str]) -> None:
        if self.busy:
            return
        self.busy = True
        self.push_screen(RunScreen())
        threading.Thread(target=self._run_all, args=(ids,), daemon=True).start()

    def _run_all(self, ids: list[str]) -> None:
        limits = _limits()
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        for cid in ids:
            for trial in range(1, self.trials + 1):
                if isinstance(self.screen, RunScreen):
                    self.call_from_thread(self.screen._log_header, cid, trial)
                self._run_one(cid, trial, limits)
        self.busy = False
        if isinstance(self.screen, RunScreen):
            self.call_from_thread(self.screen._status, "idle")
            if self.last_run_dir is not None:
                self.call_from_thread(self.push_screen, ReportScreen(self.last_run_dir))

    def _run_one(self, cid: str, trial: int, limits: Limits) -> None:
        run_dir = RUNS_ROOT / f"{new_run_id(cid)}-t{trial}"
        target = load_target(os.environ["CTF_DATASET_JSON"], cid)
        workspace = stage_workspace(target, run_dir / "workspace")
        trace = run_dir / "trace.jsonl"
        self.current_trace = trace
        self.tokens_in = 0
        self.tokens_out = 0
        self.steps_done = 0
        self.max_steps = limits.max_steps
        self.started_at = time.monotonic()

        stop = threading.Event()

        def tail() -> None:
            pos = 0
            while True:
                if trace.exists():
                    with trace.open() as fh:
                        fh.seek(pos)
                        while True:
                            line = fh.readline()
                            if not line.endswith("\n"):
                                break
                            pos = fh.tell()
                            if isinstance(self.screen, RunScreen):
                                try:
                                    self.call_from_thread(self.screen._render_event, json.loads(line))
                                except Exception:  # noqa: BLE001
                                    break
                if stop.is_set():
                    break
                time.sleep(0.4)

        t = threading.Thread(target=tail, daemon=True)
        t.start()
        try:
            result = run_agent(
                target, workspace, limits, trace,
                metadata={"challenge": cid, "run_id": run_dir.name,
                          "trial": trial, "model": os.environ.get("CTF_MODEL")},
            )
            verdict = "SOLVED" if result.verified else result.status
            is_solved = result.verified
        except Exception as e:  # noqa: BLE001
            result = None
            verdict = f"error: {type(e).__name__}: {e}"
            is_solved = False
        finally:
            stop.set()
            t.join(timeout=5)

        # Persist attempt record for metrics/reporting.
        rec = {
            "run_id": run_dir.name,
            "challenge_id": cid,
            "trial": trial,
            "model": os.environ.get("CTF_MODEL"),
            "verified": bool(result and result.verified),
            "status": (result.status if result else "tool_error"),
            "steps": (result.steps if result else None),
            "input_tokens": (result.input_tokens if result else None),
            "output_tokens": (result.output_tokens if result else None),
            "model_calls": (result.model_calls if result else None),
            "tool_calls": (result.tool_calls if result else None),
            "estimated_cost_usd": (result.estimated_cost_usd if result else None),
            "elapsed_s": (result.elapsed_s if result else round(time.monotonic() - self.started_at, 2)),
            "trace_file": str(trace),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(rec, indent=2))
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        attempts_path = RESULTS_ROOT / "attempts.jsonl"
        with attempts_path.open("a") as log:
            log.write(json.dumps(rec) + "\n")
            log.flush()
        self.last_run_dir = RESULTS_ROOT

        if isinstance(self.screen, RunScreen):
            self.call_from_thread(self.screen._log_verdict, verdict, is_solved)
            self.call_from_thread(self.screen._status, f"completed {cid} t{trial}: {verdict}")

    def _log_header(self, cid: str) -> None:
        if isinstance(self.screen, RunScreen):
            self.call_from_thread(self.screen._log_header, cid)


class SimpleReportScreen(Screen):
    """Minimal screen displaying summary JSON as plain text."""
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield ScrollableContainer(Static(self.text))
        yield Footer()

def main() -> None:
    CtfTui().run()


if __name__ == "__main__":
    main()
