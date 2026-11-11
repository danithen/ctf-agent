# ctf-agent

A tool-using LLM agent harness for the NYU CTF Challenge (CS 590.007, Duke).
The agent solves captured-flag challenges from the [CTFTiny benchmark](https://arxiv.org/abs/2508.05674)
by calling an LLM through the Duke AI Gateway (LiteLLM), running model-proposed
shell/python analysis inside a sandboxed workspace, and verifying candidate flags
against the benchmark's hidden answers.

**Project report:** see [`REPORT.md`](REPORT.md) (architecture, experiments, full results table, contributions, citations).

## Architecture (one paragraph)

`ctf_agent.runner` loads benchmark targets and verifies flags (answers never enter
the agent's context). `ctf_agent.agent` runs the chat loop: a system prompt
encoding verified sandbox facts, a tool belt (`list_files`, `read_file`,
`run_command`, `run_python`, `write_file`, `submit_flag`), a repeat-call guard,
automatic context elision near the token budget, nudge-not-kill handling of
prose-only replies, budget-aware coaching near the step limit. All tool
execution goes through `tools/sandbox.py` (workspace-confined, empty env, hard
timeouts, truncated output). Every model/tool call is traced to a local
`trace.jsonl` and LangSmith. Traces carry a `meta` first line (challenge,
run_id, trial, model) and a `done` last line (status, verified, steps); the
LangSmith root run takes the same metadata plus `[challenge, attempt]` tags,
so traces are filterable per challenge.
`ctf_agent.metrics` / `aggregate` compute pass@k,
token, cost, and status summaries from `attempts.jsonl`. `ctf_agent.flask_gui`
provides a web UI (challenge selection / live runtime feed / report) and
`ctf_agent.batch` runs the scored set headlessly.

## Setup

```bash
uv sync --all-groups          # core + dev deps; add --extra bench for nyuctf
cp .env.example .env          # then fill in tokens (never committed)
sudo apt install bubblewrap   # required: the agent sandbox (Linux/WSL2)
```

Required in `.env`: `LITELLM_BASE_URL`, `LITELLM_TOKEN` (any OpenAI-compatible
provider — Duke AI Gateway with Duke VPN, or e.g. Ollama Cloud at
`https://ollama.com`), `CTF_MODEL`, `CTF_DATASET_JSON` (CTFTiny index), and
optionally LangSmith keys for cloud tracing. Challenge data lives in
`./benchmark/` (git-ignored; clone/download separately).

## Usage

Scored single challenge (writes to `results/<dir>/attempts.jsonl`):

```bash
uv run ctf-agent <challenge_id>            # reported attempt
uv run ctf-agent rev-checker --dev         # dev run, not reported
```

Batch scored set (`configs/scored_set.json`):

```bash
uv run python -m ctf_agent.batch
```

Web GUI (select challenges, watch live tool calls, view reports, change model /
API key / limits at runtime):

```bash
uv run ctf-flask-gui                       # http://127.0.0.1:5000
```

The **Config** tab updates the model ID (dropdown from the gateway catalog when
reachable, text box otherwise), API key, gateway URL, and all run limits — including
**retries per challenge** (a *model failure* — no/wrong flag, budget/step/timeout
verdict — re-runs on a clean slate up to _N_ extra times; all attempts are recorded,
the rest are skipped once one solves, and infrastructure errors stop the chain) — for
the current session (optionally persisted to `.env`); “Test connection” verifies the
gateway without spending tokens.

Metrics from run artifacts:

```bash
uv run python -m ctf_agent.aggregate results/scored/attempts.jsonl
```

## Fixed run limits (used for all reported attempts)

30 steps · 15 min · 150 000 total-token cap · 4 000 tool-output chars ·
60 s tool timeout · 1 024 max output tokens · model `gpt-5.4-mini`
(`configs/scored_set.json` + `config.json` captured alongside every batch).

## Testing

```bash
uv run python -m pytest tests/    # 19 tests: sandbox/workspace, gateway mock,
                                  # loop scaffolding (nudge/elide/budget), Flask GUI
uv run python -m ruff check .
```

Model-call tests use scripted fakes; nothing in the suite spends gateway tokens.

## Repository layout

```
src/ctf_agent/       runner, agent loop, llm gateway client, batch, metrics,
                     aggregate, cli, flask GUI + templates, legacy TUI
src/ctf_agent/tools/ workspace (confined file ops) and sandbox (subprocess jail)
configs/             scored_set.json (challenges x trials + limits)
tests/               pytest suites (sandbox, loop, GUI; fakes, no live calls)
results/             attempts.jsonl per experiment set + notes (artifacts local)
runs/                per-run artifacts (git-ignored)
```

## Credits & licenses

- Benchmark data & challenge loader: [CTFTiny / NYU CTF Bench](https://github.com/NYU-LLM-CTF/NYU-CTF-Bench)
  (GPL-2.0, AAAI'26 paper above) via the `nyuctf` package.
- LLM calls: Duke AI Gateway (LiteLLM) through `langchain-openai`; tracing via LangSmith.
- Flask, Textual, LangChain are used unmodified as dependencies.
- All agent logic, tool sandboxing, metrics, and reporting are original to this
  repository — see [`REPORT.md`](REPORT.md) §7 for the full reused-code and
  AI-assistance disclosure required by course policy.
