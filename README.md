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
**retries per challenge** (see the evaluator loop below), **flags-in-run cap**
(`CTF_MAX_SUBMITS`, default 3) and an **on/off switch for retry handoff notes** — for
the current session (optionally persisted to `.env`); “Test connection” verifies the
gateway without spending tokens.

### Evaluator loop

The agent loop ships two deterministic evaluator mechanisms (no extra model calls,
zero tokens):

- **In-run feedback.** A rejected flag returns `INCORRECT (k of N)` with the rejected
  candidate echoed back, so the model can correct course within the run. Submissions
  are capped (`CTF_MAX_SUBMITS`, default 3); busting the cap ends the run as `no_flag`
  instead of letting it guess forever. Escalations key on the **binding** budget
  dimension — steps or tokens, whichever dies first: a halfway consolidation checkpoint
  fires once at 50 % of that binding budget, and an urgent near-limit warning fires at
  85 % of the token cap telling the agent to submit its best candidate now.
- **Between-attempt handoff (GUI retry chains only).** When a retry runs after a model
  failure, a deterministic note is distilled from the failed attempt’s trace — outcome,
  step/tool mix, repeated and errored tool calls, every rejected flag (“do not
  resubmit”) and where it left off — and injected before the next attempt’s first turn.
  It never contains the flag and costs no tokens to produce; disable with
  `CTF_FEEDBACK_NOTE=off`. Batch/scored runs deliberately do **not** use handoff notes:
  their retries stay clean-slate so pass@k measurements stay honest. (If you enable
  notes for scored runs, label the results as a different ablation.)

Every trace's first line (`meta`) also records the exact caps the run ran under — if a
run card shows `100 steps · 300000 tokens`, that run really did use those limits: check
the Config tab (or `.env`, which it persists to) before comparing runs.

Metrics from run artifacts:

```bash
uv run python -m ctf_agent.aggregate results/scored/attempts.jsonl
```

## Containers

The project ships as a reproducible container (rebuild it anywhere and get the
same software — base images are pinned by digest, and `uv sync --frozen`
installs exactly the versions in `uv.lock`):

```bash
docker build --target test -t ctf-agent:test .   # build + run the pytest gate (no network, no tokens)
docker build -t ctf-agent:latest .               # runtime image (GUI + CLI + server)
docker compose up --build                        # GUI on http://localhost:5000
docker compose down
```

The image deliberately excludes three machine-local things — `.env` (secrets),
`benchmark/` (course dataset), and `runs/`+`results/` (your data) — which
`compose.yaml` mounts read-only or shared at runtime:

| Host | Container | Mounted as |
|---|---|---|
| `./.env` | `/app/.env` | read-only (secrets injected at runtime) |
| `./benchmark` | `/data/benchmark` | read-only |
| `./runs`, `./results` | `/data/runs`, `/data/results` | shared — container runs appear in host reports |

CLI inside the container (secrets from `.env`, container paths forced with
`-e`, which overrides `--env-file`):

```bash
docker run --rm --security-opt seccomp=unconfined --env-file .env \
  -e CTF_RUNS_DIR=/data/runs -e CTF_TUI_RESULTS=/data/results \
  -e CTF_DATASET_JSON=/data/benchmark/ctftiny.json -e CTF_BENCHMARK_DIR=/data/benchmark \
  -v "$PWD/benchmark:/data/benchmark:ro" -v "$PWD/runs:/data/runs" -v "$PWD/results:/data/results" \
  ctf-agent:latest ctf-agent rev-checker --dev
```

**Why `seccomp=unconfined`?** Docker's default seccomp profile blocks the
user-namespace creation bubblewrap needs (`bwrap: No permissions to create new
namespace`). Dropping the *outer* filter does not weaken the agent jail: what
confines the agent is bwrap itself — no network, isolated pid/mount spaces,
and a filesystem limited to workspace + read-only system dirs — while the
container keeps cgroups and capability separation. The sandbox's own tests run
inside this configuration in the `--target test` gate.

Moving machines offline (no registry): `docker save ctf-agent:latest | gzip >
ctf-agent.tar.gz`, copy the file and `docker load < ctf-agent.tar.gz` on the
target machine; with a registry, `docker tag`/`push` the same image. In both
cases bring the same `.env` + `benchmark/` data (never in the image) and runs
continue with identical results directories.

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
docs/                architecture diagrams mapped to Anthropic's agent guidance
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
