# ctf-agent
## AI Use Disclaimer:
I used a custom pi agent running glm-5.3-flash to help me design and create the harness for this agent. There were a lot of changes and issues I found along the way, most notably the agent wanting to feed the solver the flags so we could get more solves. To combat this I made the agent require a checkout through the submit-flag tool which checked the flag using the NYU CTF Utils.

A tool-using LLM agent harness for the NYU CTF Challenge (CS 590.007, Duke). It
solves captured-flag challenges from the
[CTFTiny benchmark](https://arxiv.org/abs/2508.05674) by calling an LLM
(LiteLLM — any OpenAI-compatible provider; the Duke AI Gateway and Ollama Cloud
are both used in our runs), running model-proposed shell and Python analysis
in a sandboxed workspace, and checking candidate flags against the benchmark’s
hidden answers.

See [`REPORT.md`](REPORT.md) for the architecture, experiments, full results
table, contributions, and citations.

## How it works

`ctf_agent.runner` loads benchmark targets and verifies submitted flags. The
answers are never included in the agent’s context. `ctf_agent.agent` manages the
chat loop and provides a tool belt: `list_files`, `read_file`, `run_command`,
`run_python`, `write_file`, and `submit_flag`. The loop includes a repeat-call
guard, automatic context elision near the token budget, nudges for prose-only
replies, and budget-aware coaching near the step limit.

All tool execution goes through `tools/sandbox.py`, which confines access to the
workspace, uses an empty environment, enforces hard timeouts, and truncates
output. Model and tool calls are recorded in a local `trace.jsonl` and in
LangSmith. Each trace starts with a `meta` record containing the challenge,
`run_id`, trial, and model, and ends with a `done` record containing the status,
verification result, and step count. The LangSmith root run receives the same
metadata and `[challenge, attempt]` tags for filtering.

`ctf_agent.metrics` and `ctf_agent.aggregate` compute pass@k, token, cost, and
status summaries from `attempts.jsonl`. `ctf_agent.flask_gui` provides a web UI
for challenge selection, live runtime updates, and reports. `ctf_agent.batch`
runs the scored set headlessly.

## Setup

```bash
uv sync --all-groups          # Core and development dependencies; add --extra bench for nyuctf
cp .env.example .env          # Fill in tokens; never commit this file
sudo apt install bubblewrap binutils unzip file bzip2 xz-utils xxd
```

The benchmark data lives in `./benchmark/` and is git-ignored; clone or download
it separately.

### Curated environment

Tool scripts run with the plain `python3` resolved by the sandbox, using `-E
-s` and the workspace as the working directory. This allows handout sibling
modules to be imported; several CTFTiny handouts include importable `.py` files.

The default `ctf-agent:latest` image includes a CTF toolkit: binary-analysis
utilities (`objdump`, `strings`, `nm`, `readelf`, `hexdump`, `xxd`, `file`,
`unzip`, `bzip2`, and `xz`), plus `gdb`, `binwalk`, `exiftool`, `steghide`,
`outguess`, `john`, `openssl`, `socat`/`nc`, and `poppler-utils`. It also
includes a pinned Python capability layer: NumPy, CPU-only PyTorch,
PyCryptodome, SymPy, Z3, gmpy2, and pwntools. The system prompt lists these
tools so the agent does not need to probe for them. The goal is to evaluate
reasoning rather than work around missing dependencies—a confound that caused
one real run, `msc-ezmaze`, to spend its budget decoding PyTorch’s pickle format
by hand.

The capability-layer pins are in `ctf-tools.txt`. They are installed
system-wide in the Docker image, outside the uv project environment. Base
images are digest-pinned and `uv.lock` is frozen. Capability-layer packages are
version-pinned, but not hash-pinned, and are updated deliberately. The image is
about 2.65 GB and takes roughly 5–10 minutes to build; this is accepted in
exchange for the included solving tools.

The deterministic pre-flight in `tools/provision.py` remains enabled by default.
It derives requirements from the staged handout files—the same files the agent
can see—using a small whitelist: `.pt` files trigger PyTorch, and handout
imports of `Crypto`, `numpy`, `sympy`, `z3`, `gmpy2`, or `pwn` trigger their
corresponding packages. With the bundled toolkit, the pre-flight usually reports
`present` after one inexpensive jailed probe. On hosts or older images without a
package, it can still provision it.

The probe runs through the jail, using the same interpreter and environment as
`run_python`. Installation happens at the harness level, which has network
access; the jailed agent does not. Workspace text cannot request arbitrary
packages: only whitelisted signatures trigger installation. Provisioning
failures are recorded and are not fatal. Statuses (`present`, `installed`, or
`unavailable <reason>`) appear in the trace and GUI. Set `CTF_PROVISION=off` to
disable pre-flight provisioning for ablations.

### Configuration

Required `.env` values:

- `LITELLM_BASE_URL`
- `LITELLM_TOKEN`
- `CTF_MODEL`
- `CTF_DATASET_JSON` — path to the CTFTiny index

LangSmith keys are optional. Any OpenAI-compatible provider can be used, such
as the Duke AI Gateway (with Duke VPN) or Ollama Cloud at `https://ollama.com`.

## Usage

Run one scored challenge. Results are written to
`results/<dir>/attempts.jsonl`:

```bash
uv run ctf-agent <challenge_id>
```

For a development run that is not reported:

```bash
uv run ctf-agent rev-checker --dev
```

Run the scored set in batch mode:

```bash
uv run python -m ctf_agent.batch
```

Start the web UI at <http://127.0.0.1:5000>:

```bash
uv run ctf-flask-gui
```

The GUI’s Config tab lets you change the model ID, API key, gateway URL, and run
limits for the current session. The model ID is selected from the gateway
catalog when available, or entered manually. Settings can optionally be
persisted to `.env`. The tab also includes a connection test that does not spend
tokens.

Run limits include retries per challenge, the flags-in-run cap
(`CTF_MAX_SUBMITS`, default `3`), and an on/off switch for retry handoff notes.

### Evaluator loop

The agent uses two deterministic evaluator mechanisms. They require no
additional model calls and consume no tokens:

- **In-run feedback:** A rejected flag returns `INCORRECT (k of N)` and echoes
  the rejected candidate so the model can correct course within the run.
  Submissions are capped by `CTF_MAX_SUBMITS` (default `3`). Exceeding the cap
  ends the run as `no_flag` rather than allowing unlimited guesses.
- **Budget-aware escalation:** Escalations are based on the binding budget—the
  step or token limit that will run out first. A consolidation checkpoint fires
  once at 50% of that budget. An urgent warning fires at 85% of the token cap
  and tells the agent to submit its best candidate.

For GUI retry chains only, a retry after a model failure can receive a
deterministic handoff note distilled from the failed attempt’s trace. The note
summarizes the outcome, step and tool mix, repeated or errored tool calls, every
rejected flag (“do not resubmit”), and where the attempt stopped. It never
contains the flag and costs no tokens to produce. Disable it with
`CTF_FEEDBACK_NOTE=off`.

Batch and other scored runs deliberately do not use handoff notes, so retries
remain clean-slate and pass@k measurements stay honest. If notes are enabled for
scored runs, label those results as a separate ablation.

Each trace’s first `meta` line records the exact limits used. If a run card shows
`100 steps · 300000 tokens`, those were the limits for that run. Check the Config
tab or `.env` before comparing runs.

Aggregate metrics from run artifacts:

```bash
uv run python -m ctf_agent.aggregate results/scored/attempts.jsonl
```

## Fixed limits for reported runs

Reported attempts use these limits:

- 30 steps
- 15 minutes
- 150,000 total tokens
- 4,000 characters of tool output
- 60-second tool timeout
- 1,024 maximum output tokens

The first scored batch used `gpt-5.4-mini` through the Duke AI Gateway. Later
bulk sessions used `gpt-oss:120b` through Ollama Cloud after the gateway quota
was exhausted. Each attempt records its model, and each batch snapshot captures
the configuration it used in `configs/scored_set.json` and the adjacent
`config.json`.

## Reproducing the experiment

The full protocol is **pilot → fixed-caps sweep → consolidated report**:

1. **Pilot (one challenge, one trial)** — shake out infrastructure before
   spending budget on a sweep. Requires a reachable model endpoint
   (`.env`: `LITELLM_BASE_URL` + `LITELLM_TOKEN` + `CTF_MODEL`) and
   `benchmark/ctftiny.json`.

   ```bash
   uv run python -m ctf_agent.batch configs/scored_set.json --trials 1 \
       --out results/pilot --only rev-baby-mult
   ```

2. **Fixed-caps sweep (fresh trials)** — one config file pins every cap
   (`configs/scored_set.json`: 30 steps · 15 min · 150k tokens · 4k tool
   output · 60s tool timeout · ≤3 submissions, token-matched `gpt-oss:120b`).
   Every run gets a fresh staged workspace and `results/<out>/config.json`
   stamps benchmark commit + agent commit + limits.

   ```bash
   uv run python -m ctf_agent.batch configs/scored_set.json --trials 3 \
       --out results/sweep
   ```

3. **Consolidated report** — per-attempt buckets from trace `meta` (caps,
   triage flag, dev flag), pass@k, tokens, cost, and error tax:

   ```bash
   uv run python scripts/build_results_report.py
   ```

Writes `results/RESULTS.md` + `results/full_summary.json` over all known
ledgers (including `results/ablation-v18/`). Dev runs and `rev-checker` are
excluded automatically by the `dev` meta flag.

## Containers

The project ships as a reproducible container: rebuild it anywhere and get the
same software (base images pinned by digest; `uv sync --frozen` installs exactly
the versions in `uv.lock`).

### From scratch (fresh clone → GUI)

```bash
git clone https://github.com/danithen/ctf-agent.git ctf-agent && cd ctf-agent
cp .env.example .env            # then fill in LITELLM_BASE_URL / LITELLM_TOKEN / CTF_MODEL
git clone https://github.com/NYU-LLM-CTF/CTFTiny.git benchmark
git -C benchmark checkout f1c9531   # pin the exact commit the scored config snapshot uses
docker compose up --build       # first build ~2.65 GB, ~5–10 min; GUI on http://localhost:5000
```

No host-side `uv sync` or bubblewrap install is needed for the container route —
those are host-run prerequisites (see Setup). Port 5000 busy on the host? Start with
`CTF_GUI_HOST_PORT=5001 docker compose up -d` instead. Optional gate before opening
the GUI:

```bash
docker build --target test -t ctf-agent:test .
docker run --rm --security-opt seccomp=unconfined ctf-agent:test   # 63 passed, 3 skipped
```

### Manual container workflow

```bash
docker build --target test -t ctf-agent:test .   # build + run the pytest gate (no network, no tokens)
docker build -t ctf-agent:latest .               # runtime image (GUI + CLI + server)
docker compose up --build                        # GUI on http://localhost:5000
docker compose down
```

The image excludes machine-local data: `.env` (secrets), `benchmark/` (course
dataset), and `runs/` and `results/` (your data). Docker Compose mounts them at
runtime:

| Host | Container | Mount |
|---|---|---|
| `./.env` | `/app/.env` | Read-only |
| `./benchmark` | `/data/benchmark` | Read-only |
| `./runs` | `/data/runs` | Shared |
| `./results` | `/data/results` | Shared |

Run the CLI in the container with secrets from `.env` and container paths
explicitly set with `-e`, which overrides `--env-file`:

```bash
docker run --rm --security-opt seccomp=unconfined --env-file .env \
  -e CTF_RUNS_DIR=/data/runs -e CTF_TUI_RESULTS=/data/results \
  -e CTF_DATASET_JSON=/data/benchmark/ctftiny.json -e CTF_BENCHMARK_DIR=/data/benchmark \
  -v "$PWD/benchmark:/data/benchmark:ro" -v "$PWD/runs:/data/runs" -v "$PWD/results:/data/results" \
  ctf-agent:latest ctf-agent rev-checker --dev
```

Docker’s default seccomp profile blocks the user-namespace creation that
bubblewrap needs, resulting in `bwrap: No permissions to create new namespace`.
The command therefore uses `seccomp=unconfined` to remove the outer filter.
The agent’s jail is still provided by bubblewrap: it has no network, isolated
PID and mount namespaces, and access only to the workspace and read-only system
directories. The container retains cgroups and capability separation. The
sandbox tests run under this configuration in the `--target test` gate.

To move the image to an offline machine:

```bash
docker save ctf-agent:latest | gzip > ctf-agent.tar.gz
# On the target machine:
docker load < ctf-agent.tar.gz
```

Alternatively, tag and push the image to a registry. In either case, bring the
same `.env` and `benchmark/` data separately; neither is included in the image.
Runs continue in the same results directories.

## Testing

```bash
uv run python -m pytest tests/
uv run python -m ruff check .
```

The pytest suite covers sandbox and workspace behavior, a mocked gateway, agent
loop scaffolding (nudges, context elision, and budgets), and the Flask GUI.
Model-call tests use scripted fakes and do not spend gateway tokens.

## Repository layout

```text
src/ctf_agent/       Runner, agent loop, LLM gateway client, batch, metrics,
                     aggregation, CLI, Flask GUI and templates, legacy TUI
configs/             Scored set (challenges, trials, and limits)
docs/                Architecture diagrams mapped to Anthropic’s agent guidance
tests/               Pytest suites with fakes and no live model calls
results/             Experiment attempts and notes (local artifacts)
runs/                Per-run artifacts (git-ignored)
```

## Credits and licenses

- Benchmark data and challenge loader: [CTFTiny / NYU CTF Bench](https://github.com/NYU-LLM-CTF/NYU-CTF-Bench), licensed under GPL-2.0 and associated with the AAAI ’26 paper, accessed through the `nyuctf` package.
- LLM inference: via `langchain-openai` through any OpenAI-compatible gateway — the Duke AI Gateway and Ollama Cloud in our runs; tracing via LangSmith.
- Flask, Textual, and LangChain are used as unmodified dependencies.
- Agent logic, sandboxing, metrics, and reporting are original to this repository. See [`REPORT.md`](REPORT.md) §6 for the course-required disclosure of reused code and AI assistance.
