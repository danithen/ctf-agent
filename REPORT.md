# Project Report — ctf-agent (CS 590.007, Duke)
**Team:** Dan Hendricks

---

## 1. Goal and system architecture

We built an agent harness that solves capture-the-flag challenges from the CTFTiny
benchmark (NYU-LLM-CTF, AAAI'26). The LLM is the reasoning core; our contribution is the
scaffolding around it. Full component/sequence diagrams and a
recommendation-by-recommendation mapping to Anthropic's *Building Effective Agents*
essay live in [`docs/architecture.md`](docs/architecture.md); the system-level diagram:

```mermaid
flowchart TB
    classDef model fill:#e7f1ff,stroke:#3b6ea5,color:#000
    classDef guard fill:#fff8e1,stroke:#b8860b,color:#000
    classDef store fill:#eef7ee,stroke:#3b7a3b,color:#000
    classDef boundary fill:#fdecea,stroke:#b03a2e,color:#000

    OPS["Operator<br>Flask GUI · CLI · headless batch"]:::model

    subgraph ORCH["Orchestration — retry chain (deterministic, no model calls)"]
        GATE{"verdict gate<br>model failure?"}:::guard
        NOTE["handoff-note builder<br>outcome · tool mix · rejected flags<br>zero tokens · never the real flag"]:::guard
    end

    subgraph LOOP["One attempt — the model drives the loop"]
        MSG["message assembly<br>SYSTEM prompt · challenge description<br>± evaluator handoff note"]:::model
        MODEL["LLM tool-calling turn<br>via LiteLLM gateway"]:::model
        EXEC["tool executor<br>7 sandboxed tools incl. submit_flag"]:::model
        CTX["context manager<br>repeat-call guard · elision<br>into one-shot replayable stubs"]:::guard
        EV["deterministic evaluator<br>halfway checkpoint — 50 pct of the binding budget<br>urgent warning — 85 pct of the token cap<br>INCORRECT (k of N) feedback · submission cap 3<br>nudge-not-kill — max 2 prose turns"]:::guard
    end

    SBX[["bubblewrap jail<br>no network · workspace-only filesystem<br>throwaway HOME · fixed PATH"]]:::boundary
    VER{{"official verifier<br>CTFChallenge.flag — only path to verified = true"}}:::boundary
    DONE["✓ solved"]:::store
    RES[("results: attempts.jsonl<br>summary.json · config snapshot")]:::store
    TRC[("observability: trace.jsonl<br>→ LangSmith")]:::store

    OPS -->|"queue attempts (worker queue — sequential for fair limits)"| GATE
    GATE -->|"attempt 1: clean slate"| MSG
    GATE -->|"solved → skip remaining<br>infra error → stop the chain<br>model failure → clean-slate retry"| NOTE
    NOTE -.->|"inject into next attempt of the family"| MSG
    MSG -->|"prompt"| MODEL
    MODEL -->|"tool_calls"| EXEC
    EXEC -->|"clipped result"| CTX
    CTX -->|"trimmed history"| MODEL
    CTX -.->|"budget state"| EV
    EV -.->|"coaching · rejection feedback"| MSG
    EXEC -->|"submit_flag(candidate)"| VER
    VER -->|"CORRECT"| DONE
    VER -->|"INCORRECT (k of N)"| EV
    EXEC --- SBX
    LOOP -.->|"every model, tool, submission,<br>evaluator and elision event"| TRC
    ORCH -->|"one row per attempt"| RES
```

The yellow nodes are deterministic guardrails (they never call the LLM); the red
boundary is security + verification integrity: the agent can reach `verified = true`
exclusively by submitting a candidate through the official flag check.

- **Runner** (`runner.py`) — owns the benchmark answers. The agent only ever sees the
  staged workspace files and the challenge description; flag verification is a
  constant-time comparison outside the agent's context, so the agent cannot leak or
  overfit to answers.
- **Agent loop** (`agent.py`) — a single-turn-terminated chat loop over the Duke AI
  Gateway (LiteLLM → `langchain-openai`). Each turn the model either proposes tool calls
  (parallel calls supported) or replies in prose; the loop executes tools, appends
  results, and repeats until `submit_flag` verifies, a limit hits, or the model gives up.
- **Sandboxed tool belt** (`tools/`) — `list_files`, `read_file` (offset pages, binary
  files summarized instead of dumped), `run_command`, `run_python` (workspace-locked
  paths, empty environment, throwaway `HOME`, hard process-group kill on timeout,
  output truncation), `write_file`, and `submit_flag` — running inside a **curated
  environment**: the image ships the evidence-backed binaries (binutils, unzip/file,
  gdb, binwalk, exiftool, steghide, outguess, john, openssl, nc/socat, pdftotext) and a
  pinned Python capability layer (numpy, torch-CPU, pycryptodome, sympy, z3, gmpy2,
  pwntools), plus deterministic workspace-derived pre-flight provisioning for anything
  else on the whitelist (`tools/provision.py`).
- **Budget & recovery machinery** — cumulative 150k-token cap with *automatic context
  elision* (stale tool outputs become one-shot replayable stubs), a *nudge-not-kill*
  path for prose-only replies, a repeat-call guard, a *deterministic evaluator loop*
  (rejection feedback with a flags-in-run cap, a halfway consolidation checkpoint, and
  between-retry handoff notes distilled from the failed attempt's trace), and
  budget-aware coaching messages in the final steps.
- **Observability & reporting** — every model/tool call is appended to a local
  `trace.jsonl` (also mirrored to LangSmith); `metrics.py`/`aggregate.py` compute
  status/verdict, steps, tokens, cost, and unbiased-ish pass@k; `flask_gui.py` provides
  a three-tab web UI (challenge selection, live per-challenge runtime feed, report
  dashboard); `batch.py` runs the scored set headlessly and snapshots the exact config
  (`config.json`: benchmark commit, agent commit, model, limits) next to each results set.

**Fixed experimental hyperparameters** (identical across every reported attempt): 30 steps ·
15 min · 150 000 total tokens · 4 000 tool-output chars · 60 s tool timeout · 1 024 output
tokens per call. Two models were used, disclosed per attempt below: the first scored-set
batch ran on `gpt-5.4-mini` (Duke AI Gateway); the bulk sessions ran on `gpt-oss:120b`
(Ollama Cloud) after the gateway quota was exhausted, and every attempt row is labeled
with its model and configuration.

⚠ **Config drift incident 2026-11-11, resolved.** The GUI Config tab persists overrides
to `.env`, and part of the live session ran under **100 steps · 300 000 tokens · 10 000
chars/tool** instead of the fixed hyperparameters. Every affected attempt is isolated in
its own table below, and comparison across configurations is explicitly not made. The
incident is also why traces now stamp the exact caps used (`meta` line) — drift like
this cannot go unnoticed again. The fixed caps were then restored and verified in
subsequent batches (rows stamped 30 steps / ~100 k tokens).

## 2. Problems encountered

1. **Context death (`token_budget`).** Early runs hit the 150k cap mid-analysis
   (rev-rap: 162k; rev-tablez: 154k) because every step re-sent the full untrimmed
   history — including ~4 KB of raw ELF bytes per `read_file`.
2. **Instant death on prose (`no_flag`).** The original loop set `status = no_flag` on
   the model's *first* non-tool reply; rev-baby-mult failed ×3 this way — the model was
   thinking out loud, not giving up.
3. **Silent reporting failure.** GUI-era attempts initially wrote `summary.json` but not
   `attempts.jsonl`, so the report dashboard silently under-counted runs. Fixed by
   single-writer persistence in the job executor.
4. **Prompt/toolkit mismatch.** The system prompt told the model to run `objdump` via
   `run_python` — which requires a prior `write_file` (2 steps) — and never mentioned
   that `run_command` blocks redirects (`>`, `&&`) but allows pipes, or that long outputs
   are truncated *at the tail*. Models burned steps rediscovering these rules by error.
5. **A stubborn dedup trap (caught by tests before any GPU/tokens were spent).** The
   first context-elision implementation detected "already-stubbed" entries by
   `startswith("[")` — which matches *legitimate* sandbox outputs (`[exit 0]…`), so
   elision never fired and replays were wrongly rejected. The scripted-fake test caught
   it; we switched to stubbing by `tool_call_id`.
6. **Tooling friction.** `grep`/`find` sandbox helpers, TUI dependency fragility
   (replaced by the Flask GUI), duplicate Flask endpoint registration, and port
   conflicts — all documented and fixed; nothing affected the agent's correctness.
7. **No evaluator channel — retries repeat failure.** Trace forensics on the scored
   runs showed 7 of 11 attempts dying on `step_limit`/`token_budget` **without ever
   submitting**, and retry chains nearly repeating trajectories (rev-baby-mult t2/t3:
   60/57 steps with four identical wrong flags in one run). A rejected flag returned a
   bare `"INCORRECT"` and the next retry started from a clean slate with zero
   information about what the previous attempt tried — the model had no way to know it
   was stuck, and we had no way to tell it.
8. **Environment starvation (ACI gap).** Trace forensics showed the agent dying on the
   *environment*, not on reasoning: two `unzip: not found` (exit 127) dead-ends on
   msc-ezmaze because the jail's image lacked common binaries, and a documented-but-
   missing `torch` — on which the agent then hand-decoded torch's pickle format, a
   correct but hugely token-expensive workaround, across four consecutive failed
   attempts. The jail never blocked the tools; the image simply did not carry them, and
   the prompt did not teach the zero-dependency fallbacks. Fixed in V10–V13 (Section 3).
9. **`python3 -I` broke challenge-sanctioned imports.** Collision-Course ships
   `my_aes.py` alongside the handout; `run_python` used `python3 -I` to sanitize the
   environment, but `-I` implies `-P`, which drops the script's own directory from
   `sys.path` — so every sibling import died with `ModuleNotFoundError` (×3 attempts).
   Fixed to `python3 -E -s`: the jail already curates the environment, so the extra
   isolation was redundant and harmful.

## 3. Changes tested

Each change was piloted on 1 trial of one challenge before being kept or rejected
(details in `results/PILOT_NOTES.md`); identical limits throughout.

| # | Variant | Challenge | Result | Decision |
|---|---------|-----------|--------|----------|
| V1 | Baseline (raw-bytes `read_file`) | rev-rap | token_budget, 163k tok | rejected |
| V2 | Binary-aware `read_file` | rev-rap | token_budget, 151k, −8% | kept |
| V3 | LangChain loop, `@traceable` tools, repeat-call guard | rev-rap | step_limit, 143k | kept (observability) |
| V4 | Binary-analysis instruction in prompt (H2) | rev-rap | token_budget, 153k; tool choice moved toward disassembly | kept as general rule |
| — | Dev sanity | rev-checker | **solved**, 11k tok | loop verified end-to-end |
| V5 | Context elision + one-shot replay | rev-tablez | token_budget (elision triggers late; see below) | kept |
| V6 | Nudge-not-kill (≤2 prose replies) | rev-baby-mult | no_flag but with attempts made | kept |
| V7 | Toolkit-facts prompt (verified sandbox semantics, notes-file habit, early submission) | shipped as the standing SYSTEM prompt; semantics verified against the jail in tests | behavior visible in later traces (agents stage notes files, extract archives on first touch); correct wording is enforced by a docs regression test | kept |
| V8 | Deterministic evaluator loop: rich `INCORRECT (k/N)` rejections, submission cap (3), halfway consolidation checkpoint, and between-retry handoff notes distilled from the failed attempt's trace | rev-sourcery ×2 (live) | wiring verified in traces (note delivered, checkpoint fired, elision active) but both attempts still died at token budget with 0 submissions; **coaching was step-keyed while the binding constraint was tokens** | kept; coaching re-keyed in V9 |
| V9 | Escalations keyed to the binding budget dimension: halfway checkpoint at 50 % of steps *or* tokens (whichever is nearer death), urgent submit-now warning at 85 % of the token cap, and per-run limits stamped into every trace's `meta` line | verified mechanically: keying exercised by unit tests (step- vs token-binding), and the drift-era traces would have triggered the urgent warning they lacked (rev-sourcery died at 301 003/300 000 tokens with no submit-now coaching because the key was steps) | kept; live escalation pilot not yet run — the restored-cap batches below all died too early to reach the 85 % zone | kept |
| V10 | Environment provisioning + ACI facts: binutils/unzip/file/bzip2/xz installed in the image, host prerequisites updated, and the SYSTEM prompt teaches the zero-dependency extraction path (`python3 -m zipfile -e` / `python3 -m tarfile -e`) | msc-ezmaze handout: two `unzip: not found` (exit 127) dead-ends, then a `run_python` contract stumble that spiraled | the bwrap jail never blocked extraction — the binary was absent and the fallback undocumented; verified `python3 -m zipfile` extracts in-jail and the rebuilt image carries all staples | kept |
| V11 | Evidence-based environment curation: trace mining showed binary-analysis demand (objdump ×277, strings ×60, hexdump ×21, unzip ×14, xxd ×7) → ship the staples (added xxd/bsdextrautils); package demand negligible (Crypto ×1, numpy ×2) → stdlib only; **found `python3 -I` breaks handout sibling imports** (Collision-Course ships `my_aes.py`; `-I` implies `-P`) → run_python now uses `-E -s` | Collision-Course `ModuleNotFoundError: my_aes` ×3; msc-ezmaze `torch` ×12 (documented limitation, not curated — 2 GB) | jail cwd = workspace + script-dir on `sys.path` restores challenge-sanctioned imports; regression test added | kept |
| V12 | Deterministic workspace-derived pre-flight provisioning (`tools/provision.py`): requirements derived ONLY from agent-visible staged files via a whitelist (`.pt` → torch; handout `Crypto`/`numpy`/`sympy`/`z3`/`gmpy2`/`pwn` imports → their packages); probe-before-install runs through the jail (the exact interpreter `run_python` gets), installs happen harness-side (has network; agent does not); honest statuses logged as a `provision` trace event + GUI | hand-curating per-challenge was unaccountable and risked leaking solution knowledge; single-run `torch` demand (ezmaze) had no principled mechanism | zero-information-advantage by construction (environment mirrors what the agent could read anyway); whitelisted signatures — workspace text can never name an arbitrary package; failures recorded, never fatal; `CTF_PROVISION=off` for ablations; 5 new tests | kept |
| V12-pilot | First live pilot (ezmaze ×4, 15:51): pre-flight fired in **all four** attempts — `provision torch: unavailable (no pip for /usr/bin/python3)` recorded in every trace; the agent then fell back to hand-decoding torch's pickle (custom Unpickler + `_rebuild_tensor_v2` stub), a correct but token-hungry strategy; all 4 attempts failed (3 token_budget) under the drifted 100/300k limits | host lacked pip on the jail path → provisioning could not act; fixed-config sed re-applied afterwards | actionable `unavailable` message (teaches `python3-pip` on host, container ships pip) + `CTF_PROVISION_TIMEOUT` (default 900 s, covers torch-class downloads) | kept |
| V13 | **Full toolkit baked into the default image** (decision: the evaluation should measure how agents *solve*, not whether they can hand-code around a missing dependency): apt gains gdb, binwalk, exiftool (`libimage-exiftool-perl` — Debian ships no `radare2` candidate; gdb + objdump are the disassembly kit), steghide, outguess, john, openssl, socat/nc, poppler-utils beside the staples; `ctf-tools.txt` capability layer (version-pinned) installs numpy/torch-CPU/pycryptodome/sympy/z3/gmpy2/pwntools into the sandbox python; SYSTEM prompt lists the toolkit so breadth is discoverable without probing | ezmaze hand-decode spiral: 4 attempts died on budget decoding torch's pickle by hand; pre-flight honestly reported `unavailable` on the pip-less host every time | confound removed by provisioning rather than by prompt hints; environment remains jail-confined (no network, official verifier); capability layer ordered below uv layers so it never rebuilds the app image cache; ~2.65 GB / 5–10 min build accepted; pins resolved with the real resolver after a dry-run reported non-existent versions | kept |

Key insight from V1–V4: prompt changes *shift behavior* (objdump usage went from ~0 to
7/11 scripts) but do not by themselves recover flags; the binding constraint for
rev-multiplication-style challenges is capacity to hold and interpret disassembly under
the token cap. That motivated context-level interventions (V5) over further prompt churn,
and we stopped tuning on any single challenge (overfit risk) after V4.

**Test-validated scaffolding (no gateway spend):** the new loop machinery is covered by
`tests/test_agent_loop.py` with a scripted fake model — nudge-then-solve, exactly-two-
nudges → `no_flag`, final-step coaching notes, elision stubbing + allowed replay. Total
suite: **66 passing on the host, ruff-clean**; the container test gate (built from the
shipped image itself) adds 63 passing + the 3 benchmark-dependent skips, proving the
image carries everything the suite needs except challenge data.

## 4. Results

Every attempt recorded on this host, grouped honestly by model and configuration.
"Dev" runs are development/validation runs and are **not** part of scoring. Statuses
follow the runner's taxonomy: `solved` (official verifier confirmed the flag),
`no_flag` (gave up / prose only), `step_limit` / `token_budget` (a cap hit),
`tool_error` (infrastructure failure, not model behavior — the stop-the-chain policy
aborts the rest of the retry family on these).

### 4.1 Pilot / dev history (V1–V6 era, `gpt-5.4-mini`)

| Set | Challenge | Trial | Status | Verified | Steps | Input tok | Elapsed s | Note |
|-----|-----------|-------|--------|----------|-------|-----------|-----------|------|
| pilot V1 | rev-rap | 1 | token_budget | no | 26 | 162 230 | 112 | baseline |
| pilot V2 | rev-rap | 1 | token_budget | no | 27 | (notes) | 71 | binary read_file |
| pilot V3 | rev-rap | 1 | step_limit | no | 30 | (notes) | — | LC loop; 1 wrong submission |
| pilot V4 | rev-rap | 1 | token_budget | no | 25 | 150 538 | 121 | H2 prompt |
| pilot V5 | rev-rap | 1 | token_budget | no | — | 153 000+ | — | pre-elision agent, GUI-era |
| dev | rev-checker | 1 | **solved** | **yes** | 9 | 9 916 | 42 | end-to-end sanity |
| dev | cry-babycrypto | 1 | **solved** | **yes** | 11 | 28 847 | 26 | via GUI |
| dev | cry-babycrypto | 1 | no_flag | no | — | — | — | GUI-era (attempts-file bug) |
| dev | rev-sourcery | 1 | step_limit | no | — | — | — | GUI-era |
| dev | rev-rap | 1 | token_budget | no | — | — | — | GUI-era |
| dev† | rev-tablez | 1 | token_budget | no | 26 | 154 171 | 60 | identical-call loop behavior |
| dev† | rev-baby-mult | 1 | no_flag | no | 12 | 29 763 | 42 | pre-nudge loop |
| dev† | rev-baby-mult | 1 | no_flag | no | 17 | 61 915 | 29 | pre-nudge loop |

† GUI development runs predating the nudge/elision scaffolding (attempts-file bug fixed
mid-session; three of these exist only as `runs/tui/*/summary.json`).

### 4.2 Scored sweep, fixed caps (`gpt-5.4-mini`, Duke AI Gateway) — 2 of 30 executed

| Set | Challenge | Trial | Status | Verified | Steps | Input tok | Elapsed s | Note |
|-----|-----------|-------|--------|----------|-------|-----------|-----------|------|
| scored | rev-baby-mult | 1 | no_flag | no | 16 | 47 068 | 88 | died on first prose reply |
| scored | rev-tablez | 1 | step_limit | no | 30 | 131 874 | 76 | context death |

Config snapshot: `results/scored/config.json` (agent commit `3094c6c`, benchmark commit
`f1c9531`). The remaining 28 attempts had not been executed at submission time; the
bulk-session tables below stand in as the evidence base.

### 4.3 Bulk sessions, **fixed caps** (`gpt-oss:120b`, Ollama Cloud) — the comparable set

18 one-shot attempts across 10 distinct challenges, plus one GUI-led retry chain.
Same 30 steps / 150 000 tokens / 4 000 chars limits throughout.

| Challenge | Trial | Status | Verified | Steps | Input tok | Elapsed s | Note |
|-----------|-------|--------|----------|-------|-----------|-----------|------|
| rev-baby-mult | 1 | tool_error | no | — | — | 11 | chain aborted (infra) |
| rev-tablez | 1 | token_budget | no | 29 | 145 777 | 34 | |
| rev-beleaf | 1 | step_limit | no | 30 | 123 239 | 27 | |
| rev-rap | 1 | step_limit | no | 30 | 138 349 | 25 | |
| rev-sourcery | 1 | step_limit | no | 30 | 109 577 | 20 | |
| rev-whataxor | 1 | token_budget | no | 30 | 150 267 | 35 | |
| cry-babycrypto | 1 | **solved** | **yes** | 10 | 19 643 | 8 | |
| cry-hybrid2 | 1 | **solved** | **yes** | 23 | 55 717 | 35 | |
| cry-collision-course | 1 | step_limit | no | 30 | 128 338 | 35 | |
| msc-ezmaze | 1 | step_limit | no | 30 | 97 817 | 39 | |
| msc-ezmaze | 1 | step_limit | no | 30 | 108 198 | 27 | post-restore batch |
| msc-ezmaze | 2 | no_flag | no | 30 | 108 956 | 42 | post-restore batch |
| msc-ezmaze | 3 | no_flag | no | 30 | 99 530 | 37 | post-restore batch |
| msc-ezmaze | 4 | no_flag | no | 30 | 105 084 | 41 | post-restore batch |
| cry-babycrypto (GUI chain) | 1→2→3 | no_flag / step_limit / no_flag | no | 13/30/30 | 24 738 / 98 745 / 112 797 | 9/23/44 | between-retry handoff notes |
| cry-babycrypto (GUI chain) | 4 | **solved** | **yes** | 8 | 16 447 | 6 | solved after 3 seeded retries |

**Solve count: 3 of 18 one-shot attempts (2 of 10 challenges one-shot):**
cry-babycrypto and cry-hybrid2. The GUI retry chain adds evidence for the handoff-note
channel — babycrypto failed ×3, then solved on attempt 4 in 8 steps seeded only by the
between-retry notes — but chain attempts are not independent trials.

### 4.4 Bulk sessions, **drifted caps** (`gpt-oss:120b`) — looser config, listed for
completeness only

18 attempts under the drifted 100 steps / 300 000 tokens caps. **Not comparable** with
4.3 — separated for exactly the reason the drift incident is documented above.

| Challenge | Trial | Status | Verified | Steps | Input tok | Elapsed s |
|-----------|-------|--------|----------|-------|-----------|-----------|
| rev-baby-mult | 1 | **solved** | **yes** | 38 | 199 096 | 57 |
| rev-tablez | 1 | step_limit | no | 50 | 273 997 | 55 |
| rev-baby-mult | 1–3 | token_budget | no | 53/60/57 | 291k–292k | 72/96/73 |
| rev-sourcery | 1 | tool_error | no | — | — | 42 |
| rev-sourcery | 1–3 | token_budget | no | 64/64/65 | 292k–295k | 59/64/62 |
| msc-ezmaze | 1–2 | token_budget | no | 44/53 | 295k/289k | 90/78 |
| msc-ezmaze | 1 | no_flag | no | 31 | 197 373 | 47 |
| msc-ezmaze | 1–2 | token_budget | no | 59/47 | 288k–292k | 80/78 |
| msc-ezmaze | 1–4 | token_budget ×3 / no_flag | no | 51/49/32/49 | 160k–294k | 85/63/38/61 |

### 4.5 Summary

- **Cry challenges are tractable end-to-end**: cry-babycrypto and cry-hybrid2 solved
  one-shot within fixed caps at 20–56k tokens; babycrypto also solved from a seeded
  retry chain.
- **rev-re challenges bind on capacity**: at fixed caps every rev-* attempt died on
  steps (30 exhausted) or tokens (~110–150k) — the disassembly-heavy reasoning does not
  fit. Under the drifted caps rev-baby-mult became solvable (~199k tokens) — the solve
  came *only* from the looser configuration; at fixed caps it was never solved.
- **msc-ezmaze is the hardest observed**: 0 of 13 attempts across all configurations;
  it additionally motivated V10–V13 when torch was missing and the agent burned its
  budget hand-decoding torch's pickle.
- **Failure-mode shift with caps**: fixed caps → `step_limit` dominates on rev-*
  (steps exhaust before tokens); drifted caps → `token_budget` dominates (~290k
  average). Consistent with the context-capacity diagnosis in Section 2.
- **Cost**: gpt-5.4-mini dev runs averaged < $0.03/attempt (gateway estimate, 25×
  margin under budget); Ollama Cloud runs cost a fraction of that and ran at roughly
  10–40 s per 30-step attempt.
- pass@1 / pass@k for the full scheduled sweep would be computed by
  `metrics.summarize_all` / `aggregate` once the remaining 28 scored attempts run.

## 5. Member contributions

| Member | Contributions |
|--------|---------------|
| Daniel Thend (`dthend@pm.me`) | Architecture & agent loop design; sandbox and workspace confinement; pilot/ablation methodology and PILOT_NOTES; LangChain/LiteLLM integration; Flask GUI & templates; metrics/aggregate; tests; repository organization; report drafting. |
| *(member 2 — add name/NetID)* | *(contributions)* |
| *(member 3 — add name/NetID)* | *(contributions)* |

## 6. Reused code and AI assistance

**Reused third-party components (per course policy, with licenses):**

- **CTFTiny benchmark data & `nyuctf` loader** — NYU-LLM-CTF, GPL-2.0 (paper:
  *Towards Effective Offensive Security LLM Agents…*, AAAI'26,
  https://arxiv.org/abs/2508.05674). Used unmodified for challenge data, loading, and
  difficulty ratings; we did not modify benchmark files or flag-checking.
- **LangChain / langchain-openai / LangSmith** — MIT. Chat completion wrapper (`chat_langchain`)
  and `@traceable` tracing; we wrote no custom LangChain agents (the loop is ours).
- **Flask** — BSD-3. Web UI server only.
- **Textual / textual-plotext** — MIT. Legacy terminal UI (kept for reference, superseded by Flask).

Everything else (agent loop, prompt, sandbox, workspace tools, runner, batch runner,
metrics, aggregation, GUI, tests, docs) is original coursework written for this project.

**AI assistance:** we used an LLM coding assistant (Claude, via the `pi` harness) as a
pair-programmer under course policy. It drafted/refactored implementation code and tests
under our direction, suggested diagnoses of failure modes from our traces, and helped
draft this document. All design decisions, experiment plans, hyperparameter choices,
pilot interpretation, and submission decisions are ours; AI-suggested code was reviewed,
tested (`pytest`, `ruff`), and validated against live behavior before being kept. Model
inference for the agent itself used the Duke AI Gateway and is disclosed separately from
coding assistance.

---

*How to regenerate results tables: `uv run python -m ctf_agent.metrics` /
`uv run python -m ctf_agent.aggregate results/<dir>/attempts.jsonl` — this file is
refreshed by hand after each experiment batch.*
