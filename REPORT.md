# Design and Evaluation of `ctf-agent`: An LLM-Based Framework for Autonomous Capture-the-Flag Resolution
**Course:** CS 590.007, Duke University  
**Author:** Dan Hendricks

---

## 1. System Architecture and Design Goals

The objective of this project was to develop a robust agentic framework capable of autonomously solving capture-the-flag (CTF) challenges from the CTFTiny benchmark (NYU-LLM-CTF, AAAI'26). The system is designed as a decoupled architecture where the Large Language Model (LLM) serves as the cognitive reasoning core, while a deterministic scaffolding layer manages execution, security, and resource constraints.

### 1.1 Architectural Overview
The system is partitioned into three primary domains: the **Operator** (interface), the **Orchestration Layer** (retry and state management), and the **Agent Loop** (the execution cycle). 

<img width="2400" height="1830" alt="image" src="https://github.com/user-attachments/assets/dd59c785-7c0a-40bc-91d9-9ae0c1b30b6a" />

The architecture prioritizes security and verification integrity through the following mechanisms:
*   **Security Boundary:** All tools are executed within a **bubblewrap jail**, featuring a restricted workspace-only filesystem, no network access, and a sanitized environment.
*   **Verification Integrity:** The agent interacts with the official verifier exclusively through a `submit_flag` tool. Flag verification is performed as a constant-time comparison outside the agent's context, preventing the model from leaking answers or overfitting to known flags.

### 1.2 Component Specifications
*   **Runner (`runner.py`):** Manages benchmark data and ensures a "clean slate" workspace for each challenge, isolating the agent from the ground-truth answer key.
*   **Agent Loop (`agent.py`):** Implements a single-turn-terminated chat loop via the Duke AI Gateway (LiteLLM $\rightarrow$ `langchain-openai`). The loop supports parallel tool calling and a "nudge-not-kill" mechanism to prevent premature termination during multi-turn reasoning.
*   **Sandboxed Toolkit (`tools/`):** A curated environment providing essential binaries (`binutils`, `gdb`, `binwalk`, `openssl`, etc.) and a pinned Python capability layer (`pycryptodome`, `z3`, `pwntools`).
*   **Resource Management:** To mitigate context window exhaustion, the system implements **automatic context elision**, which converts stale tool outputs into one-shot replayable stubs.
*   **Observability:** Every model turn, tool execution, and evaluation event is logged to a local `trace.jsonl` and mirrored to LangSmith for forensic analysis.

---

## 2. Design Challenges and Iterative Refinements

Through trace forensics and pilot testing, several critical failure modes were identified. These were addressed through a systematic iterative process.

### 2.1 Resource and Context Constraints
Early trials indicated that raw binary reads (e.g., reading ELF bytes) rapidly exhausted the 150k token budget. 
*   **Binary-Aware I/O:** We replaced raw dumps with summarized binary views, reducing token consumption by $\sim 8\%$.
*   **Context Elision:** We implemented a `tool_call_id` based stubbing system. This prevented the agent from re-processing massive, redundant outputs in every turn, effectively extending the functional context window.

### 2.2 State and Control Flow
Analysis of failed runs revealed a tendency for agents to either give up prematurely or repeat unsuccessful trajectories.
*   **Nudge-not-Kill Logic:** We modified the loop to allow a limited number of prose-only replies ($\le 2$), ensuring the agent could "think out loud" without triggering a termination state.
*   **Deterministic Evaluator Loop:** We implemented a "coaching" channel that provides the agent with specific feedback (e.g., `INCORRECT (k of N)`), a submission cap, and a halfway consolidation checkpoint.
*   **Handoff Notes:** To prevent trajectory repetition across retries, the system now distills failed attempts into a "handoff note," providing the subsequent attempt with a summary of previously rejected flags and tool-mix outcomes.

### 2.3 Environment and Tooling Gaps
Forensics showed that several failures were due to environment starvation (e.g., missing `unzip` or `torch`) rather than reasoning errors.
*   **Full Toolkit Integration:** We transitioned from a lean image to a comprehensive "capability layer" image (V13), baking in all required binaries and Python libraries to remove infrastructure as a variable.
*   **Dynamic Provisioning:** We developed a workspace-derived pre-flight provisioner (`provision.py`) that installs whitelisted dependencies based on files visible to the agent, ensuring the environment mirrors the challenge's requirements without leaking solution data.
*   **Import Resolution:** We resolved a critical issue where `python3 -I` isolated scripts from sibling imports by transitioning to `python3 -E -s`.

### 2.4 Opaque Error Surfaces (repeat traps)
Config-D trace mining (200-step budget) exposed a costlier class: tool errors that were *honest but unactionable*, which the model then retried verbatim until the budget died.
*   **`run_python` contract:** ~60 attempts passed code via `stdin` on non-existent files and received only "run_python needs a .py file". The error now names the fix (write the file first; `stdin` is program *input*) and the working inline idiom (`python3 - <<'PY' ... PY`).
*   **Hallucinated tool calls:** ~14 turns were lost calling shell binaries (`strings`, `objdump`, `hexdump`, `nm`) as if they were agent tools. "unknown tool X" now enumerates the real tools and redirects to `run_command`.
*   **Shell-redirect habits:** `2>/dev/null` hit the command blocklist ~15 times per sweep; the error now teaches the redirect-free idiom (filter inside the pipe).
*   **Malformed arguments and prompt lies:** bare `KeyError('path')` outputs became tool+argument-naming errors; and the SYSTEM prompt stopped advertising `radare2` (removed from the image in V13) after a live `radare2: not found` turn, eliminating claims the environment cannot back.

### 2.5 Method-Level Rev Failures (oracle, observation, verification)
Across the rev-* challenges the same four method-level death patterns recurred, each trace-evidenced:
*   **Oracle never consulted.** Challenges ship a runnable executable — an unlimited yes/no oracle. `tablez` was probed exactly once (`AAAA...`) over four attempts while 30+ static-analysis steps were spent elsewhere.
*   **Manual reconstruction instead of gdb.** ~10 steps of virtual-address→file-offset arithmetic (all rejected or wrong-section) when `gdb -batch -ex 'b *0x84a' -ex 'x/510bx 0xADDR'` prints the table directly from memory.
*   **Context thrash.** 20–40 overlapping `objdump` slices were re-fetched per attempt because disassembly was never persisted to disk once and re-read.
*   **Unverified glue code.** A parser that returned an empty mapping (`MAPPING = {}`) seeded five more solver attempts — the run never checked its own output before building on it.

Two interventions were added (V15/V16): a **deterministic triage brief** that pre-computes the 5–8 static facts every attempt otherwise re-derives (inventory, ELF class/arch/entry, capped needle-string hits — the same bytes the agent could read itself, so zero information advantage), and a **method doctrine** in the system prompt (oracle probing, gdb direct memory reads, persist-once, verify-before-build). Both are ablatable: the brief is gated by `CTF_TRIAGE=off`, and every trace's `meta` line records whether it ran.

---

## 3. Ablation Study and Testing

Each system modification was piloted on a specific challenge before being integrated. The following table summarizes the iterative development (V1–V16).

| Version | Modification | Target Challenge | Result | Decision |
|---|---|---|---|---|
| **V1-V4** | Prompt tuning & Binary-aware I/O | `rev-rap` | Reduced token drift | **Kept** |
| **V5** | Context Elision + Replay Stubs | `rev-tablez` | Prevented context death | **Kept** |
| **V6** | "Nudge-not-kill" prose support | `rev-baby-mult` | Reduced premature exit | **Kept** |
| **V7** | Sandbox-semantics in System Prompt | General | Improved tool usage | **Kept** |
| **V8-V9** | Budget-keyed Evaluator/Coaching | `rev-sourcery` | Synchronized warnings with caps | **Kept** |
| **V10** | Zero-dep extraction paths | `msc-ezmaze` | Resolved Exit 127 errors | **Kept** |
| **V11** | Import fix (`-E -s` vs `-I`) | `Collision-Course` | Resolved `ModuleNotFoundError` | **Kept** |
| **V12** | Workspace-derived provisioning | `msc-ezmaze` | Automated dep. management | **Kept** |
| **V13** | Baked-in Capability Layer | `msc-ezmaze` | Eliminated env. starvation | **Kept** |
| **V14** | Actionable tool errors + prompt truth | Config-D sweep (`rev-*`) | Broke verbatim retry traps (~90 wasted steps/sweep) | **Kept** |
| **V15** | Deterministic triage brief (zero-info) | config-D `rev-*` traces | Removed 5–8 redundant triage steps/attempt | **Kept** |
| **V16** | Rev method doctrine (oracle/gdb/persist-once) | `rev-tablez`, `rev-beleaf` traces | Addresses the four method death patterns | **Kept** |

**Validation:** The final scaffolding was validated using `tests/test_agent_loop.py` with a scripted mock model, ensuring that nudge-logic, elision, and coaching were functionally correct before applying them to LLMs.

---

## 4. Experimental Results

### 4.1 Hyperparameters
For all reported attempts, the following fixed hyperparameters were used: **30 steps, 15-minute timeout, 150k total tokens, 4k character tool-output limit, and a 60s tool timeout.**

*Note: A comparative set was executed under "drifted" caps (100 steps / 300k tokens). These are isolated in Section 4.3 to avoid baseline skew.*

### 4.2 Fixed-Cap Performance (`gpt-oss:120b`)
Out of 18 one-shot attempts across 10 challenges, the agent successfully solved 3.

| Challenge | Trial | Status | Verified | Steps | Input Tokens | Note |
|-----------|-------|--------|----------|-------|--------------|------|
| `cry-babycrypto` | 1 | **Solved** | **Yes** | 10 | 19,643 | One-shot success |
| `cry-hybrid2` | 1 | **Solved** | **Yes** | 23 | 55,717 | One-shot success |
| `rev-tablez` | 1 | token_budget | No | 29 | 145,777 | Context limit reached |
| `rev-beleaf` | 1 | step_limit | No | 30 | 123,239 | Step limit reached |
| `msc-ezmaze` | 1 | step_limit | No | 30 | 97,817 | Complexity wall |

*Trace-meta erratum:* this batch was executed **before per-attempt meta stamping** existed, so the
"fixed caps" label rests on the batch config (not on per-attempt trace evidence). These runs are
classified as `unlabeled (pre-meta harness)` in the consolidated report (`results/RESULTS.md`).

### 4.3 Comparative Analysis of Resource Constraints (Drifted Caps)
To test the hypothesis that reverse-engineering challenges are bound by context capacity, a set of runs was conducted with expanded limits (300k tokens / 100 steps).

*   **Finding:** Under fixed caps, `rev-baby-mult` was never solved. Under expanded caps, it was successfully solved (38 steps, $\sim 199\text{k}$ tokens).
*   **Conclusion (as drawn at the time):** This suggested that the primary bottleneck for the `rev-*` class of challenges is the capacity to hold and interpret disassembly within the token window.

*   **Post-hoc correction (trace-driven, same day):** `rev-baby-mult` was subsequently solved
    under the honest fixed caps (`30 steps · 150k tokens · ≤3 submits`): run
    `rev-baby-mult-20261111T180100-t1`, 16 steps, 54,769 input / 5,432 output tokens.
    The "never solved under fixed caps" premise is therefore **falsified**; budget
    expands the search envelope but is not strictly binding — method variance dominates
    (see §2.4–2.5 for the mechanism).

### 4.4 Summary of Findings
1.  **Tractability of Cryptography:** Cryptographic challenges were solved efficiently within fixed caps, typically requiring $< 60\text{k}$ tokens.
2.  **Capacity Bottlenecks in RE (revised):** Expanded caps converted budget into retry loops,
    not solves: the `200 steps / 1M tokens` bulk sweep produced **0 solves in 15 attempts** while
    raising the error tax vs. fixed caps (26% vs 19%), whereas a 16-step, `~60\text{k}`-token
    fixed-cap attempt solved `rev-baby-mult`. Extra budget is neither necessary (the
    `~60\text{k}`-token solve exists) nor sufficient (the 1M-token sweep).
3.  **Complexity Walls:** `msc-ezmaze` remained unsolved across all configurations, suggesting a higher reasoning threshold or a need for specialized tool-use not present in the current toolkit.
4.  **Consolidated metrics:** a regenerable cross-era report over **all 136 recorded attempts**
    (scored, dev, orphaned traces), every attempt bucketed by its own trace meta, lives at
    `results/RESULTS.md` with companion `results/full_summary.json`; regenerate with
    `uv run python scripts/build_results_report.py`. Headline: 7 verified solves in 122 scored
    attempts (3 under per-attempt-stamped fixed caps, 4 in unlabeled pre-meta runs).

---

## 5. Disclosure of Reused Code and AI Assistance

**Third-Party Components:**
*   **CTFTiny Benchmark:** (GPL-2.0) Used unmodified for challenge data and verification.
*   **LangChain / LangSmith:** (MIT) Used as a chat completion wrapper and for observability.
*   **Flask:** (BSD-3) Used for the web-based monitoring interface.

**AI Assistance:**
An LLM coding assistant (Claude) was used as a pair-programmer for refactoring implementation code and drafting this report. All design decisions, experimental plans, hyperparameter selections, and interpretations of results were made solely by the author. All AI-suggested code was reviewed and validated via `pytest` and `ruff`.
