# Agent architecture — mapped to Anthropic's *Building Effective Agents*

This document maps the ctf-agent harness onto the design guidance in
[Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
(Anthropic, Dec 2024). The one-sentence summary of that essay: **start with the
simplest thing that works** — often a single LLM call or a composed workflow — and
reach for a full **agent** (a model driving a tool loop on environmental feedback)
only when the task demands it, keeping every added pattern simple, transparent,
and verifiable.

Our target domain (rev/crypto/misc CTF challenges under a hard token budget) is
exactly the essay's case for an agent: open-ended, multi-step, no known procedure
in advance — so the core of this project *is* the essay's canonical agent loop.
Everything else we added is thin, deterministic guardrails around that loop.

## 1. System diagram

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
        EXEC["tool executor<br>list_files · read_file · run_python<br>run_command · write_file · strings · submit_flag"]:::model
        CTX["context manager<br>repeat-call guard · elision of stale outputs<br>into one-shot replayable stubs"]:::guard
        EV["deterministic evaluator<br>halfway checkpoint — 50 pct of the binding budget<br>urgent submit-now warning — 85 pct of the token cap<br>INCORRECT (k of N) rejection feedback · submission cap 3<br>nudge-not-kill for prose — max 2"]:::guard
    end

    SBX[["bubblewrap jail<br>no network · workspace-only filesystem<br>throwaway HOME · fixed PATH"]]:::boundary
    VER{{"official verifier<br>CTFChallenge.flag — the only path to verified = true"}}:::boundary
    DONE["✓ solved"]:::store
    RES[("results: attempts.jsonl<br>summary.json · config snapshot")]:::store
    TRC[("observability: trace.jsonl<br>→ LangSmith (traceable)")]:::store

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
    VER -->|"INCORRECT (k of N) — still counts as an attempt"| EV
    EXEC --- SBX
    LOOP -.->|"every model, tool, submission,<br>evaluator and elision event"| TRC
    ORCH -->|"one row per attempt"| RES
```

Reading guide:

- **The center is the model loop** (`MSG → MODEL → EXEC → CTX → MODEL`): the LLM
  decides every step; nothing in the harness scripts its behavior. This is the
  essay's "agent" — *models using tools in a loop based on environmental feedback*.
- **Yellow nodes are deterministic guardrails.** They never call the LLM, never
  invent content, and only (a) inject short coaching messages, (b) trim context,
  or (c) enforce hard stops (budgets, submission cap).
- **Red edges/nodes are the integrity boundary.** Tool execution is jailed
  (bubblewrap); the real flag lives only in the official verifier — `verified=true`
  is reachable *exclusively* by submitting a candidate through `submit_flag`.
- **The dashed edge from the handoff-note builder** is the evaluator-optimizer
  loop *between* attempts: retries of the same challenge receive a distilled
  summary of the failed attempt (what was tried, what was rejected, where it left
  off). Batch/scored runs deliberately skip this edge so pass@k stays honest.

## 2. One attempt, as a sequence

```mermaid
sequenceDiagram
    autonumber
    participant O as Orchestrator (GUI/batch)
    participant L as run_agent
    participant M as Model (gateway)
    participant T as Tools in bubblewrap jail
    participant V as Official verifier

    O->>L: attempt k — clean slate (± handoff note)
    L->>M: SYSTEM + challenge description (+ note)
    loop until solved | token budget | step limit | submission cap
        M->>L: tool_calls (the model decides)
        L->>T: execute under jail — output clipped
        T-->>L: result
        L->>L: dedupe repeats · elide stale outputs
        Note over L: evaluator: halfway checkpoint at 50 pct of the<br>binding budget · urgent warning at 85 pct of tokens
        opt model replies with prose only
            L->>M: nudge — call a tool or submit (max 2)
        end
        opt model chooses to submit_flag
            L->>V: hmac compare against CTFChallenge.flag
            V-->>L: CORRECT (solved) · INCORRECT (k of N) — keep going
        end
    end
    L-->>O: verdict · steps · tokens · submissions · trace.jsonl
```

## 3. Mapping: essay recommendation → this project

| Anthropic recommendation | How this project implements it |
|---|---|
| **Agents** = model in a loop, acting on environment feedback | `run_agent`: one chat loop; the model picks from 7 tools every turn; the harness only tracks budgets and relays results |
| **Start simple; add complexity only when it pays** | No planners, no multi-agent, no RAG. Complexity was added one guardrail at a time, each kept because a measured failure motivated it (REPORT §3): elision → token deaths; evaluator → 7 of 11 deaths with no submission; retries repeating trajectories |
| **Workflows vs agents** — don't over-orchestrate | The *only* orchestration (retry chains, batch trials) is deterministic Python, never model-driven. Attempts are sequential — parallel workers would break fair time-based limits |
| **Evaluator-optimizer** pattern | Implemented *deterministically* (no judge LLM, zero extra tokens): in-run — halfway checkpoint, urgent near-limit warning, rich `INCORRECT (k of N)` feedback, submission cap; between attempts — handoff notes distilled from the failed trace |
| **Prompt chaining** | The SYSTEM prompt encodes verified tool facts (pipes allowed, redirects blocked, tail truncation, notes-file habit) so the model doesn't burn steps rediscovering the interface; the loop appends step-proximity coaching, not blind re-prompts |
| **Tool design (ACI)** matters as much as the model | Small toolset with documented, sandboxed semantics; outputs clipped at the tail with a stub marker; repeat-call guard gives immediate, actionable errors ("you already ran this exact call") instead of silent repetition |
| **Ground truth & verification** | The benchmark's official flag check is the *only* success signal (hmac compare, outside agent context). The agent can see a verdict, never the answer — so `verified=true` cannot be gamed |
| **Transparency** | Every model/tool/submission/evaluator/elision event is appended to `trace.jsonl` (first line `meta` stamps the exact run limits; last line `done` states the outcome) and mirrored to LangSmith; `attempts.jsonl` records every attempt including retries |
| **Human control points** | Config tab (model, limits, retries, handoff notes — persisted visibly), fixed-limit snapshots (`config.json`) per results set, retry gating the human chose ("repeat only when the model fails") |
| **Guardrails around the agent** | Token/step/time budgets; nudge cap; submission cap; elision at 50% of the token cap; bubblewrap jail (no network, workspace-only FS) as the security boundary |

## 4. What we deliberately do *not* do (per the essay)

- **No orchestrator-workers or parallelization** — sequential attempts keep
  time-based limits fair and results comparable.
- **No LLM-as-judge** — the essay's evaluator-optimizer uses a model in a loop;
  our evaluator is pure code. Same feedback channels, zero token cost, fully
  deterministic and testable.
- **No multi-agent / planner-executor splits** — a single strong loop with good
  tools beat those on this benchmark's budget profile; scaffolding beyond the
  loop was only added when a trace-proven failure demanded it.
- **No hidden state** — the agent's whole world (staged files, description,
  clipped outputs, evaluator notes) is visible in the trace and never includes
  benchmark answers.
