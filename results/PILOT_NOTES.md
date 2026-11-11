# Pilot notes (development only, NOT scored)

Challenge: rev-rap, 1 trial per variant, same 150k-token cap.

| Variant | Status | Steps | Last input tok | Total tok | Elapsed s |
|---|---|---|---|---|---|
| baseline (read_file returns raw bytes) | token_budget | 26 | 15525 | ~163k | 111.7 |
| + binary-aware read_file | token_budget | 27 | 9502 | ~151k | 70.8 |

Observed problems (from trace):
1. read_file dumped ~4 KB of ELF bytes into context (fixed: binary summary).
2. Model rewrote solve.py ~12 times with near-identical objdump/nm variants, re-running the same disassembly (NOT yet fixed).

Hypothesis (partly confirmed): returning a binary summary reduces per-step input growth.
Not confirmed: loop removal. Next candidate change: enforce a per-file write limit or
detect identical tool results and force a strategy change; test before scored run.

## Variant 3: + LangChain agent loop, @traceable tools, repeat-call check
rev-rap, 1 trial, same 150k cap. Result: step_limit at 30 steps, 143k tokens, 0 repeat-blocks.
- Repeat-call check never fired: model rewrites solve.py with small content changes, so
  (name, arguments) differs each time. Hypothesis keyed on exact args was wrong for this failure.
- One submit_flag made, INCORRECT.
- Model stuck on ELF metadata (size, entry point) instead of the check logic.
Decision: do NOT keep tuning on rev-rap (overfitting one challenge). Next: check tool behavior
on a second dev challenge before changing prompts.

## Dev check: rev-checker (Very Easy, original dev challenge), 1 trial, current agent
Result: solved, verified, 42 s, ~11k tokens. Path: read checker.py -> write decode.py ->
run -> write solve.py -> run -> submit_flag CORRECT. Repeat-check blocked one identical
run_python, agent recovered. Confirms tools, flag verification, and metrics end to end.

Diagnosis for rev-rap: the failure is analysis strategy (stuck on ELF size/entry), not
tooling. Hypothesis H2: adding a binary-analysis instruction to the system prompt (inspect
the comparison/loop logic in the disassembly, not header metadata) will move the agent toward
recovering the flag on rev-rap within the same budget.

## H2 result (binary-analysis instruction in system prompt), rev-rap, 1 trial
Result: token_budget at 24 steps, ~153k tokens, 0 submissions.
Behavior moved: 7 of 11 written scripts now use objdump (was ~0 meaningful disassembly before).
Conclusion: H2 changed tool choice but did not produce a recovered flag. Remaining bottleneck is
interpreting the disassembly (extracting the check), not picking tools.
Decision: stop tuning on rev-rap (single-challenge overfit risk). Freeze prompt as-is (H2 kept,
since it is a reasonable general rule and changed behavior) and proceed to scored set.
