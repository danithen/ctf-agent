"""One model/tool-call round trip to verify wiring. Not part of the agent loop."""

from __future__ import annotations

import json
import sys

from ctf_agent.config import load_env
from ctf_agent.llm import chat

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo_number",
            "description": "Return the given integer.",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
                "additionalProperties": False,
            },
        },
    }
]


def main() -> int:
    load_env()
    msgs = [
        {"role": "system", "content": "Call the echo_number tool with n=42. Do not answer in text."},
        {"role": "user", "content": "Go."},
    ]
    r1 = chat(msgs, TOOLS, max_tokens=200)
    calls = r1.message.get("tool_calls") or []
    print("tool_calls:", [(c["function"]["name"], c["function"]["arguments"]) for c in calls])
    print("usage in/out:", r1.input_tokens, r1.output_tokens)
    if not calls:
        print("FAIL: model did not call the tool; content:", r1.message.get("content"))
        return 1

    msgs.append(r1.message)
    for c in calls:
        args = json.loads(c["function"]["arguments"])
        msgs.append({"role": "tool", "tool_call_id": c["id"], "content": str(args.get("n"))})
    r2 = chat(msgs, TOOLS, max_tokens=200)
    print("final:", (r2.message.get("content") or "").strip()[:200])
    print("usage in/out:", r2.input_tokens, r2.output_tokens)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
