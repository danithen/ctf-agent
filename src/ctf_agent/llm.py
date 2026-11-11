"""Minimal OpenAI-compatible chat client for the Duke LiteLLM gateway."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class GatewayError(RuntimeError):
    pass


@dataclass
class ChatResult:
    message: dict[str, Any]
    input_tokens: int | None
    output_tokens: int | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


MAX_RETRIES = 3


def _post_with_backoff(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
    """Sequential retries only, honoring Retry-After on 429/5xx."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            retriable = e.code == 429 or 500 <= e.code < 600
            if not retriable or attempt == MAX_RETRIES:
                # Do not include response bodies that may echo request content.
                raise GatewayError(f"HTTP {e.code} from gateway") from None
            wait = float(e.headers.get("Retry-After") or 2 ** (attempt + 2))
            time.sleep(min(wait, 120))
    raise GatewayError("unreachable")


def chat_langchain(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: float = 120,
) -> ChatResult:
    """Same contract as chat(), but routed through LangChain so LangSmith traces each call."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_openai import ChatOpenAI

    base = os.environ["LITELLM_BASE_URL"].rstrip("/") + "/v1"
    if max_tokens is None:
        max_tokens = int(os.environ.get("CTF_MAX_OUTPUT_TOKENS", "1024"))
    llm = ChatOpenAI(
        model=model or os.environ["CTF_MODEL"],
        base_url=base,
        api_key=os.environ["LITELLM_TOKEN"],
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=MAX_RETRIES,
    )
    if tools:
        llm = llm.bind_tools([t["function"] for t in tools], tool_choice="auto")

    lc: list[Any] = []
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if role == "system":
            lc.append(SystemMessage(content=content))
        elif role == "user":
            lc.append(HumanMessage(content=content))
        elif role == "tool":
            lc.append(ToolMessage(content=content, tool_call_id=m["tool_call_id"]))
        elif role == "assistant":
            calls = [
                {"name": c["function"]["name"],
                 "args": json.loads(c["function"].get("arguments") or "{}"),
                 "id": c["id"]}
                for c in (m.get("tool_calls") or [])
            ]
            lc.append(AIMessage(content=content, tool_calls=calls))

    out = llm.invoke(lc)
    message: dict[str, Any] = {"role": "assistant", "content": out.content or None}
    if out.tool_calls:
        message["tool_calls"] = [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
            for tc in out.tool_calls
        ]
    usage = out.usage_metadata or {}
    return ChatResult(
        message=message,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: float = 120,
) -> ChatResult:
    base = os.environ["LITELLM_BASE_URL"].rstrip("/")
    token = os.environ["LITELLM_TOKEN"]
    body: dict[str, Any] = {
        "model": model or os.environ["CTF_MODEL"],
        "messages": messages,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if max_tokens is None:
        max_tokens = int(os.environ.get("CTF_MAX_OUTPUT_TOKENS", "1024"))
    body["max_completion_tokens"] = max_tokens

    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    data = _post_with_backoff(req, timeout)

    usage = data.get("usage") or {}
    return ChatResult(
        message=data["choices"][0]["message"],
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        raw={"id": data.get("id"), "model": data.get("model")},
    )
