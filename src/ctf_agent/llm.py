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
