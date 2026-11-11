"""Shared test fixtures.

Zero-network guarantee: unit tests must never talk to the LLM gateway or to
LangSmith. run_agent is @traceable-decorated, so tracing is disabled here.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    # Pre-flight provisioning is off in the suite: no test may trigger a real
    # pip install. Tests that exercise provisioning opt back in explicitly and
    # fake the installer.
    monkeypatch.setenv("CTF_PROVISION", "off")
    yield
