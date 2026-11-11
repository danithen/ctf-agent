# CTF Agent — reproducible container (CS 590.007)
#
# Base images are pinned by digest, so every machine building this file gets
# byte-identical layers. `uv sync --frozen` installs exactly the versions in
# uv.lock. Bump deliberately: update the tag, rebuild, and record the new
# digest in git.
#
#   docker build --target test -t ctf-agent:test .   # runs the pytest gate
#   docker build -t ctf-agent:latest .               # runtime image (GUI/CLI)
#
# The agent sandbox itself is bubblewrap (installed below). Inside a plain
# `docker run` the outer seccomp profile may block bwrap's user namespaces;
# run with --security-opt seccomp=unconfined and see README "Containers" for
# why that still leaves the sandbox intact.

FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df AS base

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# bubblewrap is a hard requirement of the agent sandbox — no silent fallback,
# so it ships in the image rather than being documented as a manual step.
# binutils (objdump/strings) · unzip · file (type triage) · bzip2/xz-utils:
# the binary-analysis staples the agent must be able to rely on — their
# absence caused exit-127 dead-ends on real runs (see docs/architecture.md).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bubblewrap ca-certificates binutils unzip file bzip2 xz-utils xxd bsdextrautils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 /uv /usr/local/bin/uv

WORKDIR /app
# Bytecode, envs and the project venv live here; kept out of later layers'
# churn by ordering dependencies before source.
ENV PATH="/app/.venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1


# --- dependencies layer: only re-runs when pyproject/lock change -----------
FROM base AS deps
COPY pyproject.toml uv.lock ./
# configs/scored_set.json gates the challenge selection (read at runtime)
COPY configs ./configs
# bench extra (nyuctf) is required by the runner — not merely a benchmark extra
RUN uv sync --frozen --no-dev --extra bench --no-install-project


# --- test gate: full dev deps + tests; `docker build --target test` = CI ---
FROM deps AS test
RUN uv sync --frozen --all-groups --extra bench --no-install-project
COPY src ./src
COPY tests ./tests
COPY README.md ./
RUN uv sync --frozen --all-groups --extra bench
# Zero-token by design: tests use an offline fake gateway and conftest forces
# LANGSMITH_TRACING=false (and the real .env is never copied into this image).
CMD ["python", "-m", "pytest", "tests/", "-q"]


# --- runtime: non-root user, runtime deps only, data on /data --------------
FROM deps AS runtime
COPY src ./src
RUN uv sync --frozen --no-dev --extra bench  # deps come from the deps layer; this adds the project

# Run defaults: container-friendly GUI binding; all state under /data so a
# single volume carries runs+results between machines.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    CTF_GUI_HOST=0.0.0.0 \
    CTF_GUI_DEBUG=0 \
    CTF_RUNS_DIR=/data/runs \
    CTF_TUI_RESULTS=/data/results \
    CTF_DATASET_JSON=/data/benchmark/ctftiny.json \
    CTF_BENCHMARK_DIR=/data/benchmark

RUN useradd --create-home --uid 1000 agent \
    && mkdir -p /data/runs/tui /data/results \
    && chown -R agent:agent /data
USER agent
WORKDIR /home/agent    # cwd for relative .env fallback; /app/.env is mounted by compose

EXPOSE 5000
CMD ["ctf-flask-gui"]
