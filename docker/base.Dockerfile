# vl-base — shared build layer.
#
# Holds uv, the lockfile and the project source with base dependencies resolved. The
# cpu image extends this as its builder so base dependencies are downloaded once
# rather than per pool. The gpu image cannot: its runtime is Ubuntu/CUDA rather than
# Debian slim, and copying a venv across distros is asking for trouble.
#
# Base dependencies deliberately exclude everything heavy — see pyproject.toml. That
# is what keeps this image ~250MB and the test suite runnable on an arm64 laptop.
#
# verified 2026-08-17: python:3.12-slim and ghcr.io/astral-sh/uv:0.8.15 both resolve,
# linux/amd64 present.

FROM python:3.12-slim AS base

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, without the project, so a source change doesn't invalidate the
# dependency layer.
# README.md is copied because pyproject declares it as the package readme, and
# hatchling reads it during the build. Omitting it fails at `uv sync`, not here.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev
