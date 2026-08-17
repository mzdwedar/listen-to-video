# vl-gpu — asr, ocr, embed-bulk.
#
# Self-contained rather than extending vl-base: the runtime here is Ubuntu/CUDA, and
# copying a Debian-slim venv into it would mean relying on interpreter paths matching
# across distros. Building the venv in place is duller and correct.
#
# Ubuntu 24.04 is chosen because it ships Python 3.12 as its system Python, so no
# third-party PPA is needed. `-cudnn-runtime` (not `-devel`) keeps this several GB
# smaller; the CUDA minor version is pinned because torch and onnxruntime-gpu
# disagreeing about cuDNN is the classic breakage in this stack.
#
# CUDA 12.6.3 rather than 12.4.1: nvidia/cuda publishes ubuntu24.04 only from 12.6.0
# onward, so 12.4.1-cudnn-runtime-ubuntu24.04 has never existed. Of the two ways to
# make the pin real — drop to ubuntu22.04, or move the CUDA minor forward — only the
# latter keeps system Python 3.12, which is the reason 24.04 was picked. 12.6 stays
# within CUDA 12's minor-version compatibility, so wheels built against 12.4 still run.
#
# verified 2026-08-17 against docker.io/nvidia/cuda (linux/amd64 present).

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON=python3.12 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv

WORKDIR /app
# README.md is copied because pyproject declares it as the package readme, and
# hatchling reads it during the build. Omitting it fails at `uv sync`, not here.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra gpu

COPY src/ ./src/
RUN uv sync --frozen --no-dev --extra gpu


FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    VL_MODEL_CACHE=/var/cache/vl/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 vl \
    && mkdir -p /var/cache/vl/models \
    && chown -R vl:vl /var/cache/vl

WORKDIR /app
COPY --from=builder --chown=vl:vl /app/.venv /app/.venv
COPY --from=builder --chown=vl:vl /app/src /app/src

USER vl

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD ["vl", "--help"]

ENTRYPOINT ["vl"]
CMD ["--help"]
