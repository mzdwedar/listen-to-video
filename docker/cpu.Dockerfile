# vl-cpu — normalize, keyframes, fer, ingest, sweepers, report.
#
# INVARIANT: no CUDA, ever (CLAUDE.md #13). Per-pool images exist precisely to keep
# ONNX Runtime and torch out of the same process, and this image is the cheap one —
# it scales to absorb ffmpeg and scene detection, which are ~18s per video of pure
# CPU work that must never occupy a GPU node.
#
# Expects `vl-base` to be built first (see `mise run build`).

FROM vl-base AS builder
RUN uv sync --frozen --no-dev --extra cpu


FROM python:3.12-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    VL_MODEL_CACHE=/var/cache/vl/models

# ffmpeg does the normalize stage; libGL and libglib are opencv's runtime deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 vl \
    && mkdir -p /var/cache/vl/models \
    && chown -R vl:vl /var/cache/vl

WORKDIR /app
COPY --from=builder --chown=vl:vl /app/.venv /app/.venv
COPY --from=builder --chown=vl:vl /app/src /app/src

USER vl

# Workers report liveness through heartbeat freshness in the jobs table; this only
# proves the interpreter and entrypoint are intact.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD ["vl", "--help"]

ENTRYPOINT ["vl"]
CMD ["--help"]
