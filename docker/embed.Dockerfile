# vl-embed — query embedding + reranking for the interactive `ask` path.
#
# A separate service because vLLM runs one model per process, so the reranker and the
# query embedder cannot share the fusion LLM's instance. Infinity hosts both, and both
# are small enough that one process is fine.
#
# Note these two models are deliberately NOT quantized (CLAUDE.md #15): retrieval works
# by comparing distances, so quantization reorders nearest neighbours and degrades
# `ask` silently, in a way that reads as a prompt bug. ~600MB is not worth that.
#
# Bulk corpus embedding does NOT go through here — it runs in-process on the gpu pool
# with large batches. Same weights, two access patterns, because a million spans and
# one query string have nothing in common but the model.
#
# NOTE: verify INFINITY_TAG against the current release before the first build — this
# pin was written from memory and has not been checked against the registry.

ARG INFINITY_TAG=0.0.77
FROM michaelf34/infinity:${INFINITY_TAG}

ENV VL_MODEL_CACHE=/var/cache/vl/models \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir boto3

COPY docker/sync_weights.py /opt/vl/sync_weights.py
COPY docker/entrypoint-serve.sh /opt/vl/entrypoint-serve.sh
RUN chmod +x /opt/vl/entrypoint-serve.sh \
    && mkdir -p /var/cache/vl/models

ENTRYPOINT ["/opt/vl/entrypoint-serve.sh"]
