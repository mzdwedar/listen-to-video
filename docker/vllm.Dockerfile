# vl-vllm — run as TWO services from this one image.
#
#   vlm-serve : Qwen2.5-VL, tuned for prefill throughput (image captioning is a large
#               prefill producing a short caption — the inverse of chat).
#   llm-serve : Qwen3-8B, guided decoding + priority scheduling so an interactive
#               `ask` preempts queued fusion work.
#
# Same bits, different command. Their vLLM flags have almost nothing in common, so
# resist any urge to unify them behind one config.
#
# NOTE: verify VLLM_TAG against the current release before the first build — this pin
# was written from memory and has not been checked against the registry.

ARG VLLM_TAG=v0.11.0
FROM vllm/vllm-openai:${VLLM_TAG}

ENV VL_MODEL_CACHE=/var/cache/vl/models \
    PYTHONUNBUFFERED=1

# boto3 only — the weight sync needs nothing else, and the vl package itself has no
# business in a serving image.
RUN pip install --no-cache-dir boto3

COPY docker/sync_weights.py /opt/vl/sync_weights.py
COPY docker/entrypoint-serve.sh /opt/vl/entrypoint-serve.sh
RUN chmod +x /opt/vl/entrypoint-serve.sh \
    && mkdir -p /var/cache/vl/models

ENTRYPOINT ["/opt/vl/entrypoint-serve.sh"]
