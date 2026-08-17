#!/usr/bin/env bash
# Sync weights from object storage, then hand off to the server.
#
# `exec` matters: the server must become PID 1 so it receives SIGTERM directly. A
# shell in between would swallow it, and on Spot that means losing the two-minute
# interruption warning.
set -euo pipefail

python3 /opt/vl/sync_weights.py

echo "[entrypoint] starting: $*"
exec "$@"
