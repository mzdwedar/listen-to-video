#!/usr/bin/env python3
"""Sync one model's weights from object storage into a local cache.

Runs in the serving images before the server starts. Exists because weights are never
baked into images and never fetched from Hugging Face at runtime (CLAUDE.md #12):
baking them would make every scale-out event a multi-gigabyte registry pull, and
fetching from Hugging Face would put a third party in the startup path of every task.

Weights are published once by `vl models prepare`, keyed by model_id.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config


def sync(bucket: str, model_id: str, dest: Path) -> None:
    marker = dest / ".complete"
    if marker.exists():
        print(f"[sync] {model_id} already cached at {dest}", flush=True)
        return

    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=os.environ.get("VL_S3_ENDPOINT") or None,
        aws_access_key_id=os.environ.get("VL_S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("VL_S3_SECRET_KEY"),
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )

    prefix = f"models/{model_id}/"
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix) :]
            if not rel:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, obj["Key"], str(target))
            count += 1

    if count == 0:
        sys.exit(
            f"[sync] no objects under s3://{bucket}/{prefix} — "
            "has `vl models prepare` been run for this model_id?"
        )

    # Written last so an interrupted sync is retried rather than trusted.
    marker.touch()
    print(f"[sync] fetched {count} files for {model_id}", flush=True)


def main() -> None:
    model_id = os.environ.get("VL_MODEL_ID")
    if not model_id:
        sys.exit("[sync] VL_MODEL_ID is required")

    bucket = os.environ.get("VL_S3_BUCKET", "vl-artifacts")
    cache = Path(os.environ.get("VL_MODEL_CACHE", "/var/cache/vl/models"))
    sync(bucket, model_id, cache / model_id)


if __name__ == "__main__":
    main()
