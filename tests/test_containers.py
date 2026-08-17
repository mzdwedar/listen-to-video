"""Container invariants, checked statically.

These encode invariants 12, 13 and 14 from CLAUDE.md. They are deliberately static
(parsing Dockerfiles rather than building them) so they run in CI on any platform in
milliseconds — a real `docker build` of the GPU images cannot run on macOS at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
DOCKER = REPO / "docker"

POOL_IMAGES = ["base", "cpu", "gpu", "vllm", "embed"]

# Images that run as long-lived containers we control the entrypoint of. `base` is a
# build layer only and `vllm` inherits an upstream entrypoint, so neither is required
# to declare its own user.
RUNTIME_IMAGES = ["cpu", "gpu"]


def dockerfile(name: str) -> str:
    return (DOCKER / f"{name}.Dockerfile").read_text()


def uncommented(text: str) -> str:
    """Dockerfile body with comments stripped.

    Comments legitimately *name* the things an image must not contain — that's how the
    invariant is explained to the next reader — so prose must not trip these checks.
    """
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#")).lower()


def instructions(text: str, keyword: str) -> list[str]:
    """Return the argument of every `KEYWORD ...` line, ignoring comments."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if line.upper().startswith(f"{keyword.upper()} "):
            out.append(line[len(keyword) + 1 :].strip())
    return out


@pytest.mark.parametrize("name", POOL_IMAGES)
def test_every_pool_has_a_dockerfile(name: str) -> None:
    assert (DOCKER / f"{name}.Dockerfile").is_file()


def test_cpu_image_has_no_cuda() -> None:
    """Invariant 13: per-pool images exist to keep ONNX Runtime and torch apart.

    The moment CUDA appears in the cpu image, that separation is gone and the
    ~600MB image becomes a multi-gigabyte one.
    """
    text = uncommented(dockerfile("cpu"))
    for forbidden in ["nvidia", "cuda", "onnxruntime-gpu", "torch"]:
        assert forbidden not in text, f"cpu.Dockerfile must not reference {forbidden!r}"


def test_gpu_image_pins_a_cuda_runtime_base() -> None:
    """`-devel` bases are gigabytes larger, and an unpinned tag breaks reproducibly."""
    froms = instructions(dockerfile("gpu"), "FROM")
    cuda = [f for f in froms if "cuda" in f.lower()]
    assert cuda, "gpu.Dockerfile must build on a CUDA base"
    for ref in cuda:
        assert "-devel" not in ref, f"use a runtime base, not devel: {ref}"
        assert ":latest" not in ref, f"pin the CUDA version: {ref}"
        assert re.search(r":\d+\.\d+", ref), f"pin a major.minor CUDA version: {ref}"


@pytest.mark.parametrize("name", RUNTIME_IMAGES)
def test_runtime_images_drop_root(name: str) -> None:
    users = instructions(dockerfile(name), "USER")
    assert users, f"{name}.Dockerfile must declare a non-root USER"
    assert users[-1] != "root", f"{name}.Dockerfile ends up running as root"


@pytest.mark.parametrize("name", POOL_IMAGES)
def test_no_image_bakes_model_weights(name: str) -> None:
    """Invariant 12: weights are synced from object storage by model_id at runtime.

    Baking them in makes every scale-out event a multi-gigabyte registry pull and
    turns a model swap into an image rebuild.
    """
    text = uncommented(dockerfile(name))
    for forbidden in ["huggingface-cli download", "snapshot_download", "hf_hub_download"]:
        assert forbidden not in text, f"{name}.Dockerfile must not download weights"


def test_compose_is_valid_and_covers_the_stack() -> None:
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    services = compose["services"]

    # Backing services that become RDS and S3 on AWS.
    assert "postgres" in services
    assert "minio" in services

    # pgvector must be available; plain postgres will not do.
    assert "pgvector" in services["postgres"]["image"]

    # Worker pools and model services.
    for expected in ["cpu-worker", "gpu-worker", "vlm-serve", "llm-serve", "embed-serve"]:
        assert expected in services, f"compose is missing {expected}"


def test_vlm_and_llm_are_one_image_two_services() -> None:
    """They share `vl-vllm` but differ in command: prefill-tuned vs guided decoding."""
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    vlm, llm = compose["services"]["vlm-serve"], compose["services"]["llm-serve"]

    def image_ref(svc: dict) -> str:
        build = svc.get("build")
        return build["dockerfile"] if isinstance(build, dict) else svc.get("image", "")

    assert image_ref(vlm) == image_ref(llm), "vlm and llm should share one image"
    assert vlm.get("command") != llm.get("command"), "they must differ in command"


def test_build_task_targets_amd64() -> None:
    """Invariant 14: dev machines are arm64, ECS is not.

    A plain `docker build` here produces images ECS silently cannot run.
    """
    text = (REPO / "mise.toml").read_text()
    assert "linux/amd64" in text, "the build task must pass --platform linux/amd64"
