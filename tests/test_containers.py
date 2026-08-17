"""Container invariants, checked statically.

These encode invariants 12, 13 and 14 from CLAUDE.md. They are deliberately static
(parsing Dockerfiles rather than building them) so they run in CI on any platform in
milliseconds — a real `docker build` of the GPU images cannot run on macOS at all.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
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


def arg_defaults(text: str) -> dict[str, str]:
    """`ARG NAME=value` defaults, used to resolve `${NAME}` inside a FROM."""
    out = {}
    for arg in instructions(text, "ARG"):
        name, _, value = arg.partition("=")
        if value:
            out[name.strip()] = value.strip()
    return out


def external_images(name: str) -> list[str]:
    """Every image reference this Dockerfile pulls from a registry.

    Covers `COPY --from=<image>` as well as `FROM`: the uv binary arrives that way, and
    an unresolvable pin there fails the build exactly as hard as a bad base.

    Stage names (`FROM x AS builder`, `COPY --from=builder`) and locally built images
    (`vl-base`) are excluded — they resolve inside the build, not against a registry.
    """
    text = dockerfile(name)
    defaults = arg_defaults(text)
    stages = {
        f.split(" AS ")[-1].strip().lower() for f in instructions(text, "FROM") if " AS " in f
    }

    refs = [f.split(" AS ")[0].strip() for f in instructions(text, "FROM")]
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        for match in re.finditer(r"--from=(\S+)", line):
            refs.append(match.group(1))

    resolved = []
    for ref in refs:
        for key, value in defaults.items():
            ref = ref.replace(f"${{{key}}}", value)
        if ref.lower() in stages or ref.startswith("vl-") or "${" in ref:
            continue
        resolved.append(ref)
    return resolved


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


@pytest.mark.network
@pytest.mark.parametrize("name", POOL_IMAGES)
def test_external_base_images_resolve(name: str) -> None:
    """Every pinned external image must exist in its registry, for linux/amd64.

    A tag that was written from memory fails at `docker build`, which on the GPU images
    means it fails on the one machine that can build them — after a context upload and
    a long wait. Resolving the manifest costs a second and finds it here instead.

    amd64 specifically, because invariant 14 is what the images are built for; an
    arm64-only tag would resolve on a dev machine and fail on ECS.
    """
    docker_cli = shutil.which("docker")
    if docker_cli is None:
        pytest.skip("docker CLI not available")

    for ref in external_images(name):
        result = subprocess.run(  # noqa: S603
            [docker_cli, "manifest", "inspect", ref],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"{name}.Dockerfile pins an unresolvable image: {ref}"

        manifest = json.loads(result.stdout)
        entries = manifest.get("manifests")
        if entries is None:  # single-architecture image
            continue
        platforms = {(m["platform"].get("os"), m["platform"].get("architecture")) for m in entries}
        assert ("linux", "amd64") in platforms, f"{ref} has no linux/amd64 variant"


@pytest.mark.parametrize("name", POOL_IMAGES)
def test_external_pins_record_when_they_were_verified(name: str) -> None:
    """A pin nobody has checked is a guess, and guesses here surface late.

    Three of these tags were originally written from memory and one of them did not
    exist. Requiring a dated note next to the pin makes the difference between
    "verified" and "plausible" visible without re-querying the registry.
    """
    if not external_images(name):
        pytest.skip(f"{name}.Dockerfile pulls nothing from a registry")

    text = dockerfile(name)
    assert re.search(r"#.*verified\s+\d{4}-\d{2}-\d{2}", text, re.IGNORECASE), (
        f"{name}.Dockerfile pins an external image without a `# verified YYYY-MM-DD` note"
    )


def test_build_metadata_files_reach_the_build_context() -> None:
    """Files the build backend reads at build time must be COPYed in.

    `pyproject.toml` declares `readme = "README.md"`, and hatchling reads it while
    building the wheel. Leaving it out of the image fails deep inside `uv sync` with an
    OSError, not at the COPY line — so this asserts it up front.
    """
    pyproject = (REPO / "pyproject.toml").read_text()
    match = re.search(r'^readme\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert match, "pyproject no longer declares a readme; drop this test if intentional"
    readme = match.group(1)

    for name in POOL_IMAGES:
        copied = " ".join(instructions(dockerfile(name), "COPY"))
        # Only images that bring in pyproject themselves need it; `cpu` inherits both
        # from `vl-base`.
        if "pyproject.toml" not in copied:
            continue
        assert readme in copied, f"{name}.Dockerfile copies pyproject without {readme}"
