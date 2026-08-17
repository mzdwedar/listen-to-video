"""The CLI surface is the product's only interface, so its shape is a contract."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from vl.cli import app

runner = CliRunner()

# Every command named in README.md and CLAUDE.md. If one of these disappears, the
# docs are lying.
EXPECTED_COMMANDS = [
    "ingest",
    "sweep",
    "repair",
    "reinterpret",
    "ask",
    "report",
    "stats",
    "doctor",
]


def test_help_succeeds() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize("command", EXPECTED_COMMANDS)
def test_command_exists(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, f"`vl {command}` is missing"


def test_models_prepare_exists() -> None:
    """Quantization happens here and nowhere else (invariant 12)."""
    result = runner.invoke(app, ["models", "prepare", "--help"])
    assert result.exit_code == 0


def test_sweep_requires_a_stage() -> None:
    """Sweeps are stage-batched by design — a stageless sweep is meaningless."""
    result = runner.invoke(app, ["sweep"])
    assert result.exit_code != 0


@pytest.mark.parametrize(
    "stage",
    ["normalize", "asr", "keyframes", "ocr", "vlm", "fer", "fuse"],
)
def test_sweep_accepts_every_pipeline_stage(stage: str) -> None:
    """Rejected at parse time means the stage enum and the pipeline disagree."""
    result = runner.invoke(app, ["sweep", "--stage", stage, "--dry-run"])
    assert result.exit_code == 0, result.output


def test_sweep_rejects_an_unknown_stage() -> None:
    result = runner.invoke(app, ["sweep", "--stage", "transcribe", "--dry-run"])
    assert result.exit_code != 0
