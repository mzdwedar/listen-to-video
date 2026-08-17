"""`vl` command surface.

Commands are declared here in full because the CLI is the product's only interface and
its shape is a documented contract (see README.md). Bodies land slice by slice; a
command whose slice hasn't been built yet exits non-zero with a pointer to the plan
rather than pretending to work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from vl.config import Settings
from vl.domain.stages import STAGE_POOL, Stage

app = typer.Typer(
    name="vl",
    help="Video listening — queryable Voice-of-the-Customer corpus.",
    no_args_is_help=True,
    add_completion=False,
)

models_app = typer.Typer(
    name="models", help="Fetch, quantize and publish models.", no_args_is_help=True
)
app.add_typer(models_app)


def _pending(slice_no: int, what: str) -> None:
    """Fail loudly for a command whose slice hasn't been implemented yet."""
    typer.secho(f"{what} is not implemented yet (build order slice {slice_no}).", fg="yellow")
    raise typer.Exit(code=2)


@models_app.command("prepare")
def models_prepare(
    only: Annotated[str | None, typer.Option(help="Prepare a single model by name.")] = None,
) -> None:
    """Fetch from Hugging Face at a pinned revision, quantize, gate, publish to S3.

    The only place quantization happens (invariant 12). Serving containers sync
    weights by model_id and never reach Hugging Face at runtime.
    """
    _pending(1, "models prepare")


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Folder of videos to ingest.")],
) -> None:
    """Register videos, dedup by content hash and perceptual hash, enqueue stage 1."""
    _pending(1, "ingest")


@app.command()
def sweep(
    stage: Annotated[Stage, typer.Option(help="Which stage to sweep.")],
    batch: Annotated[int | None, typer.Option(help="Override the batch size.")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan, claim nothing.")
    ] = False,
) -> None:
    """Batch-claim videos for one stage, load the model once, process, unload.

    Stage-batched rather than per-video: model load is 30-60s, so a worker amortises
    it across the whole batch instead of paying it every time.
    """
    settings = Settings()
    size = batch or settings.sweep_batch_size

    if dry_run:
        typer.echo(f"stage:  {stage.value}")
        typer.echo(f"pool:   {STAGE_POOL[stage].value}")
        typer.echo(f"batch:  {size} videos per claim")
        typer.echo(f"lease:  {settings.lease_duration_s}s")
        return

    _pending(2, f"sweep --stage {stage.value}")


@app.command()
def repair() -> None:
    """Retry stages missing on degraded videos, on exponential backoff.

    Late success bumps `evidence_generation`, which triggers re-interpretation of that
    video with no GPU re-extraction.
    """
    _pending(2, "repair")


@app.command()
def reinterpret(
    prompt_version: Annotated[str, typer.Option(help="Prompt version to rebuild with.")],
) -> None:
    """Rebuild every conclusion from stored evidence. No GPU extraction.

    This is the payoff of the evidence/interpretation split; slice 8 proves it holds.
    """
    _pending(8, "reinterpret")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Natural-language question.")],
) -> None:
    """Answer from the corpus, citing video_id and timestamp for every claim."""
    _pending(3, "ask")


@app.command()
def report(
    out: Annotated[Path, typer.Option(help="Where to write the HTML report.")] = Path(
        "report.html"
    ),
) -> None:
    """Complaints clustered and ranked by video count, with grounding evidence."""
    _pending(7, "report")


@app.command()
def stats(
    cost: Annotated[bool, typer.Option("--cost", help="Break down $/video by stage.")] = False,
) -> None:
    """Queue depth, failure rates, GPU seconds and cost per video."""
    _pending(9, "stats")


@app.command()
def doctor() -> None:
    """Smoke-test real models on the GPU box. Never runs in CI or on macOS."""
    _pending(1, "doctor")


if __name__ == "__main__":
    app()
