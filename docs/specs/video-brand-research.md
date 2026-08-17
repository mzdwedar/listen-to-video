# Spec: Video Listening — Voice-of-the-Customer Corpus from Video

**Status:** Active · **Scope:** whole system · **Version:** 1.0

This document restates the system defined in `README.md` (narrative) and `CLAUDE.md`
(operating contract) in a structured form that task breakdown can slice. It introduces
no new requirements. Where the two source documents disagree, `CLAUDE.md` wins and the
disagreement is recorded under [Open Questions](#open-questions).

---

## Objective

### What we're building

A system that ingests raw video and turns it into a **queryable corpus of
Voice-of-the-Customer evidence**, without depending on text tags, captions, or platform
metadata.

Social listening and call-center analytics operate on text: reviews, tickets,
transcripts. Video-native customer feedback — short-form product reviews,
user-research recordings, video support tickets — is a blind spot, because the signal
lives in three streams at once (speech, visuals, on-screen text) and none of it is
tagged.

The system extracts all three streams, aligns them on a single timeline, fuses them
into one prompt for a self-hosted LLM, and stores the result as a permanent, searchable
corpus.

### The final "listening" step

Once audio, visual, and OCR streams are structured and aligned, the LLM performs two
jobs over the interleaved timeline:

1. **Intent extraction** — is the speaker requesting a refund, reporting a bug,
   praising a feature, comparing against a competitor?
2. **Contextual sentiment** — the transcript says *"oh, this is just great"*; the visual
   stream reports a frustrated expression and a screen showing an error dialog. Fusing
   them classifies the utterance as sarcastic/negative rather than positive.

The second job is the reason the system exists and the reason it is expensive. It is
also the project's largest unproven claim — see [Risks](#risks).

### Who it's for

A **brand or CX analyst**. They monitor text channels today and sample video by hand.
The goal is for them to see *"17 videos complain about the same bug this week"* now,
rather than inferring it from a support-ticket spike a month later.

### What success looks like for that analyst

```
$ vl ask "what are people saying about battery life since the v3 update?"

Battery complaints rose sharply after v3, concentrated on overnight drain
rather than heavy use. 14 videos mention it directly.

  vid_a81f @ 0:47   "it dies by noon and I barely touch it"
  vid_c024 @ 2:13   "I'm charging twice a day now, never used to"
  vid_9e3b @ 1:02   screen shows "Battery 12%" at 4h uptime
  … 11 more
```

Every claim cites a video and a timestamp. Opening the video at that moment shows the
evidence.

### The organizing principle: evidence vs. interpretation

This split governs every other decision in the codebase.

|  | Evidence layer | Interpretation layer |
|---|---|---|
| **Contents** | transcript spans, keyframe captions, OCR text, expression signals, embeddings | intents, sentiment, complaints, clusters |
| **Cost** | GPU-hours | minutes of LLM over cached evidence |
| **Mutability** | append-only, immutable | versioned, freely recomputable |
| **Keyed by** | `model_id` | `prompt_version` + `model_id` |

Extraction is expensive and permanent. Interpretation is cheap and changes constantly —
the complaint taxonomy gets revised, the sentiment prompt improves, analysts ask
questions nobody anticipated. Keeping them apart is what makes this possible:

```bash
vl reinterpret --prompt-version v4   # rebuilds every conclusion, zero GPU extraction
```

If both lived in one table, every prompt tweak would mean re-running the GPU over the
entire corpus.

---

## Invariants

Seventeen rules, transcribed from `CLAUDE.md`. **Violating any of these is a bug even if
tests pass.** They are the acceptance criteria that apply to every task in every slice,
and any task that cannot satisfy them needs the spec changed first, not the invariant
bent.

| # | Invariant | Why it exists |
|---|---|---|
| 1 | Evidence is immutable; interpretation is versioned. Never write LLM-derived output into evidence tables. | If a prompt change would require re-running the GPU, the split is broken. |
| 2 | Every stage handler is a pure function of `(video_id, input artifacts) → evidence`, idempotent under key `(video_id, stage, model_id)`. | Keeps the control plane swappable for Temporal or Ray later. |
| 3 | Nothing reads the source video after `normalize`. Downstream stages read only derived artifacts (WAV, keyframe JPEGs). | Reaching back to the source breaks per-stage pools. |
| 4 | All timestamps are normalized-media seconds, never source-container time. | Container start offsets and VFR cause 200–500 ms A/V drift, invisible without the dedicated drift test. |
| 5 | The fan-in barrier waits on terminal states, not success: `done \| failed_permanent \| timed_out`. | No stage may block a video indefinitely. |
| 6 | A missing modality is not a neutral one. Degraded fusion prompts must state explicitly what is absent. | Otherwise the model reads silence as agreement and returns a confident wrong answer. |
| 7 | Quotes are validated verbatim against evidence before being persisted or returned — in fusion and in `ask` alike. | A citation that isn't in the video destroys analyst trust faster than a missed complaint. |
| 8 | Cap VLM frames by area (`max_pixels`), never by longest side. | Aspect ratios run 9:16 to 2.35:1; area is what costs money, and a longest-side cap gives a 2.35× cost spread. |
| 9 | Invariant prompt text goes before images. | Otherwise prefix caching does nothing across a million videos. |
| 10 | Guided decoding is mandatory for all structured output. | Unconstrained JSON fails on a fraction of videos, silently and non-uniformly. |
| 11 | No ORM. psycopg3 and hand-written SQL; `COPY` for bulk span writes. | The hot path is `SKIP LOCKED` claims, partitions and pgvector — all handled badly by ORMs. |
| 12 | Weights are never fetched at runtime. Sync from object storage by `model_id`. Quantization happens only in `vl models prepare`. | Reproducibility and cold-start time. |
| 13 | Never add CUDA to `vl-cpu`. | Per-pool images exist specifically to keep ONNX Runtime and torch out of the same process. |
| 14 | Always build `--platform linux/amd64`. | Dev machines are arm64; ECS is not. |
| 15 | Do not quantize the embedding model or reranker. | They set retrieval quality; quantization reorders nearest neighbours and the degradation is silent, looking like a prompt bug. |
| 16 | SIGTERM releases leases rather than draining. | Spot gives two minutes; a 200-video batch won't finish. |
| 17 | Tests use real Postgres, never SQLite. | `SKIP LOCKED`, partitioning and pgvector are the things under test. |

---

## Tech Stack

**Python 3.12**, managed with `uv`, task-run with `mise`. Packaged as `vl` (hatchling,
`src/` layout), exposing a single `vl` console script.

### Base dependencies

Must stay installable on macOS arm64 so the test suite runs locally. Anything heavy or
platform-bound belongs in an extra.

| Package | Constraint | Role |
|---|---|---|
| `typer` | `>=0.15` | CLI |
| `pydantic` / `pydantic-settings` | `>=2.10` / `>=2.7` | config validation |
| `omegaconf` | `>=2.3` | config composition |
| `psycopg[binary,pool]` | `>=3.2` | Postgres — no ORM (invariant 11) |
| `boto3` | `>=1.35` | S3 / MinIO artifact store |
| `httpx` | `>=0.28` | model service clients |
| `structlog` | `>=24.4` | structured logging |

### Extras, by image

| Extra | Packages | Image |
|---|---|---|
| `cpu` | `opencv-python-headless`, `scenedetect>=0.6.4`, `imagehash`, `pillow`, `mediapipe`, `onnxruntime` | `vl-cpu` |
| `gpu` | `faster-whisper>=1.1`, `onnxruntime-gpu`, `rapidocr-onnxruntime`, `torch>=2.5`, `sentence-transformers`, `pillow` | `vl-gpu` |
| `compress` | `llmcompressor>=0.3`, `transformers`, `huggingface-hub` | model prep only |

### Dev group

`pytest>=8.3`, `pytest-cov`, `ruff>=0.8`, `testcontainers[postgres]>=4.9`, `pyyaml`,
`docker`.

### Models, by stage

| Stage | Model / tool | Pool |
|---|---|---|
| normalize, keyframes | ffmpeg (CFR), PySceneDetect, dHash | `cpu` |
| expression recognition | MediaPipe + HSEmotion, ONNX Runtime CPU | `cpu` |
| speech | faster-whisper large-v3, `int8_float16` | `gpu-small` |
| on-screen text | RapidOCR (PP-OCR weights on ONNX Runtime) | `gpu-small` |
| bulk embeddings | Qwen3-Embedding-0.6B | `gpu-small` |
| keyframe captions | Qwen2.5-VL-7B via vLLM, FP8 | `gpu-large` |
| fusion + ask | Qwen3-8B FP8, guided decoding | `llm-serve` |
| query embed + rerank | Qwen3-Embedding + bge-reranker via Infinity | `embed-serve` |

Model identifiers encode `name@revision+scheme`, so re-quantizing produces distinct
provenance and fp16/FP8 evidence for the same video can be compared side by side.
`vl models prepare` fetches from Hugging Face at a pinned revision, quantizes with
llm-compressor (or CTranslate2 for Whisper), gates on an eval set, and publishes to
object storage.

### Infrastructure

| | Local | AWS |
|---|---|---|
| Database | Postgres container (`pgvector/pgvector:pg16`) | RDS Postgres + pgvector |
| Object storage | MinIO container | S3 |
| Workers | Compose services | ECS Services, scaled on queue depth |
| GPU | one box | EC2 capacity providers on Spot |

Two AWS constraints shape deployment: **Fargate has no GPU support**, so every GPU pool
needs EC2 capacity providers; and **GPU extraction runs on Spot**, which is safe because
extraction is idempotent with lease-based retry, and which is the single largest cost
lever in the project.

---

## Commands

### Development

```bash
mise run install          # uv sync
mise run fmt              # uv run ruff format . && uv run ruff check --fix .
mise run check            # ruff format --check . && ruff check . && pytest
mise run build            # all five images, linux/amd64
mise run up               # docker compose up -d  (postgres, minio, cpu worker)
mise run up:gpu           # docker compose --profile gpu up -d  (Linux + NVIDIA only)
mise run down             # docker compose down
```

Individual image builds: `mise run build:base`, `build:cpu`, `build:gpu`, `build:vllm`,
`build:embed`. `build:cpu` depends on `build:base`; `build:gpu` is self-contained.

### Application

```bash
vl models prepare [--only NAME]        # fetch → quantize → eval gate → publish to S3
vl ingest <path>                       # dedup by content hash + phash, enqueue
vl sweep --stage <stage> [--batch N] [--dry-run]
vl repair                              # retry stages missing on degraded videos
vl reinterpret --prompt-version <v>    # rebuild conclusions, no GPU extraction
vl ask "<question>"                    # hybrid retrieval → answer with citations
vl report [--out PATH]                 # clustered complaints ranked by video count
vl stats [--cost]                      # $/video per stage, queue depth, GPU seconds
vl doctor                              # real-model smoke test (GPU box only)
```

A full extraction pass, one stage at a time:

```bash
vl sweep --stage normalize
vl sweep --stage asr
vl sweep --stage keyframes
vl sweep --stage ocr
vl sweep --stage vlm
vl sweep --stage fer
vl sweep --stage fuse
```

**Sweeps are stage-batched, not per-video.** Model load is 30–60 s, so a worker claims a
batch of ~200 videos, loads the model once, processes all of them, and unloads. This is
why the queue is partitioned by stage.

---

## Project Structure

```
src/vl/
├── cli.py              Typer entrypoints
├── config.py           OmegaConf + Pydantic settings
├── domain/             pure types — Span, Timeline, Interpretation, Finding; no I/O
├── io/
│   ├── db/             psycopg3 query modules + migrations
│   ├── objects.py      S3/MinIO artifact store
│   └── sources/        VideoSource protocol, LocalFilesSource
├── stages/             one module per stage; each a pure function
├── timeline.py         spine construction and interleave
├── models/             prepare, quantize, model_id registry
├── retrieval/          hybrid search, RRF, rerank, citation validation
├── cluster/            incremental clustering
├── workers/            sweep loop, lease/claim, SIGTERM, repair sweeper
└── report/
docker/                 one Dockerfile per pool + entrypoint-serve.sh, sync_weights.py
migrations/             numbered SQL (0001_queue.sql, …)
models.yaml             pinned HF revisions + quantization recipes
tests/                  flat; test_<module>.py, shared fixtures in conftest.py
docs/specs/             this document and its successors
tasks/                  plan.md and todo.md, produced by task breakdown
```

**Currently present:** `cli.py`, `config.py`, `domain/{stages,jobs,barrier}.py`,
`io/db/{queue,migrate}.py`, `workers/shutdown.py`, `migrations/0001_queue.sql`, and the
five Dockerfiles. Everything else in the tree above is a target, created by the build
order below.

### Pools

Split by **resource profile, not convenience**. Do not move a stage between pools
without re-checking the cost table.

| Pool | Stages | Notes |
|---|---|---|
| `cpu` | normalize, keyframes, fer, ingest, sweepers, report | no CUDA, ~600 MB image |
| `gpu-small` | asr, ocr, embed-bulk | CTranslate2 + ORT-GPU + torch |
| `gpu-large` | vlm | vLLM only; ~55% of all GPU time |
| `llm-serve` | fusion, ask | shared; `ask` preempts fusion via priority scheduling |
| `embed-serve` | query embed, rerank | Infinity; vLLM is one model per process |

`cpu`, `gpu-small`, `gpu-large` and `llm-serve` are queue pools with a `Pool` enum
member and claimable `Stage` work. `embed-serve` is a **serving process, not a queue
pool** — query embedding and reranking are synchronous calls made during `ask`, not
leased jobs, which is why no `Stage` maps to it.

### Stage graph

Declared once in `src/vl/domain/stages.py`, because three things must agree about it:
the jobs table, the fan-in barrier, and the worker images.

```
normalize ──┬─→ asr ──┬─→ keyframes ──┬─→ ocr ──┐
            │         │               ├─→ vlm ──┤
            │         │               └─→ fer ──┤
            └─────────┘                         │
                                    asr ────────┴─→ fuse
```

Dependencies are **soft**: a dependency reaching `failed_permanent` still unblocks its
dependents, which then run degraded (invariant 5). Keyframe selection prefers frames
inside speech spans, so it waits for ASR — but falls back to cuts + phash alone if ASR
failed.

- `FUSE_REQUIRES_ANY_OF = (asr, ocr)` — any text signal at all is the price of
  admission. Covers audio-only reviews (ASR only) and silent screen recordings (OCR
  only). Losing both means dead-letter; never a fused record built on nothing.
- `FUSE_OPTIONAL = (keyframes, vlm, fer)` — may be missing; the record is fused and
  flagged degraded.

### Alignment

Every extractor emits `(t_start, t_end, modality, payload)` in **normalized-media
seconds** (invariant 4). `normalize` forces constant frame rate and strips the container
start offset; everything reports against that clock.

Modalities have incompatible granularity: speech gives intervals, captions and OCR give
points. The timeline spine is **speech segments, with synthetic `[silence]` segments
filling gaps**, so visual events during silent product shots aren't orphaned. Fusion
sees a flat, time-ordered interleave:

```
[00:12–00:18] SPEECH  "so the battery just dies by noon"
[00:14]       SCREEN  "Battery 12%"
[00:15]       VISUAL  hands holding phone, low-battery warning dialog
[00:16]       FACE    frustrated 0.61  (low confidence)
[00:18–00:23] SILENCE
[00:20]       VISUAL  close-up of charging port, visible lint
```

That interleave is what lets the model catch a positive-sounding transcript over footage
of a broken product.

### Partial failure

Three rules keep differing stage completion times from stalling the corpus:

- **The barrier waits on terminal states, not success.** Every stage reaches
  `done | failed_permanent | timed_out` in bounded time, so fusion is always reachable.
- **A short grace period, then a provisional record.** Once there's any text signal, a
  5-minute timer waits for optional stages; on expiry the video fuses with whatever
  exists and is flagged `needs_repair` with a list of what's missing. A repair sweeper
  retries on backoff `(5 min, 30 min, 4 h, 1 day, 1 day)`, then abandons and marks the
  record permanently degraded — still counted, never silently dropped.
- **Late repair is free.** `evidence_generation` increments on any new evidence write,
  and a sweeper re-interprets videos whose evidence has moved ahead of their
  interpretation. A captioning stage that succeeds hours later upgrades that video
  automatically, with no GPU re-extraction.

---

## Code Style

Ruff, line length 100, rules `E, F, I, UP, B, SIM`. `ruff format` is authoritative;
`mise run fmt` before committing. This excerpt from `src/vl/domain/barrier.py` is the
reference for what good output looks like here:

```python
"""Fan-in decisions, as pure functions of state.

No clock, no database, no I/O — a caller reads stage states and a grace flag from
Postgres and asks these functions what to do. Keeping it pure is what makes the
awkward cases (everything failed; text arrived but captions didn't; the grace period
expired mid-flight) cheap to enumerate in tests rather than reproduce in staging.

The governing rule: **terminal, not successful.** A dependency that failed
permanently unblocks its dependents, which then run degraded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from vl.domain.jobs import StageState
from vl.domain.stages import FUSE_REQUIRES_ANY_OF, EXTRACTION_STAGES, Stage


@dataclass(frozen=True, slots=True)
class FuseDecision:
    action: FuseAction
    missing: tuple[Stage, ...] = ()
    """Extraction stages that did not succeed. Drives the degraded prompt and repair."""

    @property
    def degraded(self) -> bool:
        return bool(self.missing)


def fuse_decision(
    stage_states: dict[Stage, StageState],
    *,
    grace_expired: bool = False,
) -> FuseDecision:
    """Decide whether a video is ready to interpret.

    `grace_expired` is passed in rather than computed so this stays clock-free. The
    caller sets a deadline when the text requirement is first satisfied, and the
    sweeper reports whether it has passed.
    """
    text_stages = [stage_states[s] for s in FUSE_REQUIRES_ANY_OF]

    # Any text signal at all — spoken or on-screen — is the price of admission. This
    # covers audio-only reviews and silent screen recordings alike.
    if not any(state.succeeded for state in text_stages):
        if all(state.is_terminal for state in text_stages):
            return FuseDecision(FuseAction.DEAD_LETTER)
        return FuseDecision(FuseAction.WAIT)

    unsettled = [s for s in EXTRACTION_STAGES if not stage_states[s].is_terminal]
    if unsettled and not grace_expired:
        return FuseDecision(FuseAction.WAIT)

    # Anything that didn't succeed is missing, whether it failed or simply ran out of
    # grace. The distinction matters for repair, not for the prompt: either way the
    # fusion prompt must state the evidence is absent (CLAUDE.md #6).
    missing = tuple(s for s in EXTRACTION_STAGES if not stage_states[s].succeeded)
    return FuseDecision(FuseAction.FUSE, missing=missing)
```

Conventions demonstrated above, all of which are expected in new code:

- **`from __future__ import annotations`** at the top of every module.
- **Module docstrings explain *why*, not what.** They state the constraint the module
  exists to satisfy and what breaks without it.
- **Comments cite the invariant they enforce** (`CLAUDE.md #6`, `invariant 5`) so a
  future reader knows the line is load-bearing rather than incidental.
- **Pure domain logic takes state as arguments** — no clock, no database, no I/O in
  `domain/`. Time-dependence enters as a boolean the caller computes.
- **`@dataclass(frozen=True, slots=True)`** for value types; `StrEnum` for identifiers
  that cross the database boundary; `Enum, auto()` for internal ones.
- **Attribute docstrings** on non-obvious fields and module constants, rather than
  trailing comments.
- **Keyword-only flags** (`*, grace_expired: bool = False`) so call sites read.
- **Full type annotations**, including `tuple[Stage, ...]` and `int | None` returns.
- **Narrow `noqa` with a reason** where a lint rule genuinely doesn't apply
  (`# noqa: BLE001 - any docker failure means skip, not fail`).

SQL is hand-written in `io/db/` modules (invariant 11); migrations are numbered files in
`migrations/`.

---

## Testing Strategy

**pytest**, configured in `pyproject.toml`: `testpaths = ["tests"]`,
`addopts = "-q --strict-markers"`. Tests live flat in `tests/` as `test_<module>.py`,
with shared fixtures in `tests/conftest.py`.

### Markers

| Marker | Meaning |
|---|---|
| `integration` | needs Docker for a real Postgres; skips cleanly when unavailable |
| `gpu` | requires an NVIDIA GPU — never runs in CI or on macOS |

`--strict-markers` means an unregistered marker is an error, not a silent no-op.

### Database tests

Real Postgres in a container — `pgvector/pgvector:pg16`, via `testcontainers`
(invariant 17). **SQLite cannot stand in**: `FOR UPDATE SKIP LOCKED`, declarative
partitioning and pgvector are the things under test, and none of them exist there.
pgvector rather than plain postgres because later slices index embeddings in this
database, and discovering the extension is missing then is a worse time to find out.

Fixtures:

- `postgres_dsn` (session) — throwaway container; skips cleanly when Docker is down.
- `conn` — connection to a freshly migrated database, truncated between tests rather
  than recreated, which keeps the suite quick while still isolating each test.
- `second_conn` — an independent connection, needed to test real lock contention. Two
  workers claiming from one pool is the behaviour that matters, and a single connection
  cannot exercise `SKIP LOCKED` at all. It sets `lock_timeout=5000` so that a regression
  dropping `SKIP LOCKED` for a bare `FOR UPDATE` fails in seconds with a clear error
  instead of hanging the suite.

### Test levels

| Level | Target | Runs where |
|---|---|---|
| Pure unit | `domain/` — barrier decisions, stage graph, backoff, timeline construction | anywhere, no Docker |
| Integration | `io/db/` — batch claim, lease expiry, heartbeat, dead-letter, barrier transitions | Docker required |
| Container | image invariants — non-root, no CUDA in `vl-cpu`, imports resolve | Docker required |
| GPU smoke | `vl doctor` — real models end to end | GPU box only, never CI |

### Tests that exist because a specific failure is invisible without them

These are not optional coverage; each one guards an invariant whose violation is silent.

- **A/V drift test** — asserts audio and visual timestamps agree on normalized-media
  seconds for a source with a non-zero container offset and variable frame rate
  (invariant 4). Without it, a 200–500 ms drift breaks the exact correlation the project
  depends on and nothing else fails.
- **Lock-contention test** — two connections, concurrent batch claim, no overlap and no
  blocking (invariant 17).
- **No-CUDA-in-`vl-cpu` test** — invariant 13.
- **Verbatim-quote validation tests** — a fabricated quote must be rejected in the
  fusion path and in the `ask` path alike (invariant 7).
- **Degraded-prompt test** — a fusion prompt built with a missing modality must name
  what is absent (invariant 6).
- **Idempotency test** — re-running a stage handler under the same
  `(video_id, stage, model_id)` produces no duplicate evidence (invariant 2).

---

## Boundaries

### Always

- Run `mise run check` before committing.
- Build images with `--platform linux/amd64` (invariant 14).
- Keep `domain/` free of I/O — pure functions of state, no clock, no database.
- Write evidence append-only; put every conclusion in `interpretations` / `findings`
  (invariant 1).
- Emit timestamps in normalized-media seconds (invariant 4).
- Validate quotes verbatim against evidence before persisting or returning them
  (invariant 7).
- State missing modalities explicitly in degraded fusion prompts (invariant 6).
- Use guided decoding for all structured LLM output (invariant 10).
- Count degraded records in aggregates and clustering, and surface `degraded_pct` per
  cluster.
- Cite the invariant in a comment when writing a line that exists to satisfy one.

### Ask first

- Adding a dependency to the base `dependencies` list — it must stay installable on
  macOS arm64.
- Moving a stage between pools — the cost table has to be re-checked first.
- Changing the fusion prompt structure — prefix caching depends on invariant text
  preceding images (invariant 9).
- Schema changes beyond an additive numbered migration.
- Changing quantization schemes or pinned model revisions in `models.yaml`.
- Re-opening any deferred decision — the trigger table below is the gate.
- Changing keyframe budget or `max_pixels` — both are direct `$/video` levers.

### Never

- Write LLM-derived output into evidence tables (invariant 1).
- Read the source video after `normalize` (invariant 3).
- Add CUDA to `vl-cpu` (invariant 13).
- Introduce an ORM (invariant 11).
- Fetch weights from Hugging Face at runtime (invariant 12).
- Quantize the embedding model or reranker (invariant 15).
- Substitute SQLite for Postgres in tests (invariant 17).
- Drain in-flight work on SIGTERM instead of releasing leases (invariant 16).
- Let a stage block a video indefinitely (invariant 5).
- Silently drop a degraded record from aggregates.

---

## Build Order

Thin vertical slices. **Each must work end to end before the next begins.** This is the
decomposition that task breakdown slices against.

| # | Slice | Status |
|---|---|---|
| 0 | Docs + five images + compose | ✅ partial — see below |
| 1 | Skeleton: queue, artifact store, sources, repair sweeper, `models prepare` | 🟡 in progress |
| 2 | Audio only, stored: normalize → asr → spans → embeddings → spine → fusion → interpretations | ⬜ |
| 3 | `vl ask` on audio-only evidence | ⬜ |
| 4 | Keyframe detector + OCR | ⬜ |
| 5 | VLM captions | ⬜ |
| 6 | FER, with `fer_frames_pct` surfaced | ⬜ |
| 7 | Incremental clustering + `vl report` | ⬜ |
| 8 | `vl reinterpret` | ⬜ |
| 9 | `vl stats --cost` | ⬜ |

**Slice 0.** `vl-base` (128 MB) and `vl-cpu` (569 MB) build and run, verified on
linux/amd64: non-root, no CUDA, ffmpeg and all cpu extras import. **Still unverified:**
the `gpu`, `vllm` and `embed` images — they need an NVIDIA host, and the `vllm` and
`infinity` base tags are pinned from memory and must be checked against their
registries.

**Slice 1.** Done: migrations, jobs table (batch claim, lease, heartbeat, dead-letter),
`video_stages` barrier, fan-out, SIGTERM lease release. Remaining: artifact store,
`VideoSource` protocol + `LocalFilesSource`, grace timer + repair sweeper (needs the
fuse stage), `vl models prepare` (needs a GPU to verify).

Rationale for the ordering worth preserving: slice 3 lands `ask` early, *while evidence
is still cheap*. Slice 4 exists so the keyframe detector's picks can be eyeballed before
incurring VLM cost. Slice 5 is the first real `$/video` measurement. Slice 8 is what
proves the two-layer split is real rather than asserted.

---

## Key Numbers

Targets and budgets, not aspirations — each one is a check against a specific decision.

- **VLM is ~55% of GPU time** — the only stage worth optimizing hard. ~11,000 GPU-hours
  per million videos.
- **~18 s per video is pure CPU** (ffmpeg + scene detection). Never run it on GPU nodes.
- **Keyframe budget: ~40 frames** selected from ~5,400 (a 3-minute video at 30 fps), at
  **~256 visual tokens each**. Scene-cut detection picks candidates, perceptual-hash
  dedup drops near-identical frames, and a budget cap keeps the ones that are visually
  distinct and fall under narration.
- **Batch size ~200 videos** per sweep claim; model load is 30–60 s.
- **Grace period: 5 minutes** before a provisional degraded record is written.
- **Repair backoff:** 5 min → 30 min → 4 h → 1 day → 1 day, then abandon.
- **Measure `$/video` on ≥1,000 videos** before extrapolating to a million.

---

## Success Criteria

Specific and testable. The system is done when all of these hold.

### Correctness

1. `mise run check` passes: format, lint, and the full pytest suite.
2. Every stage handler is idempotent — re-running under the same
   `(video_id, stage, model_id)` produces no duplicate evidence.
3. The A/V drift test passes on a source with a non-zero container start offset and
   variable frame rate.
4. No video ever occupies a non-terminal state indefinitely; the barrier test covers
   all-failed, text-only, and grace-expired-mid-flight cases.
5. `vl reinterpret --prompt-version v4` rebuilds every conclusion in the corpus with
   **zero GPU extraction jobs enqueued** — asserted by the job counter, not by
   inspection.

### Analyst-facing

6. `vl ask "<question>"` returns an answer where **every** quoted span appears verbatim
   in the evidence tables, and every citation resolves to a `video_id @ timestamp` that
   plays back to the claimed moment.
7. The verbatim-validation **drop rate is logged as a first-class metric** in both the
   fusion path and the `ask` path — visible in `vl stats`, not buried in logs.
8. `vl report` ranks clustered complaints by video count and surfaces `degraded_pct`
   per cluster.
9. `fer_frames_pct` is present on every video record, so the weakness of expression
   recognition is visible in output rather than hidden in an aggregate.

### The core claim

10. A hand-labelled set of **25–30 videos**, including **5 or more where the footage
    contradicts the words**, exists and is version-controlled.
11. **Text-only accuracy vs. fused accuracy is reported against that set.** The claim
    "visual corrects text" is either demonstrated with a number or withdrawn.
    `sentiment.text_only_label` is stored beside the fused label specifically so this is
    measurable rather than asserted.

### Cost

12. `vl stats --cost` breaks down `$/video` per stage, queue depth and GPU seconds.
13. `$/video` has been measured on **≥1,000 real videos** before any scale-out decision.
14. The keyframe selector holds the ~40-frame budget on videos ranging from 9:16 to
    2.35:1 aspect ratio, capped by area rather than longest side.

---

## Risks

Two, both named in `CLAUDE.md`, both first-class rather than incidental.

### 1. Hallucinated quotes

In both fusion and `ask`. A citation that isn't in the video destroys analyst trust
faster than a missed complaint does.

**Mitigation:** verbatim validation in both paths (invariant 7), with the drop rate as a
logged, surfaced metric — so degradation is observable rather than discovered by an
analyst.

### 2. "Visual corrects text" being unfalsifiable

The system's central claim is that fusing modalities produces better sentiment than text
alone. Without a labelled set this is an assertion that cannot fail, and therefore
cannot be trusted.

**Mitigation:** the hand-labelled set and the text-only-vs-fused comparison in success
criteria 10–11. `sentiment.text_only_label` is stored beside the fused label for exactly
this reason.

### Related design notes

- **Facial expression recognition is deliberately included** despite ~60–65% in-the-wild
  accuracy and firing on only a minority of frames. It ships with `fer_frames_pct` per
  video so its weakness is visible in output rather than hidden in an aggregate.
- **Degraded records are counted** in aggregates and clustering. Excluding them would
  bias every count toward easily-processed videos. Reports surface `degraded_pct` per
  cluster.

---

## Deferred Decisions

Settled as "not now," each with the trigger that reopens it. **Do not relitigate without
the trigger.**

| Deferred | Reopen when |
|---|---|
| SGLang instead of vLLM | fusion prompt is stable and `$/video` is known — RadixAttention targets our shared-prefix shape |
| WhisperX word-level timestamps | analysts complain citations land early (we have 3–8 s segment precision) |
| OCR on the CPU pool | slice-3 benchmark shows CPU keeps pace at ~130 frames/video |
| Temporal or Ray control plane | "where is video X and why did it fail" becomes a recurring question |
| Lakehouse evidence storage | span count passes ~100–300 M and pgvector becomes the bottleneck |
| Speaker diarization | user-research recordings enter scope (`evidence_spans.speaker` is reserved) |
| Arabic support | English pipeline is proven end to end; embeddings are already multilingual |
| W4A16 instead of FP8 | concurrency turns out memory-limited *and* corpus keyframes exist for calibration |

Invariant 2 — pure, idempotent stage handlers — is what keeps the Temporal/Ray row cheap
to act on if its trigger fires.

---

## Open Questions

Discrepancies found between the source documents and the code while writing this spec.
None were resolved here; each needs a decision.

1. **`ty` is specified but not wired.** `CLAUDE.md` describes `mise run check` as
   running "ty, ruff, pytest," but the `check` task in `mise.toml` runs
   `ruff format --check`, `ruff check` and `pytest` only, and `ty` is absent from the
   dev dependency group. Either add the type checker to both, or correct `CLAUDE.md`.
   This spec documents the task as it currently behaves.

2. **`models.yaml` does not exist.** Both `README.md` and invariant 12 depend on it for
   pinned Hugging Face revisions and quantization recipes. It is a slice-1 deliverable
   (`vl models prepare`) that has not landed.

3. **Three images are unverified.** `gpu`, `vllm` and `embed` have never been built on
   an NVIDIA host, and the `vllm` and `infinity` base image tags were pinned from memory
   rather than checked against their registries. Slice 0 is not closed until this is
   done, and any task depending on those images inherits the risk.

4. **`embed-serve` has no `Pool` enum member.** The pools table lists five pools; the
   `Pool` StrEnum has four. This spec reads that as intentional — query embedding and
   reranking are synchronous calls during `ask`, not leased jobs — but the asymmetry
   between the table and the code is worth confirming rather than assuming.

---

## Provenance

Restated from `README.md` and `CLAUDE.md`, cross-checked against `pyproject.toml`,
`mise.toml`, `src/vl/domain/{stages,barrier}.py` and `tests/conftest.py` as of
2026-08-17. `CLAUDE.md` remains the operating contract; where this document and it
diverge, `CLAUDE.md` wins and this document is wrong and should be corrected.
