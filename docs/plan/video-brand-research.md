# Implementation Plan: Video Listening

Task breakdown for [`docs/specs/video-brand-research.md`](../specs/video-brand-research.md).
Plan document and task list are combined in this one file, replacing the default
`tasks/plan.md` + `tasks/todo.md` split.

**44 tasks across 10 phases**, ordered by dependency. Phases map 1:1 onto the spec's
build order.

**Progress: 1 / 44 complete** — A1 ✅

---

## Overview

The spec defines a system that turns raw video into a queryable Voice-of-the-Customer
corpus. This plan sequences it as thin vertical slices, each ending in something
runnable, because the two things most likely to kill the project — `$/video` being
unaffordable, and "visual corrects text" being unfalsifiable — are both only measurable
once a slice runs end to end.

**What already exists:** the queue (batch claim, lease, heartbeat, dead-letter,
`video_stages` barrier, fan-out, SIGTERM lease release), the domain barrier as pure
functions, the full CLI surface as `_pending()` stubs, `Settings` with every tuning knob
already declared, `migrations/0001_queue.sql`, and 70 passing tests. `vl-base` and
`vl-cpu` build and run on linux/amd64.

**What this plan builds:** everything else.

---

## Architecture Decisions

Carried from the spec; recorded here because they shape task boundaries.

- **Tasks never cross the evidence/interpretation line.** A task writes evidence or
  writes interpretation, never both. This is what keeps `reinterpret` (Phase I) a small
  task instead of a rewrite.
- **Stage handlers are pure functions of `(video_id, artifacts) → evidence`.** Every
  stage task therefore splits naturally into a pure handler (unit-testable without
  Docker) plus a thin worker wiring (integration-tested). Task sizes below assume that
  split.
- **`Settings` already holds the knobs.** `vlm_max_pixels`, `max_keyframes`,
  `grace_period_s`, `quote_match_ratio` and the rest exist in `src/vl/config.py`. Tasks
  consume them rather than introducing new configuration.
- **`default_model_resolver` is a placeholder.** It returns `f"{stage}@unset"`. Task B4
  replaces it with the real registry; every task before B4 tolerates the placeholder,
  and none hard-code it.
- **The labelled set is started early (B8), used late (H2).** It is human work with a
  long lead time, and it gates the project's central claim. Starting it at the end means
  discovering the claim is wrong after paying for slices 4–6.
- **Detail decays with distance.** Phases A–D are specified at file level. Phases E–K
  carry acceptance criteria but coarser file lists, because their shape depends on
  measurements from earlier phases — notably the slice-3 OCR benchmark (D5) and the
  first `$/video` reading (F4).

---

## Dependency Graph

```
A: close slice 0 (images verified)
   │
   ├─────────────────────────┐
   ▼                         ▼
B: foundation           B8: labelled set  ── (human work, runs in parallel throughout)
   artifact store            │
   sources → ingest          │
   model registry            │
   sweep loop                │
   │                         │
   ▼                         │
C: audio evidence → fuse     │
   normalize → asr → embed   │
   timeline → fusion         │
   grace timer → repair      │
   │                         │
   ▼                         │
D: vl ask (audio only)       │
   │                         │
   ▼                         │
E: keyframes + OCR           │
   │                         │
   ▼                         │
F: VLM  ── first $/video ────┤
   │                         │
   ▼                         │
G: FER                       │
   │                         │
   ▼                         ▼
H: evaluation — text-only vs fused accuracy
   │
   ├──→ I: clustering + report
   ├──→ J: reinterpret
   └──→ K: stats --cost
```

---

## Phase A — Close Slice 0

The spec lists three unverified images and one documentation discrepancy. All of Phase B
onward assumes these images run; finding out otherwise later invalidates work.

### Task A1: Verify pinned base image tags against registries — ✅ **Done**

**Description:** The `vllm` and `infinity` base tags in `docker/vllm.Dockerfile` and
`docker/embed.Dockerfile` were pinned from memory (spec § Open Questions 3). Check each
against its actual registry and correct or confirm.

**Acceptance criteria:**
- [x] Every `FROM` tag across the five Dockerfiles resolves to a real, currently-published image
- [x] Any corrected tag is pinned to a digest or an immutable version tag, not `latest`
- [x] A comment on each `FROM` records where the tag was verified and when

**Verification:**
- [x] `docker manifest inspect <tag>` succeeds for every base image — now automated as `test_external_base_images_resolve`
- [ ] `mise run build` completes — **deferred to A2**, which is the task that builds the GPU images

**Outcome:** One tag was wrong, and it was **not** one of the two flagged as written from
memory. `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04` has never existed — nvidia
publishes `ubuntu24.04` only from CUDA 12.6.0 onward. Corrected to
`12.6.3-cudnn-runtime-ubuntu24.04`, which preserves the documented reason 24.04 was
chosen (system Python 3.12, no PPA); dropping to `ubuntu22.04` would have kept CUDA
12.4.1 but broken that. `vllm/vllm-openai:v0.11.0` and `michaelf34/infinity:0.0.77` both
turned out to be real. Infinity publishes a **single-arch linux/amd64** manifest, so it
cannot run on an arm64 dev machine at all — noted in the Dockerfile.

Two tests added, so this cannot silently regress: `test_external_base_images_resolve`
(marked `network`, resolves every `FROM` **and** `COPY --from=` pin, asserting a
linux/amd64 variant) and `test_external_pins_record_when_they_were_verified` (static,
requires a dated `# verified` note).

**Dependencies:** None
**Files:** `docker/{base,cpu,gpu,vllm,embed}.Dockerfile`, `tests/test_containers.py`, `pyproject.toml`
**Scope:** S

---

### Task A2: Build and verify the gpu, vllm and embed images

**Description:** Build the three unverified images for linux/amd64 and extend the
existing container test suite (`tests/test_containers.py`, 9 tests) to cover them the way
it already covers `vl-base` and `vl-cpu`.

**Acceptance criteria:**
- [ ] `vl-gpu`, `vl-vllm`, `vl-embed` build cleanly for `linux/amd64` (invariant 14)
- [ ] Each runs as a non-root user
- [ ] `vl-gpu` imports every `gpu` extra; `vl-cpu` still imports no CUDA (invariant 13)
- [ ] Image sizes recorded in the spec's slice-0 note

**Verification:**
- [ ] `mise run build`
- [ ] `uv run pytest tests/test_containers.py`

**Dependencies:** A1
**Files:** `tests/test_containers.py`, `docker/*.Dockerfile`
**Scope:** M

---

### Task A3: GPU-host bring-up of the gpu compose profile

**Description:** On an NVIDIA host, start the GPU profile and confirm `vlm-serve`,
`llm-serve` and `embed-serve` reach a healthy state and answer a trivial request. This is
the first time the serving topology runs at all.

**Acceptance criteria:**
- [ ] `docker compose --profile gpu up -d` brings all five services healthy
- [ ] `vlm-serve` and `llm-serve` answer an OpenAI-compatible `/v1/models` request
- [ ] `embed-serve` returns an embedding for a test string
- [ ] Port assignments match `Settings.vlm_url` / `llm_url` / `embed_url` defaults

**Verification:**
- [ ] `mise run up:gpu` then `docker compose ps` shows all healthy
- [ ] Manual: `curl` each service's health/models endpoint

**Dependencies:** A2
**Files:** `docker-compose.yml` (fixes only)
**Scope:** S — but **requires GPU hardware**; blocks nothing in Phase B except B5.

---

### Task A4: Resolve the `ty` discrepancy

**Description:** `CLAUDE.md` describes `mise run check` as running "ty, ruff, pytest";
`mise.toml` runs ruff and pytest only, and `ty` is not in the dev group (spec § Open
Questions 1). Pick one and make the two agree.

**Acceptance criteria:**
- [ ] Either `ty` is added to the dev group and to the `check` task and passes clean, or `CLAUDE.md` is corrected to describe the task as it is
- [ ] `mise run check` and its documentation describe the same thing

**Verification:**
- [ ] `mise run check`

**Dependencies:** None — parallel with A1–A3
**Files:** `mise.toml`, `pyproject.toml`, `CLAUDE.md`
**Scope:** XS

---

### ✅ Checkpoint A

- [ ] All five images build for linux/amd64 and pass container tests
- [ ] GPU profile verified on real hardware (or A3 explicitly deferred with a named owner)
- [ ] `mise run check` passes and matches its documentation
- [ ] Spec § Open Questions 1 and 3 closed

---

## Phase B — Finish Slice 1 (Foundation)

Everything downstream reads artifacts and identifies models. Both are missing.

### Task B1: Artifact store

**Description:** `io/objects.py` — put, get, exists and presigned-URL access over
S3/MinIO, keyed content-addressably by `(video_id, stage, artifact_name)`. This is what
makes invariant 3 enforceable: downstream stages read derived artifacts, never source
video.

**Acceptance criteria:**
- [ ] Round-trips bytes and file paths against MinIO
- [ ] Key layout is deterministic from `(video_id, stage, name)` — re-running a stage overwrites rather than duplicating (invariant 2)
- [ ] `exists()` lets a handler skip work already done
- [ ] Reads and writes stream rather than buffering whole videos in memory

**Verification:**
- [ ] `uv run pytest tests/test_objects.py` against the compose MinIO
- [ ] Manual: object visible in the MinIO console at the expected key

**Dependencies:** None
**Files:** `src/vl/io/objects.py`, `tests/test_objects.py`
**Scope:** S

---

### Task B2: VideoSource protocol and LocalFilesSource

**Description:** `io/sources/` — a `VideoSource` protocol yielding
`(source_uri, local_path, metadata)`, plus `LocalFilesSource` walking a folder. The
protocol exists now so a platform connector later is an addition rather than a
refactor.

**Acceptance criteria:**
- [ ] `VideoSource` is a `typing.Protocol`, no base-class inheritance required
- [ ] `LocalFilesSource` walks a directory, filters by extension, yields stable ordering
- [ ] Metadata carries what `videos` needs: `source_uri`, optional `platform`, optional `published_at`
- [ ] Unreadable or non-video files are skipped with a structured log line, not a crash

**Verification:**
- [ ] `uv run pytest tests/test_sources.py` — no Docker needed

**Dependencies:** None — parallel with B1
**Files:** `src/vl/io/sources/__init__.py`, `src/vl/io/sources/local.py`, `tests/test_sources.py`
**Scope:** S

---

### Task B3: `vl ingest` — dedup and enqueue

**Description:** Wire `LocalFilesSource` to `register_video()` and `fan_out()`. Content
hash becomes `videos.id`; perceptual hash catches re-encodes and watermarks that content
hashing misses, setting `dedup_of`.

**Acceptance criteria:**
- [ ] Content hash of the file bytes is the `video_id` — same bytes ingested twice produces one row, second call reports "already known"
- [ ] A visually near-identical variant (re-encoded or watermarked) within `phash_hamming_max` sets `dedup_of` and is not enqueued for extraction
- [ ] Successful registration enqueues `normalize` and nothing else (fan-out respects the stage graph)
- [ ] Ingest is resumable: interrupting mid-folder and re-running skips what landed

**Verification:**
- [ ] `uv run pytest tests/test_ingest.py -m integration`
- [ ] Manual: `vl ingest ./videos` twice; second run enqueues zero jobs

**Dependencies:** B1, B2
**Files:** `src/vl/cli.py`, `src/vl/io/db/videos.py`, `tests/test_ingest.py`
**Scope:** M

---

### Task B4: `models.yaml` and the model_id registry

**Description:** Create `models.yaml` (pinned HF revisions + quantization recipes) and
`models/registry.py` resolving `Stage → model_id` in the documented
`name@revision+scheme` form. Replaces `default_model_resolver`'s `f"{stage}@unset"`
placeholder in `io/db/queue.py`.

**Acceptance criteria:**
- [ ] Every stage in `Stage` resolves to a `name@revision+scheme` identifier
- [ ] Identifiers differ between quantization schemes, so fp16 and FP8 evidence for one video coexist as distinct rows (invariant 2)
- [ ] `default_model_resolver` is replaced at all call sites; no `@unset` reachable in normal operation
- [ ] A malformed or missing `models.yaml` fails loudly at startup, not at claim time

**Verification:**
- [ ] `uv run pytest tests/test_registry.py tests/test_queue.py`
- [ ] Manual: `vl sweep --stage asr --dry-run` prints a real model_id

**Dependencies:** None — parallel with B1–B3
**Files:** `models.yaml`, `src/vl/models/registry.py`, `src/vl/io/db/queue.py`, `tests/test_registry.py`
**Scope:** M — closes spec § Open Questions 2

---

### Task B5: `vl models prepare`

**Description:** Fetch from Hugging Face at the pinned revision, quantize with
llm-compressor (CTranslate2 for Whisper), gate on an eval set, publish to object storage
by `model_id`. The only place quantization happens (invariant 12).

**Acceptance criteria:**
- [ ] Fetches at the pinned revision from `models.yaml`, never `main`
- [ ] Quantizes only what the recipe names — the embedding model and reranker are **skipped** (invariant 15)
- [ ] Publishing is gated on an eval threshold; a failing model is not published
- [ ] Published artifacts are addressed by `model_id`, and containers sync by that key with no runtime HF access
- [ ] `--only NAME` prepares a single model

**Verification:**
- [ ] `uv run pytest tests/test_models_prepare.py` (recipe parsing, skip-list, gate logic — no GPU)
- [ ] Manual on GPU host: `vl models prepare --only whisper` publishes to MinIO

**Dependencies:** B4, A3 (GPU host for the real run)
**Files:** `src/vl/models/prepare.py`, `src/vl/models/quantize.py`, `src/vl/cli.py`, `tests/test_models_prepare.py`
**Scope:** M — **highest-uncertainty task in Phase B**; needs hardware to fully verify.

---

### Task B6: Sweep loop worker

**Description:** `workers/sweep.py` — claim a batch, load the model once, process each
video, complete or fail, heartbeat throughout, release on SIGTERM. Uses the existing
`ShutdownGuard` and `process_batch` in `workers/shutdown.py` and the existing queue
functions; this task is wiring, not new queue logic.

**Acceptance criteria:**
- [ ] Model loads once per batch, not once per video (the whole reason the queue is stage-partitioned)
- [ ] Heartbeat extends leases for work still in flight; a stalled batch does not lose its lease
- [ ] SIGTERM releases leases without consuming a retry attempt (invariant 16), verified by attempts count before and after
- [ ] A single video's failure fails that job only; the rest of the batch completes
- [ ] Handler is injected, so a stage's pure function is testable without the loop

**Verification:**
- [ ] `uv run pytest tests/test_sweep.py -m integration`
- [ ] Manual: start a sweep, `docker kill -s TERM`, confirm jobs return to `ready` with attempts unchanged

**Dependencies:** B4
**Files:** `src/vl/workers/sweep.py`, `src/vl/cli.py`, `tests/test_sweep.py`
**Scope:** M

---

### Task B7: `vl doctor`

**Description:** Real-model smoke test on the GPU box: each service reachable, each model
loads, one trivial inference per stage. Never runs in CI.

**Acceptance criteria:**
- [ ] Checks Postgres, MinIO, and all three model services, reporting each independently
- [ ] Marked `gpu`; skipped cleanly on macOS and in CI
- [ ] Exits non-zero with a specific, actionable message naming what failed

**Verification:**
- [ ] `uv run pytest -m "not gpu"` — doctor's tests skip
- [ ] Manual on GPU host: `vl doctor` passes

**Dependencies:** B5, A3
**Files:** `src/vl/cli.py`, `src/vl/doctor.py`, `tests/test_doctor.py`
**Scope:** S

---

### Task B8: Start the hand-labelled evaluation set ⟂

**Description:** Begin assembling the 25–30 video labelled set required by spec success
criteria 10–11, **including 5 or more where the footage contradicts the words**. Human
work with a long lead time — start now, use in Phase H.

**Acceptance criteria:**
- [ ] 25–30 videos collected, version-controlled by reference (URI + content hash, not the bytes)
- [ ] At least 5 carry footage that contradicts the spoken sentiment
- [ ] Each has a ground-truth intent label and sentiment label recorded in a fixed schema
- [ ] Labelling instructions written down, so a second labeller produces comparable output

**Verification:**
- [ ] Manual review: contradiction cases actually contradict
- [ ] `uv run pytest tests/test_eval_set.py` validates schema and counts

**Dependencies:** None — **parallel with everything**, and the earlier it starts the better
**Files:** `eval/labelled_set.yaml`, `eval/README.md`, `tests/test_eval_set.py`
**Scope:** S in code, large in human hours

---

### ✅ Checkpoint B

- [ ] `vl ingest ./videos` registers, dedups and enqueues against real Postgres + MinIO
- [ ] Every stage resolves a real `model_id`; no `@unset` in the database
- [ ] A sweep claims a batch, survives SIGTERM without burning retry attempts
- [ ] `mise run check` passes
- [ ] Labelled set underway

---

## Phase C — Slice 2: Audio Only, Stored

The first vertical slice that produces an interpretation. Everything here is on the
critical path.

### Task C1: Migration 0002 — evidence and interpretation tables

**Description:** `evidence_spans` (partitioned, with `speaker` reserved for the deferred
diarization row), `span_embeddings` (pgvector), `interpretations` and `findings`. Two
physically separate groups of tables, because invariant 1 is enforced by schema, not by
discipline.

**Acceptance criteria:**
- [ ] `evidence_spans` carries `(video_id, t_start, t_end, modality, payload, model_id)` and is partitioned
- [ ] `evidence_spans.speaker` exists and is nullable (reserved for deferred diarization)
- [ ] `interpretations` is keyed by `(video_id, prompt_version, model_id)` and holds `sentiment.text_only_label` beside the fused label
- [ ] No LLM-derived column exists on any evidence table (invariant 1)
- [ ] pgvector index present on `span_embeddings`
- [ ] Migration is additive and re-runnable

**Verification:**
- [ ] `uv run pytest tests/test_migrations.py -m integration`
- [ ] Manual: `\d+ evidence_spans` shows expected partitioning

**Dependencies:** Phase B
**Files:** `migrations/0002_evidence.sql`, `src/vl/io/db/spans.py`, `tests/test_migrations.py`
**Scope:** M

---

### Task C2: `normalize` stage

**Description:** ffmpeg to constant frame rate, strip container start offset, extract
WAV, write both to the artifact store, record `duration_s` and `fps` on `videos`. **This
task establishes the clock every other timestamp in the system is measured against.**

**Acceptance criteria:**
- [ ] Output is CFR at a recorded `fps`, with the container start offset removed
- [ ] WAV and normalized video land in the artifact store under `(video_id, normalize, …)`
- [ ] `videos.duration_s` and `videos.fps` populated
- [ ] Re-running produces identical artifacts and no duplicate rows (invariant 2)
- [ ] Runs on the `cpu` pool with no CUDA import reachable

**Verification:**
- [ ] `uv run pytest tests/test_normalize.py`
- [ ] Manual: `ffprobe` the output confirms CFR and zero start offset

**Dependencies:** C1
**Files:** `src/vl/stages/normalize.py`, `tests/test_normalize.py`
**Scope:** M

---

### Task C3: A/V drift test ⚠

**Description:** The dedicated drift test from the spec's testing strategy. Build a
fixture with a **non-zero container start offset and variable frame rate**, then assert
audio and visual timestamps agree in normalized-media seconds.

**Acceptance criteria:**
- [ ] Fixture genuinely has a non-zero start offset and VFR — verified with `ffprobe`, not assumed
- [ ] Test asserts audio/visual agreement within a stated tolerance well under 200 ms
- [ ] Test fails if `normalize` is changed to pass through source timestamps

**Verification:**
- [ ] `uv run pytest tests/test_drift.py`
- [ ] Deliberately break `normalize` to pass through source time; confirm the test fails

**Dependencies:** C2
**Files:** `tests/test_drift.py`, `tests/fixtures/`
**Scope:** S — **do not defer.** Drift is invisible without this test (invariant 4).

---

### Task C4: `asr` stage

**Description:** faster-whisper large-v3 at `int8_float16` over the normalized WAV,
emitting `(t_start, t_end, SPEECH, text)` spans written with `COPY` (invariant 11).

**Acceptance criteria:**
- [ ] Spans carry normalized-media seconds, not whisper's source-relative output
- [ ] Bulk write uses `COPY`, not row-by-row inserts
- [ ] Detected language recorded on `videos.language`
- [ ] `spans_written` reported back through `complete()` so the barrier row reflects it
- [ ] Segment-level precision is 3–8 s and that limitation is documented at the call site (word-level is deferred)

**Verification:**
- [ ] `uv run pytest tests/test_asr.py -m integration`
- [ ] Manual on GPU host: `vl sweep --stage asr` over 10 videos

**Dependencies:** C2, C3
**Files:** `src/vl/stages/asr.py`, `src/vl/io/db/spans.py`, `tests/test_asr.py`
**Scope:** M

---

### Task C5: `embed-bulk` stage

**Description:** Qwen3-Embedding-0.6B over speech spans into `span_embeddings`.
**Unquantized** (invariant 15).

**Acceptance criteria:**
- [ ] Embeddings written with `COPY` in batches
- [ ] Model is the fp16 original — a quantized embedding model is rejected at load with a clear error
- [ ] Dimensionality asserted against the pgvector column at startup, not at insert
- [ ] Idempotent under `(video_id, stage, model_id)`

**Verification:**
- [ ] `uv run pytest tests/test_embed.py -m integration`
- [ ] Manual: nearest-neighbour query over a small corpus returns sane neighbours

**Dependencies:** C4
**Files:** `src/vl/stages/embed.py`, `tests/test_embed.py`
**Scope:** S

---

### Task C6: Timeline spine and interleave

**Description:** `timeline.py` — speech segments as the spine, synthetic `[silence]`
segments filling gaps longer than `silence_gap_s`, point events attached within
`point_attach_tolerance_s`. Pure functions, no I/O.

**Acceptance criteria:**
- [ ] Gaps longer than `silence_gap_s` become explicit `[silence]` segments
- [ ] Point events (OCR, captions, FER) attach to the containing or nearest span within tolerance; nothing is orphaned
- [ ] Output is flat and time-ordered, matching the interleave format in the spec
- [ ] A video with speech only produces a valid timeline (the Phase C case)
- [ ] Zero I/O — testable without Docker

**Verification:**
- [ ] `uv run pytest tests/test_timeline.py`

**Dependencies:** C4
**Files:** `src/vl/timeline.py`, `tests/test_timeline.py`
**Scope:** M

---

### Task C7: Fusion prompt builder

**Description:** Build the fusion prompt from a timeline, with invariant text **before**
any images (invariant 9) and an explicit statement of every missing modality
(invariant 6). Consumes `FuseDecision.missing` from the existing
`domain/barrier.py`.

**Acceptance criteria:**
- [ ] Invariant prefix precedes all variable content, so prefix caching is effective
- [ ] Missing modalities are named explicitly in the prompt text — asserted by a test that builds a degraded prompt and greps for the absent stage
- [ ] A prompt built with **no** missing modalities contains no misleading absence language
- [ ] Prompt version is an explicit parameter, stored with the interpretation

**Verification:**
- [ ] `uv run pytest tests/test_fusion_prompt.py`
- [ ] Manual: eyeball a degraded prompt and a complete one side by side

**Dependencies:** C6
**Files:** `src/vl/stages/fuse.py`, `src/vl/prompts/fusion.py`, `tests/test_fusion_prompt.py`
**Scope:** M

---

### Task C8: Verbatim quote validator ⚠

**Description:** `retrieval/citations.py` — validate every quote against
`evidence_spans` at `quote_match_ratio` before persistence, and log the drop rate as a
first-class metric. Used by fusion now and by `ask` in Phase D (invariant 7).

**Acceptance criteria:**
- [ ] A fabricated quote is rejected; a verbatim quote passes
- [ ] Near-misses are governed by `Settings.quote_match_ratio`, not a hard-coded constant
- [ ] Every rejection increments a counter that surfaces in `vl stats` — not just a log line
- [ ] The same function is the single implementation used by both fusion and `ask`
- [ ] Rejection drops the quote without discarding the surrounding answer

**Verification:**
- [ ] `uv run pytest tests/test_citations.py`
- [ ] Manual: feed a hallucinated quote through; confirm rejection and counter increment

**Dependencies:** C1
**Files:** `src/vl/retrieval/citations.py`, `tests/test_citations.py`
**Scope:** S — retires risk 1 for the fusion path

---

### Task C9: `fuse` stage

**Description:** Call the LLM with guided decoding (invariant 10), validate quotes
(C8), write to `interpretations` — never to evidence tables (invariant 1). Store
`sentiment.text_only_label` beside the fused label.

**Acceptance criteria:**
- [ ] Structured output uses guided decoding; an unconstrained-JSON path does not exist
- [ ] Intent and sentiment both produced, per the spec's two listening jobs
- [ ] `sentiment.text_only_label` stored alongside the fused label (required by Phase H)
- [ ] Every quote passes C8 before persistence
- [ ] Interpretation records the `evidence_generation` it consumed
- [ ] Writes touch `interpretations`/`findings` only — a test asserts no write reaches an evidence table

**Verification:**
- [ ] `uv run pytest tests/test_fuse.py -m integration`
- [ ] Manual on GPU host: `vl sweep --stage fuse` over the audio corpus

**Dependencies:** C7, C8
**Files:** `src/vl/stages/fuse.py`, `tests/test_fuse.py`
**Scope:** M

---

### Task C10: Grace timer and barrier evaluation sweeper

**Description:** Set `grace_expires_at` when the text requirement is first satisfied;
sweep for videos whose barrier says FUSE and enqueue them. `fan_out()` deliberately
excludes `fuse`, so this sweeper is what makes fusion reachable. Uses the existing
`fuse_decision()`.

**Acceptance criteria:**
- [ ] `grace_expires_at` set exactly once, when ASR or OCR first succeeds
- [ ] On expiry with stages unsettled, the video is enqueued for fuse and flagged `needs_repair` with the missing list
- [ ] A video where both ASR and OCR terminally fail is dead-lettered, never fused
- [ ] No video sits non-terminal indefinitely — asserted with a time-advanced test
- [ ] The decision comes from `domain/barrier.fuse_decision`, not reimplemented in SQL

**Verification:**
- [ ] `uv run pytest tests/test_grace.py -m integration`
- [ ] Manual: fail a stage deliberately, confirm fusion still happens after the grace period

**Dependencies:** C9
**Files:** `src/vl/workers/barrier_sweeper.py`, `src/vl/io/db/queue.py`, `tests/test_grace.py`
**Scope:** M

---

### Task C11: Repair sweeper and `vl repair`

**Description:** Retry missing stages on the `(5 min, 30 min, 4 h, 1 day, 1 day)`
backoff using the existing `next_repair_at()`; abandon after exhaustion, keeping the
record permanently degraded. Late success bumps `evidence_generation` and triggers
re-interpretation of that video only.

**Acceptance criteria:**
- [ ] Backoff schedule matches `REPAIR_BACKOFF_S`; abandonment sets `repair_state = 'abandoned'`
- [ ] An abandoned record is **retained and still counted**, never deleted
- [ ] A stage succeeding late bumps `evidence_generation`, and the video re-interprets with **zero GPU extraction jobs enqueued**
- [ ] Re-interpretation is scoped to that video, not the corpus

**Verification:**
- [ ] `uv run pytest tests/test_repair.py -m integration`
- [ ] Manual: fail VLM, let repair succeed later, confirm the interpretation upgrades and no extraction job was created

**Dependencies:** C10
**Files:** `src/vl/workers/repair.py`, `src/vl/cli.py`, `tests/test_repair.py`
**Scope:** M

---

### ✅ Checkpoint C — first end-to-end slice

- [ ] `vl ingest` → `normalize` → `asr` → `embed` → `fuse` produces an interpretation with citations
- [ ] Drift test passes on a VFR, offset fixture
- [ ] A deliberately failed stage still reaches fusion, flagged degraded, and repairs later
- [ ] No LLM output in any evidence table
- [ ] `mise run check` passes
- [ ] **Human review before Phase D**

---

## Phase D — Slice 3: `vl ask` on Audio Evidence

Deliberately early, while evidence is cheap and the corpus is small.

### Task D1: Hybrid retrieval and RRF

**Description:** `retrieval/` — a lexical leg (Postgres full-text) and a dense leg
(pgvector), fused with Reciprocal Rank Fusion, `retrieval_candidates` per leg.

**Acceptance criteria:**
- [ ] Both legs run and are individually testable
- [ ] RRF fusion is a pure function over two ranked lists
- [ ] Exact-phrase queries retrieve via the lexical leg even when embeddings miss them
- [ ] Results carry `video_id` and timestamps for citation

**Verification:**
- [ ] `uv run pytest tests/test_retrieval.py -m integration`
- [ ] Manual: known-answer query returns the expected span in the top 5

**Dependencies:** Phase C
**Files:** `src/vl/retrieval/hybrid.py`, `src/vl/retrieval/rrf.py`, `tests/test_retrieval.py`
**Scope:** M

---

### Task D2: Query embedding and rerank client

**Description:** Infinity client for query embedding and bge-reranker, cutting
`retrieval_candidates` down to `retrieval_top_k`. **Neither model is quantized**
(invariant 15).

**Acceptance criteria:**
- [ ] Reranking narrows candidates to `retrieval_top_k`
- [ ] Reranker demonstrably reorders — a test asserts the order differs from RRF order on a crafted case
- [ ] Service unavailability degrades to RRF-only with a warning, rather than failing the query
- [ ] Client targets `Settings.embed_url`

**Verification:**
- [ ] `uv run pytest tests/test_rerank.py`
- [ ] Manual on GPU host against live `embed-serve`

**Dependencies:** D1, A3
**Files:** `src/vl/retrieval/rerank.py`, `tests/test_rerank.py`
**Scope:** S

---

### Task D3: `vl ask` — answer with validated citations

**Description:** Retrieval → prompt → guided decoding → **C8 validation** → formatted
answer. Output matches the spec's worked example: prose, then `video_id @ timestamp`
citations.

**Acceptance criteria:**
- [ ] Every quote in the output appears verbatim in `evidence_spans` (invariant 7)
- [ ] Every citation resolves to a real `video_id` and a timestamp within that video's duration
- [ ] Output format matches the spec example, including the "… N more" truncation
- [ ] A question with no supporting evidence returns "no evidence found", never a confident invention
- [ ] `ask` preempts fusion via priority scheduling on the shared `llm-serve`

**Verification:**
- [ ] `uv run pytest tests/test_ask.py -m integration`
- [ ] Manual: `vl ask "what breaks most often in the first week?"` — open a cited video at its timestamp and confirm the evidence is there

**Dependencies:** D2, C8
**Files:** `src/vl/cli.py`, `src/vl/ask.py`, `src/vl/prompts/ask.py`, `tests/test_ask.py`
**Scope:** M

---

### Task D4: Citation drop-rate metric surfaced

**Description:** Make C8's counter observable in both paths — fusion and `ask` — as a
queryable metric rather than a log line. Required by spec success criterion 7.

**Acceptance criteria:**
- [ ] Drop rate persisted per `(path, prompt_version, model_id)`
- [ ] Readable ahead of `vl stats` existing (Phase K wires the display)
- [ ] A rising drop rate is visible without grepping logs

**Verification:**
- [ ] `uv run pytest tests/test_metrics.py -m integration`
- [ ] Manual: force rejections, confirm the metric moves

**Dependencies:** D3
**Files:** `src/vl/io/db/metrics.py`, `migrations/0003_metrics.sql`, `tests/test_metrics.py`
**Scope:** S — retires risk 1 for the `ask` path

---

### Task D5: OCR-on-CPU benchmark 🔓

**Description:** The spec's deferred table reopens "OCR on the CPU pool" when a slice-3
benchmark shows CPU keeps pace at ~130 frames/video. Run that benchmark now, because it
determines whether Phase E's OCR stage targets `cpu` or `gpu-small`.

**Acceptance criteria:**
- [ ] RapidOCR throughput measured on the `cpu` image at ~130 frames/video
- [ ] Compared against `gpu-small` throughput on identical input
- [ ] Result recorded in the spec's deferred table with a date and a decision
- [ ] Decision consumed by Task E3's pool assignment

**Verification:**
- [ ] Benchmark script runs reproducibly and reports frames/second per pool

**Dependencies:** Phase C
**Files:** `bench/ocr_pool.py`, `docs/specs/video-brand-research.md` (deferred table)
**Scope:** S — **must run before E3**, or E3 guesses at a pool assignment

---

### ✅ Checkpoint D — first analyst-visible value

- [ ] An analyst can ask a question and get cited answers from audio-only evidence
- [ ] Citation drop rate is measurable in both paths
- [ ] OCR pool decision made on data, not intuition
- [ ] **Human review — this is the first point the product is demonstrable**

---

## Phase E — Slice 4: Keyframes and OCR

Before any VLM cost is incurred. The detector's picks get eyeballed first, because every
bad frame is paid for at ~256 visual tokens.

### Task E1: `keyframes` stage
Scene-cut detection (`scene_threshold`), phash dedup (`phash_hamming_max`), mid-shot
frames for shots over `long_shot_s`, capped at `max_keyframes`, preferring frames inside
speech spans — falling back to cuts + phash alone when ASR failed.
**Acceptance:** ≤`max_keyframes` frames emitted; near-identical frames dropped; the ASR-failed fallback path is tested; runs on `cpu` within the ~18 s/video CPU budget.
**Verify:** `uv run pytest tests/test_keyframes.py`; manual timing on a 3-minute video.
**Depends:** Phase D · **Files:** `src/vl/stages/keyframes.py` · **Scope:** M

### Task E2: Keyframe eyeball harness
Contact-sheet output so a human can judge the detector's picks before VLM spend.
**Acceptance:** renders selected frames with timestamps for a sample of ≥20 videos; picks reviewed and the reviewer's verdict recorded.
**Verify:** manual review of the contact sheets.
**Depends:** E1 · **Files:** `bench/keyframe_sheet.py` · **Scope:** S

### Task E3: `ocr` stage
RapidOCR over keyframes → `SCREEN` point spans, on the pool D5 selected.
**Acceptance:** spans carry normalized-media timestamps; low-confidence detections filtered; pool matches the D5 decision; `COPY` bulk write.
**Verify:** `uv run pytest tests/test_ocr.py -m integration`.
**Depends:** E1, D5 · **Files:** `src/vl/stages/ocr.py` · **Scope:** M

### Task E4: OCR into timeline and fusion
**Acceptance:** `SCREEN` events interleave correctly (C6 tolerance rules); a silent screen recording with OCR but no ASR fuses successfully, exercising `FUSE_REQUIRES_ANY_OF`.
**Verify:** `uv run pytest tests/test_timeline.py tests/test_fuse.py`.
**Depends:** E3 · **Files:** `src/vl/timeline.py`, `src/vl/prompts/fusion.py` · **Scope:** S

### ✅ Checkpoint E
- [ ] Keyframe picks reviewed by a human and judged acceptable
- [ ] A silent screen recording fuses on OCR alone
- [ ] Frame budget holds across 9:16, 16:9 and 2.35:1 inputs

---

## Phase F — Slice 5: VLM Captions

~55% of all GPU time. The first real `$/video` measurement.

### Task F1: vLLM client with area-based frame capping ⚠
**Acceptance:** frames capped by `vlm_max_pixels` **area**, never longest side (invariant 8); a 9:16 and a 2.35:1 frame of equal area produce equal token counts — asserted by test, this is the 2.35× cost spread the invariant exists to prevent; `vlm_min_pixels` prevents upscaling.
**Verify:** `uv run pytest tests/test_vlm_client.py`.
**Depends:** Phase E · **Files:** `src/vl/stages/vlm.py`, `src/vl/models/vision.py` · **Scope:** M

### Task F2: Prompt ordering for prefix caching
**Acceptance:** invariant prompt text precedes all images (invariant 9); cache hit rate measured and reported over ≥100 videos; a test fails if variable content moves ahead of the prefix.
**Verify:** `uv run pytest tests/test_vlm_prompt.py`; manual cache-hit-rate reading from vLLM metrics.
**Depends:** F1 · **Files:** `src/vl/prompts/vlm.py` · **Scope:** S

### Task F3: `vlm` stage → caption spans
**Acceptance:** captions emitted as `VISUAL` point spans at ~256 visual tokens/frame; guided decoding for structured output (invariant 10); idempotent; interleaves into fusion.
**Verify:** `uv run pytest tests/test_vlm.py -m integration`; manual GPU sweep.
**Depends:** F2 · **Files:** `src/vl/stages/vlm.py` · **Scope:** M

### Task F4: First `$/video` measurement 📊
**Acceptance:** measured on **≥1,000 real videos**, broken down per stage; VLM share of GPU time reported against the ~55% expectation; extrapolation to a million videos recorded with its assumptions stated.
**Verify:** measurement run completes; numbers written into the spec's Key Numbers section.
**Depends:** F3 · **Files:** `bench/cost.py`, `docs/specs/video-brand-research.md` · **Scope:** S — **gate: do not scale out before this**

### ✅ Checkpoint F
- [ ] `$/video` measured on ≥1k videos and matches the cost model within a stated tolerance
- [ ] Frame budget holds; prefix cache hit rate acceptable
- [ ] **Human review — this is the affordability go/no-go**

---

## Phase G — Slice 6: Facial Expression Recognition

Included deliberately despite ~60–65% in-the-wild accuracy, and shipped with its weakness
visible.

### Task G1: `fer` stage
MediaPipe + HSEmotion on ONNX Runtime **CPU** (invariant 13 — no CUDA in `vl-cpu`).
**Acceptance:** emits `FACE` point spans with an explicit confidence; frames with no detected face are skipped, not guessed at; runs on the `cpu` pool.
**Verify:** `uv run pytest tests/test_fer.py`.
**Depends:** Phase F · **Files:** `src/vl/stages/fer.py` · **Scope:** M

### Task G2: `fer_frames_pct` surfaced
**Acceptance:** present on **every** video record, including those where FER found nothing; carried into fusion prompts so the model knows how thin the signal is; visible in `vl report` output.
**Verify:** `uv run pytest tests/test_fer.py -m integration`; manual: a video with no faces still reports a value (0), not null.
**Depends:** G1 · **Files:** `src/vl/stages/fer.py`, `migrations/0004_fer.sql` · **Scope:** S

### ✅ Checkpoint G
- [ ] All six extraction stages produce evidence; the full interleave from the spec renders
- [ ] FER's weakness is visible in output rather than hidden in an aggregate

---

## Phase H — Retire the Central Risk

The point of the whole system. Everything before this is machinery.

### Task H1: Evaluation harness
Run the B8 labelled set through the pipeline and score fused output against ground truth.
**Acceptance:** produces intent accuracy and sentiment accuracy against the labelled set; reproducible from a single command; results version-controlled with the `model_id` and `prompt_version` that produced them.
**Verify:** `uv run pytest tests/test_eval.py`; harness runs end to end on the labelled set.
**Depends:** Phase G, B8 · **Files:** `eval/run.py`, `tests/test_eval.py` · **Scope:** M

### Task H2: Text-only vs fused accuracy report ⚠
The claim "visual corrects text" either gets a number or gets withdrawn.
**Acceptance:** reports text-only accuracy (from `sentiment.text_only_label`) beside fused accuracy on the same videos; **breaks out the 5+ contradiction cases separately**, since they are where the claim actually lives; the result is published whether or not it is favourable; if fusion does not beat text-only, that is written into the spec as a finding rather than quietly dropped.
**Verify:** report generated and reviewed; spec updated with the measured numbers.
**Depends:** H1 · **Files:** `eval/report.py`, `docs/specs/video-brand-research.md` · **Scope:** S — **retires risk 2, or kills the premise**

### ✅ Checkpoint H
- [ ] Spec success criteria 10 and 11 satisfied with real numbers
- [ ] **Human review — if fusion does not beat text-only on the contradiction cases, stop and reconsider before Phases I–K**

---

## Phases I–K: Aggregate, Prove, Measure

### Task I1: Incremental clustering
**Acceptance:** clusters complaints incrementally as new interpretations land, without full recomputation; degraded records **are included** (excluding them would bias every count toward easily-processed videos).
**Verify:** `uv run pytest tests/test_cluster.py -m integration`.
**Depends:** Phase H · **Files:** `src/vl/cluster/` · **Scope:** M

### Task I2: `vl report`
**Acceptance:** complaints ranked by video count; `degraded_pct` and `fer_frames_pct` surfaced per cluster; every cluster links to grounding evidence with citations.
**Verify:** `uv run pytest tests/test_report.py`; manual: open the HTML and follow a citation to its video.
**Depends:** I1 · **Files:** `src/vl/report/`, `src/vl/cli.py` · **Scope:** M

### Task J1: `vl reinterpret` ⚠
**Acceptance:** rebuilds every conclusion at a new `prompt_version` with **zero GPU extraction jobs enqueued** — asserted by a job counter, not by inspection (spec success criterion 5); old interpretations retained for comparison; a corpus-wide run is resumable.
**Verify:** `uv run pytest tests/test_reinterpret.py -m integration`; manual: `vl reinterpret --prompt-version v4`, confirm the extraction queue stayed empty.
**Depends:** Phase H · **Files:** `src/vl/cli.py`, `src/vl/workers/reinterpret.py` · **Scope:** M — **proves the two-layer split is real**

### Task J2: Provenance comparison
**Acceptance:** fp16 and FP8 evidence for the same video can be queried side by side, keyed on `model_id`.
**Verify:** `uv run pytest tests/test_provenance.py -m integration`.
**Depends:** J1 · **Files:** `src/vl/io/db/spans.py` · **Scope:** S

### Task K1: `vl stats --cost`
**Acceptance:** `$/video` per stage, queue depth, GPU seconds, failure rates, and the D4 citation drop rate all in one output.
**Verify:** `uv run pytest tests/test_stats.py -m integration`; manual: numbers reconcile with F4's measurement.
**Depends:** Phase I · **Files:** `src/vl/cli.py`, `src/vl/io/db/metrics.py` · **Scope:** M

### Task K2: Scale-out gate review
**Acceptance:** all 14 spec success criteria checked off with evidence; deferred-decision triggers re-evaluated against real numbers (SGLang, WhisperX, Temporal/Ray, lakehouse, W4A16); go/no-go on scale-out recorded.
**Verify:** documented review against the spec.
**Depends:** K1 · **Files:** `docs/specs/video-brand-research.md` · **Scope:** S

### ✅ Checkpoint K — complete
- [ ] All 14 success criteria met
- [ ] `$/video` measured on ≥1k videos before any million-video extrapolation
- [ ] Ready for scale-out review

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **"Visual corrects text" is unfalsifiable** | **High** — it is the premise of the product | B8 starts the labelled set in Phase B; H2 publishes the number whether or not it flatters the system. Checkpoint H is an explicit stop-and-reconsider gate. |
| **Hallucinated quotes** | **High** — destroys analyst trust faster than a missed complaint | C8 is one implementation shared by fusion and `ask`; D4 makes the drop rate a queryable metric, not a log line |
| `$/video` unaffordable at scale | High | F4 measures on ≥1k videos before extrapolation; Checkpoint F is a go/no-go. E2 eyeballs frame picks *before* VLM spend |
| A/V drift ships silently | High — invisible without a dedicated test | C3 is a standalone task with an explicit "break it and confirm the test fails" step |
| GPU images never verified | Medium — blocks B5, C4 onward | Phase A front-loads it; A3 is the earliest hardware-dependent task |
| `models.yaml` absent, `@unset` leaks into data | Medium — corrupts idempotency keys | B4 replaces `default_model_resolver` at all call sites before any real extraction runs |
| Labelled set slips | Medium — silently delays Phase H | B8 starts in Phase B and is marked parallel; treat it as long-lead procurement |
| Fusion prompt reads absence as agreement | Medium | C7 tests a degraded prompt for explicit absence language (invariant 6) |

---

## Parallelization

**Safe to parallelize**
- B1, B2, B4, B8 — independent, no shared files
- B8 runs alongside every phase and should
- E2 (eyeball harness) alongside E3 (OCR)
- Test-writing for a completed stage alongside the next stage's implementation

**Must be sequential**
- All migrations — 0002 → 0003 → 0004, one at a time
- C2 → C3 → C4: the clock must exist and be proven before anything reads it
- C7 → C9, D2 → D3: prompt shape before the caller
- H1 → H2: no report without a harness

**Needs coordination**
- C6 (timeline) is touched again by E4 and G1 — agree the point-event attachment contract in C6 and don't renegotiate it
- C8 is consumed by both C9 and D3 — freeze its signature at C8

**Hardware-gated**
- A3, B5, B7, C4, C5, C9, D2, F1–F4, G1 need a GPU box. Everything else runs on macOS arm64.

---

## Open Questions

Inherited from the spec, plus what this breakdown surfaced.

1. **Who owns the labelled set (B8)?** It is the longest-lead item and the only one that is not code. Without a named owner it will slip to Phase H and stall the project's central claim.
2. **Is a GPU host available now?** A3 and B5 need one. If not, Phase B completes except for B5/B7 and Phase C stalls at C4 — worth knowing before starting rather than at the wall.
3. **Which eval set gates `vl models prepare` (B5)?** The spec says publishing is gated on an eval threshold but does not name the set or the threshold. Needs deciding before B5.
4. **`ty` — adopt or drop?** Task A4 forces the choice; it is XS either way, but the answer changes what "clean" means for every subsequent task.
5. **Does `report` output HTML only?** `vl report --out report.html` implies HTML. Analysts may want CSV for their own tooling. Confirm before I2.

---

## Provenance

Generated from `docs/specs/video-brand-research.md`, cross-checked against
`src/vl/io/db/queue.py`, `src/vl/domain/{stages,jobs,barrier}.py`, `src/vl/config.py`,
`src/vl/cli.py`, `migrations/0001_queue.sql`, `docker-compose.yml` and the 70 existing
tests, as of 2026-08-17. Where this plan and the spec disagree, the spec wins.
