# CLAUDE.md

Video → Voice-of-the-Customer corpus. Extract speech, keyframe captions, and on-screen text from video; align on one timeline; fuse into a self-hosted LLM; store as a queryable corpus that answers analyst questions with citations to `video_id @ timestamp`.

See `README.md` for the narrative version. This file is the operating contract.

## Invariants

Violating any of these is a bug even if tests pass. They were expensive to arrive at and are cheap to break.

1. **Evidence is immutable; interpretation is versioned.** Never write LLM-derived output into evidence tables. Conclusions belong in `interpretations` / `findings`. If a prompt change would require re-running the GPU, the split has been broken.
2. **Every stage handler is a pure function** of `(video_id, input artifacts) → evidence`, idempotent under key `(video_id, stage, model_id)`. This is what keeps the control plane swappable for Temporal or Ray later.
3. **Nothing reads the source video after `normalize`.** Downstream stages read only derived artifacts (WAV, keyframe JPEGs). Reaching back to the source breaks per-stage pools.
4. **All timestamps are normalized-media seconds.** Never source-container time. Container start offsets and variable frame rate cause 200–500ms audio/visual drift that is invisible without the dedicated drift test.
5. **The fan-in barrier waits on terminal states, not success.** `done | failed_permanent | timed_out`. No stage may block a video indefinitely.
6. **A missing modality is not a neutral one.** Degraded fusion prompts must state explicitly what is absent, or the model reads silence as agreement and returns a confident wrong answer.
7. **Quotes are validated verbatim** against evidence before being persisted or returned — in fusion and in `ask` alike. A citation that isn't in the video destroys analyst trust faster than a missed complaint.
8. **Cap VLM frames by area (`max_pixels`), never by longest side.** Aspect ratios vary from 9:16 to 2.35:1; area is what costs money, and a longest-side cap gives a 2.35× cost spread.
9. **Invariant prompt text goes before images**, or prefix caching does nothing across a million videos.
10. **Guided decoding is mandatory** for all structured output. Unconstrained JSON fails on a fraction of videos, silently and non-uniformly.
11. **No ORM.** psycopg3 and hand-written SQL. `COPY` for bulk span writes.
12. **Weights are never fetched at runtime.** Sync from object storage by `model_id`. Quantization happens only in `vl models prepare`.
13. **Never add CUDA to `vl-cpu`.** Per-pool images exist specifically to keep ONNX Runtime and torch out of the same process.
14. **Always build `--platform linux/amd64`.** Dev machines are arm64; ECS is not.
15. **Do not quantize the embedding model or reranker.** They set retrieval quality, and quantization reorders nearest neighbors — degradation is silent and looks like a prompt bug.
16. **SIGTERM releases leases** rather than draining. Spot gives two minutes; a 200-video batch won't finish.
17. **Tests use real Postgres**, never SQLite. `SKIP LOCKED`, partitioning, and pgvector are the things under test.

## Commands

```bash
vl models prepare                 # fetch → quantize → eval gate → publish to S3
vl ingest <path>                  # dedup by content hash + phash, enqueue
vl sweep --stage <stage>          # batch-claim ~200 videos, load model once, process
vl repair                         # retry stages missing on degraded videos
vl reinterpret --prompt-version v # rebuild conclusions, no GPU extraction
vl ask "<question>"               # hybrid retrieval → answer with citations
vl report                         # clustered complaints ranked by video count
vl stats --cost                   # $/video per stage, queue depth, GPU seconds
vl doctor                         # real-model smoke test (GPU box only)

mise run check                    # ty, ruff, pytest
mise run build                    # all five images, linux/amd64
docker compose up -d              # Postgres, MinIO, model services
```

Sweeps are **stage-batched, not per-video**: model load is 30–60s, so a worker claims a batch, loads once, processes all, unloads. This is why the queue is partitioned by stage.

## Layout

```
src/vl/
├── cli.py          Typer entrypoints
├── config.py       OmegaConf + Pydantic
├── domain/         pure types, no I/O
├── io/db/          SQL modules + migrations
├── io/objects.py   artifact store
├── io/sources/     VideoSource protocol, LocalFilesSource
├── stages/         one module per stage, all pure functions
├── timeline.py     spine construction, interleave
├── models/         prepare, quantize, model_id registry
├── retrieval/      hybrid search, RRF, rerank, citation validation
├── cluster/        incremental clustering
├── workers/        sweep loop, lease/claim, SIGTERM, repair sweeper
└── report/
```

## Pools

Split by resource profile. Do not move a stage between pools without re-checking the cost table.

| Pool | Stages | Notes |
|---|---|---|
| `cpu` | normalize, keyframes, fer, ingest, sweepers, report | no CUDA, ~600MB image |
| `gpu-small` | asr, ocr, embed-bulk | CTranslate2 + ORT-GPU + torch |
| `gpu-large` | vlm | vLLM only; ~55% of all GPU time |
| `llm-serve` | fusion, ask | shared; `ask` preempts fusion via priority scheduling |
| `embed-serve` | query embed, rerank | Infinity; vLLM is one model per process |

## Key numbers

- VLM is **~55% of GPU time** — the only stage worth optimizing hard. ~11,000 GPU-hours per million videos.
- **~18s per video is pure CPU** (ffmpeg + scene detection). Never run it on GPU nodes.
- Keyframe budget: **~40 frames** from ~5,400, at **~256 visual tokens each**.
- Measure `$/video` on ≥1k videos before extrapolating to a million.

## Deferred decisions

Settled as "not now," with the trigger that reopens each. Don't relitigate without the trigger.

| Deferred | Reopen when |
|---|---|
| SGLang instead of vLLM | fusion prompt is stable and `$/video` is known — RadixAttention targets our shared-prefix shape |
| WhisperX word-level timestamps | analysts complain citations land early (we have 3–8s segment precision) |
| OCR on the CPU pool | slice-3 benchmark shows CPU keeps pace at ~130 frames/video |
| Temporal or Ray control plane | "where is video X and why did it fail" becomes a recurring question |
| Lakehouse evidence storage | span count passes ~100–300M and pgvector becomes the bottleneck |
| Speaker diarization | user-research recordings enter scope (`evidence_spans.speaker` is reserved) |
| Arabic support | English pipeline is proven end-to-end; embeddings already multilingual |
| W4A16 instead of FP8 | concurrency turns out memory-limited *and* corpus keyframes exist for calibration |

## Build order

Thin vertical slices. Each must work end-to-end before the next.

0. Docs + five images + compose — **written; image builds unverified** (no Docker daemon
   available yet). Run `mise run build` on a machine with Docker before trusting them,
   and verify the `vllm` and `infinity` base image tags against their registries.
1. Skeleton: CLI, migrations with partitioning, jobs table (batch claim, lease, heartbeat, dead-letter), `video_stages` barrier, grace timer, repair sweeper, SIGTERM lease release, `vl models prepare`
2. Audio only, stored: normalize → asr → spans → embeddings → spine → fusion → interpretations
3. `vl ask` on audio-only evidence — early, while evidence is cheap
4. Keyframe detector + OCR — eyeball the detector's picks before incurring VLM cost
5. VLM captions — first real `$/video` measurement
6. FER, with `fer_frames_pct` surfaced
7. Incremental clustering + `vl report`
8. `vl reinterpret` — proves the two-layer split is real
9. `vl stats --cost` — gate before any scale-out

## Two known project risks

**Hallucinated quotes** in both fusion and `ask`. Verbatim validation in both paths; drop rate is a first-class logged metric.

**"Visual corrects text" being unfalsifiable.** Requires a hand-labeled set of 25–30 videos including 5+ where footage contradicts the words. Report text-only vs. fused accuracy against it. `sentiment.text_only_label` is stored beside the fused label specifically so this is measurable rather than asserted.

## Notes

- Facial expression recognition is **deliberately included** despite ~60–65% in-the-wild accuracy and firing on a minority of frames. It ships with `fer_frames_pct` per video so its weakness is visible in output rather than hidden in an aggregate.
- Degraded records **are** counted in aggregates and clustering. Excluding them would bias every count toward easily-processed videos. Reports surface `degraded_pct` per cluster.
- Model identifiers encode `name@revision+scheme`, so re-quantizing produces distinct provenance and fp16/FP8 evidence for the same video can be compared side by side.
- GPU images build on macOS but cannot run there. Use `vl doctor` on the GPU box for real-model verification.
