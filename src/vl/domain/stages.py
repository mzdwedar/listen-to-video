"""Pipeline stages and the pools they run on.

Stages are declared in one place because three separate things must agree about them:
the jobs table, the fan-in barrier, and the worker images. A stage that exists in the
pipeline but not here is unreachable; the reverse is a silently idle queue.
"""

from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    NORMALIZE = "normalize"
    ASR = "asr"
    KEYFRAMES = "keyframes"
    OCR = "ocr"
    VLM = "vlm"
    FER = "fer"
    FUSE = "fuse"


class Pool(StrEnum):
    CPU = "cpu"
    GPU_SMALL = "gpu-small"
    GPU_LARGE = "gpu-large"
    LLM_SERVE = "llm-serve"


# Split by resource profile, not convenience. `normalize` and `keyframes` are pure
# ffmpeg/scene-detection work — running them on GPU nodes burns GPU-hours on video
# decoding. `vlm` is ~55% of all GPU time, so it scales independently of everything.
STAGE_POOL: dict[Stage, Pool] = {
    Stage.NORMALIZE: Pool.CPU,
    Stage.KEYFRAMES: Pool.CPU,
    Stage.FER: Pool.CPU,
    Stage.ASR: Pool.GPU_SMALL,
    Stage.OCR: Pool.GPU_SMALL,
    Stage.VLM: Pool.GPU_LARGE,
    Stage.FUSE: Pool.LLM_SERVE,
}

# Stages that must reach a terminal state before this one may be claimed. These are
# *soft*: a dependency reaching `failed_permanent` still unblocks the dependent stage,
# which then runs degraded. See invariant 5 — the barrier waits on terminal states,
# not on success.
STAGE_DEPENDS_ON: dict[Stage, tuple[Stage, ...]] = {
    Stage.NORMALIZE: (),
    Stage.ASR: (Stage.NORMALIZE,),
    # Keyframe selection prefers frames inside speech spans, so it waits for ASR to
    # finish — but falls back to cuts + phash alone if ASR failed.
    Stage.KEYFRAMES: (Stage.NORMALIZE, Stage.ASR),
    Stage.OCR: (Stage.KEYFRAMES,),
    Stage.VLM: (Stage.KEYFRAMES,),
    Stage.FER: (Stage.KEYFRAMES,),
    Stage.FUSE: (Stage.ASR, Stage.OCR, Stage.VLM, Stage.FER),
}

# Fusion needs any text signal at all: spoken words or on-screen text. This covers
# silent screen recordings (OCR only) and audio-only reviews (ASR only). Losing both
# means dead-letter — never a fused record built on nothing.
FUSE_REQUIRES_ANY_OF: tuple[Stage, ...] = (Stage.ASR, Stage.OCR)

# Everything else may be missing; the record is fused and flagged degraded.
FUSE_OPTIONAL: tuple[Stage, ...] = (Stage.KEYFRAMES, Stage.VLM, Stage.FER)
