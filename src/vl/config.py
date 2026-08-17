"""Settings, read from the environment with a `VL_` prefix.

Several defaults here are load-bearing decisions rather than arbitrary numbers; those
carry a comment explaining what breaks if they change.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

VISUAL_TOKEN_BLOCK = 28 * 28
"""Qwen2.5-VL charges one visual token per 28x28 pixel block."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VL_",
        env_file=".env",
        extra="ignore",
    )

    # --- storage -----------------------------------------------------------------
    postgres_dsn: str = "postgresql://vl:vl@localhost:5432/vl"
    s3_endpoint: str = "http://localhost:9000"  # MinIO locally, empty for real S3
    s3_bucket: str = "vl-artifacts"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"

    # --- model services ----------------------------------------------------------
    vlm_url: str = "http://localhost:8001/v1"
    llm_url: str = "http://localhost:8002/v1"
    embed_url: str = "http://localhost:8003"

    # --- frame budget ------------------------------------------------------------
    # Cap by AREA, never by longest side (invariant 8). The corpus mixes 9:16 social
    # video, 16:9 reviews and near-square screen recordings; a longest-side cap gives
    # those a 2.35x cost spread, which makes $/video unforecastable.
    vlm_max_pixels: int = 256 * VISUAL_TOKEN_BLOCK  # 200,704 -> ~256 tokens/frame
    vlm_min_pixels: int = 4 * VISUAL_TOKEN_BLOCK  # avoid upscaling tiny frames

    # --- keyframe selection ------------------------------------------------------
    max_keyframes: int = 40  # from ~5,400 frames in a 3-minute video
    scene_threshold: float = 27.0  # PySceneDetect ContentDetector, HSV frame delta
    phash_hamming_max: int = 5  # drop near-identical frames within this distance
    long_shot_s: float = 4.0  # shots longer than this also get a mid-shot frame

    # --- timeline ----------------------------------------------------------------
    silence_gap_s: float = 2.0  # gaps longer than this become [silence] segments
    point_attach_tolerance_s: float = 2.0  # how far a point event may reach for a span

    # --- work distribution -------------------------------------------------------
    # Model load is 30-60s, so workers claim a batch and load once rather than paying
    # that per video. This is why the queue is partitioned by stage.
    sweep_batch_size: int = 200
    lease_duration_s: int = 1800
    max_attempts: int = 3

    # --- fan-in and repair -------------------------------------------------------
    # Once there is any text signal, wait briefly for the optional stages, then fuse
    # provisionally and flag the video for repair rather than blocking the corpus.
    grace_period_s: int = 300
    repair_rounds: int = 5
    repair_backoff_s: tuple[int, ...] = (300, 1800, 14400, 86400, 86400)

    # --- serving -----------------------------------------------------------------
    # Fusion and `ask` share one LLM service. Capping fusion concurrency keeps an
    # interactive question from queueing behind a saturating sweep.
    fusion_concurrency: int = 8

    # --- retrieval ---------------------------------------------------------------
    retrieval_candidates: int = 100  # fetched per leg, before reranking
    retrieval_top_k: int = 20  # handed to the LLM
    quote_match_ratio: float = 0.9  # verbatim validation threshold (invariant 7)
