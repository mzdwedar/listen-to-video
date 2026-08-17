"""Config defaults encode tuning decisions, so a few of them are worth asserting."""

from __future__ import annotations

import pytest

from vl.config import Settings

VISUAL_TOKEN_BLOCK = 28 * 28  # Qwen2.5-VL charges one token per 28x28 pixel block


def test_defaults_load_without_env() -> None:
    s = Settings()
    assert s.postgres_dsn
    assert s.s3_bucket


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VL_S3_BUCKET", "other-bucket")
    assert Settings().s3_bucket == "other-bucket"


def test_frame_budget_is_an_exact_token_multiple() -> None:
    """Invariant 8: frames are capped by area to get a hard token ceiling.

    If max_pixels isn't an exact multiple of the 28x28 block, the ceiling is not the
    round number it looks like and $/video forecasts drift.
    """
    s = Settings()
    assert s.vlm_max_pixels % VISUAL_TOKEN_BLOCK == 0
    assert s.vlm_max_pixels // VISUAL_TOKEN_BLOCK == 256


def test_frame_budget_is_uniform_across_aspect_ratios() -> None:
    """The whole point of an area cap: cost must not depend on shape.

    A longest-side cap would give 9:16 and 1:1 wildly different token counts.
    """
    s = Settings()
    for w, h in [(9, 16), (16, 9), (1, 1), (235, 100)]:
        scale = (s.vlm_max_pixels / (w * h)) ** 0.5
        tokens = (w * scale) * (h * scale) / VISUAL_TOKEN_BLOCK
        assert tokens == pytest.approx(256, rel=0.01), f"{w}:{h} costs {tokens:.0f} tokens"


def test_pipeline_tuning_defaults() -> None:
    s = Settings()
    assert s.max_keyframes == 40
    assert s.grace_period_s == 300  # 5 minutes, then fuse provisionally
    assert s.sweep_batch_size == 200  # amortises 30-60s model load
    assert s.repair_rounds == 5
