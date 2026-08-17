"""Fan-in semantics.

The rule under test throughout: **the barrier waits on terminal states, not on
success** (CLAUDE.md #5). Every stage reaches done, failed_permanent or timed_out in
bounded time, so fusion is always reachable and no stage can hang a video.
"""

from __future__ import annotations

import pytest

from vl.domain.barrier import (
    FuseAction,
    fuse_decision,
    next_repair_at,
    ready_to_claim,
)
from vl.domain.jobs import StageState as S
from vl.domain.stages import Stage

TERMINAL_FAIL = [S.FAILED_PERMANENT, S.TIMED_OUT]


def states(**overrides: S) -> dict[Stage, S]:
    """All stages pending unless overridden."""
    base = dict.fromkeys(Stage, S.PENDING)
    for name, value in overrides.items():
        base[Stage(name)] = value
    return base


# --- claimability ---------------------------------------------------------------


def test_stage_with_no_dependencies_is_immediately_claimable() -> None:
    assert ready_to_claim(Stage.NORMALIZE, states())


def test_stage_waits_while_a_dependency_runs() -> None:
    assert not ready_to_claim(Stage.ASR, states(normalize=S.RUNNING))


def test_stage_waits_while_a_dependency_is_pending() -> None:
    assert not ready_to_claim(Stage.ASR, states())


def test_stage_proceeds_once_a_dependency_succeeds() -> None:
    assert ready_to_claim(Stage.ASR, states(normalize=S.DONE))


@pytest.mark.parametrize("failure", TERMINAL_FAIL)
def test_dependencies_are_soft_a_failed_dependency_still_unblocks(failure: S) -> None:
    """Keyframe selection prefers frames inside speech spans, so it waits for ASR.

    But if ASR failed permanently it must still run — degraded, on cuts and phash
    alone. Requiring *success* here would strand every video with bad audio.
    """
    assert ready_to_claim(Stage.KEYFRAMES, states(normalize=S.DONE, asr=failure))


def test_visual_stages_wait_for_keyframes_then_proceed_regardless() -> None:
    st = states(normalize=S.DONE, asr=S.DONE, keyframes=S.RUNNING)
    assert not ready_to_claim(Stage.VLM, st)

    st[Stage.KEYFRAMES] = S.FAILED_PERMANENT
    assert ready_to_claim(Stage.VLM, st)


# --- fuse policy ----------------------------------------------------------------


def test_waits_while_text_stages_are_still_undecided() -> None:
    st = states(normalize=S.DONE, asr=S.RUNNING)
    assert fuse_decision(st).action is FuseAction.WAIT


@pytest.mark.parametrize("asr_fail", TERMINAL_FAIL)
@pytest.mark.parametrize("ocr_fail", TERMINAL_FAIL)
def test_dead_letters_when_all_text_signal_is_lost(asr_fail: S, ocr_fail: S) -> None:
    """No spoken words and no on-screen text means there is nothing to interpret.

    Fusing here would produce a confident record built on nothing at all.
    """
    st = states(normalize=S.DONE, asr=asr_fail, keyframes=S.DONE, ocr=ocr_fail)
    assert fuse_decision(st).action is FuseAction.DEAD_LETTER


def test_fuses_on_speech_alone_when_ocr_failed() -> None:
    """An audio-only review is a perfectly good record."""
    st = states(
        normalize=S.DONE,
        asr=S.DONE,
        keyframes=S.DONE,
        ocr=S.FAILED_PERMANENT,
        vlm=S.DONE,
        fer=S.DONE,
    )
    decision = fuse_decision(st)
    assert decision.action is FuseAction.FUSE
    assert decision.degraded
    assert Stage.OCR in decision.missing


def test_fuses_on_on_screen_text_alone_when_speech_failed() -> None:
    """A silent screen recording with a readable error dialog is real evidence."""
    st = states(
        normalize=S.DONE,
        asr=S.FAILED_PERMANENT,
        keyframes=S.DONE,
        ocr=S.DONE,
        vlm=S.DONE,
        fer=S.DONE,
    )
    decision = fuse_decision(st)
    assert decision.action is FuseAction.FUSE
    assert Stage.ASR in decision.missing


def test_complete_video_fuses_undegraded() -> None:
    st = dict.fromkeys(Stage, S.DONE)
    decision = fuse_decision(st)
    assert decision.action is FuseAction.FUSE
    assert not decision.degraded
    assert decision.missing == ()


def test_waits_for_optional_stages_inside_the_grace_period() -> None:
    """Text is in, captions are still running: worth a short wait for a full record."""
    st = states(normalize=S.DONE, asr=S.DONE, keyframes=S.DONE, ocr=S.DONE, vlm=S.RUNNING)
    assert fuse_decision(st, grace_expired=False).action is FuseAction.WAIT


def test_fuses_provisionally_once_the_grace_period_expires() -> None:
    """ "Come back to it later": ship a provisional record, flag what's missing."""
    st = states(normalize=S.DONE, asr=S.DONE, keyframes=S.DONE, ocr=S.DONE, vlm=S.RUNNING)
    decision = fuse_decision(st, grace_expired=True)
    assert decision.action is FuseAction.FUSE
    assert decision.degraded
    assert Stage.VLM in decision.missing
    assert Stage.FER in decision.missing


def test_missing_never_includes_fuse_itself() -> None:
    st = states(normalize=S.DONE, asr=S.DONE, keyframes=S.DONE, ocr=S.DONE)
    decision = fuse_decision(st, grace_expired=True)
    assert Stage.FUSE not in decision.missing


def test_grace_expiry_cannot_rescue_a_video_with_no_text() -> None:
    """Timing out is not a substitute for having something to say."""
    st = states(normalize=S.DONE, asr=S.TIMED_OUT, keyframes=S.DONE, ocr=S.TIMED_OUT)
    assert fuse_decision(st, grace_expired=True).action is FuseAction.DEAD_LETTER


# --- repair scheduling ----------------------------------------------------------


def test_repair_backoff_grows() -> None:
    delays = [next_repair_at(r) for r in range(5)]
    assert all(d is not None for d in delays)
    assert delays == sorted(delays)  # type: ignore[type-var]


def test_repair_is_abandoned_after_the_last_round() -> None:
    """The record survives, permanently degraded and counted — never silently dropped."""
    assert next_repair_at(5) is None


def test_first_repair_is_prompt_and_last_is_patient() -> None:
    assert next_repair_at(0) == 300  # 5 min
    assert next_repair_at(4) == 86400  # a day
