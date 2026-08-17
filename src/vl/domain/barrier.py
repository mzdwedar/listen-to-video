"""Fan-in decisions, as pure functions of state.

No clock, no database, no I/O — a caller reads stage states and a grace flag from
Postgres and asks these functions what to do. Keeping it pure is what makes the
awkward cases (everything failed; text arrived but captions didn't; the grace period
expired mid-flight) cheap to enumerate in tests rather than reproduce in staging.

The governing rule: **terminal, not successful.** A dependency that failed
permanently unblocks its dependents, which then run degraded. Requiring success
anywhere here would strand videos forever on a single bad stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from vl.domain.jobs import StageState
from vl.domain.stages import (
    FUSE_REQUIRES_ANY_OF,
    STAGE_DEPENDS_ON,
    Stage,
)

REPAIR_BACKOFF_S: tuple[int, ...] = (300, 1800, 14400, 86400, 86400)
"""5 min, 30 min, 4 h, 1 day, 1 day. Past the last round, repair is abandoned."""

EXTRACTION_STAGES: tuple[Stage, ...] = tuple(s for s in Stage if s is not Stage.FUSE)
"""Everything that produces evidence. `fuse` consumes it, so it is never 'missing'."""


class FuseAction(Enum):
    WAIT = auto()
    """Not settled yet, and the grace period has not expired."""

    FUSE = auto()
    """Interpret now. May be degraded — check `missing`."""

    DEAD_LETTER = auto()
    """No text signal survived. Nothing to interpret; never fuse."""


@dataclass(frozen=True, slots=True)
class FuseDecision:
    action: FuseAction
    missing: tuple[Stage, ...] = ()
    """Extraction stages that did not succeed. Drives the degraded prompt and repair."""

    @property
    def degraded(self) -> bool:
        return bool(self.missing)


def ready_to_claim(stage: Stage, stage_states: dict[Stage, StageState]) -> bool:
    """Whether `stage` may be claimed for a video in the given state.

    Dependencies are *soft*: they must be settled, not successful.
    """
    return all(stage_states[dependency].is_terminal for dependency in STAGE_DEPENDS_ON[stage])


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
        # Still a chance one of them lands; otherwise there is nothing to interpret,
        # and no amount of waiting changes that.
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


def next_repair_at(
    completed_rounds: int,
    backoff: tuple[int, ...] = REPAIR_BACKOFF_S,
) -> int | None:
    """Seconds to wait before the next repair attempt, or None to abandon.

    Abandoning keeps the record: permanently degraded, still counted in `vl stats`.
    Dropping it would quietly bias every aggregate toward easily-processed videos.
    """
    if completed_rounds >= len(backoff):
        return None
    return backoff[completed_rounds]
