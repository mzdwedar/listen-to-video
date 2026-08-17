"""Job and stage state.

Two separate state machines, deliberately:

- `JobState` tracks one *attempt* at a stage — what a worker is holding a lease on.
- `StageState` tracks whether a stage is *settled* for a video, which is what the
  fan-in barrier reads.

Collapsing them would make "this stage failed twice and will be retried" and "this
stage has permanently failed" the same value, and the barrier cannot tell those apart
without hanging on the first.
"""

from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
    READY = "ready"
    """Claimable, dependencies satisfied."""

    BLOCKED = "blocked"
    """Waiting on a dependency to reach a terminal state."""

    LEASED = "leased"
    """Held by a worker; reclaimable once `lease_until` passes."""

    DONE = "done"

    RETRY = "retry"
    """Failed, attempts remain."""

    DEAD = "dead"
    """Failed with attempts exhausted; the row keeps the error and artifact URIs."""


class StageState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED_PERMANENT = "failed_permanent"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        """Settled one way or another — which is all the barrier needs to proceed."""
        return self in _TERMINAL

    @property
    def succeeded(self) -> bool:
        return self is StageState.DONE


_TERMINAL = frozenset(
    {
        StageState.DONE,
        StageState.FAILED_PERMANENT,
        StageState.TIMED_OUT,
    }
)


class RepairState(StrEnum):
    NONE = "none"
    """Nothing missing."""

    NEEDS_REPAIR = "needs_repair"
    """Fused provisionally; missing stages are queued for retry on backoff."""

    ABANDONED = "abandoned"
    """Repair rounds exhausted. Record retained, permanently degraded, still counted."""
