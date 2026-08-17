"""Graceful shutdown.

Invariant 16: SIGTERM releases leases rather than draining. Spot gives two minutes and
a 200-video batch will not finish in that time, so the worker stops claiming, hands
back what it is holding, and exits.
"""

from __future__ import annotations

import os
import signal

from vl.workers.shutdown import ShutdownGuard, process_batch


def test_starts_clean() -> None:
    with ShutdownGuard() as guard:
        assert not guard.requested


def test_sigterm_sets_the_flag() -> None:
    with ShutdownGuard() as guard:
        os.kill(os.getpid(), signal.SIGTERM)
        assert guard.requested


def test_sigint_also_stops_work() -> None:
    """Ctrl-C during a local sweep should behave like a deploy rolling the pod."""
    with ShutdownGuard() as guard:
        os.kill(os.getpid(), signal.SIGINT)
        assert guard.requested


def test_previous_handlers_are_restored() -> None:
    """A sweep must not permanently alter the process's signal disposition."""
    before = signal.getsignal(signal.SIGTERM)
    with ShutdownGuard():
        pass
    assert signal.getsignal(signal.SIGTERM) is before


def test_batch_runs_to_completion_when_undisturbed() -> None:
    done: list[int] = []
    released: list[list[int]] = []

    remaining = process_batch(
        [1, 2, 3],
        handler=done.append,
        release=released.append,
    )

    assert done == [1, 2, 3]
    assert remaining == []
    assert released == []


def test_batch_stops_and_releases_the_remainder_on_sigterm() -> None:
    """The item in flight finishes; everything unstarted goes back to the queue.

    Finishing the current item is deliberate — it is seconds of work, and abandoning
    it would waste GPU time already spent.
    """
    done: list[int] = []
    released: list[list[int]] = []

    def handler(item: int) -> None:
        done.append(item)
        if item == 2:
            os.kill(os.getpid(), signal.SIGTERM)

    remaining = process_batch(
        [1, 2, 3, 4, 5],
        handler=handler,
        release=released.append,
    )

    assert done == [1, 2, 3] or done == [1, 2]
    assert remaining
    assert released == [remaining]
    assert set(done).isdisjoint(remaining), "an item was both processed and released"


def test_every_item_is_either_processed_or_released() -> None:
    """No item may be silently dropped — that is a video that never gets analysed."""
    items = list(range(10))
    done: list[int] = []
    released: list[list[int]] = []

    def handler(item: int) -> None:
        done.append(item)
        if item == 4:
            os.kill(os.getpid(), signal.SIGTERM)

    remaining = process_batch(items, handler=handler, release=released.append)

    assert sorted(done + remaining) == items


def test_a_failing_item_does_not_abort_the_batch() -> None:
    """One corrupt video must not cost the other 199 their model load."""
    done: list[int] = []
    failed: list[tuple[int, str]] = []

    def handler(item: int) -> None:
        if item == 2:
            raise RuntimeError("moov atom not found")
        done.append(item)

    process_batch(
        [1, 2, 3],
        handler=handler,
        release=lambda _: None,
        on_error=lambda item, exc: failed.append((item, str(exc))),
    )

    assert done == [1, 3]
    assert failed == [(2, "moov atom not found")]
