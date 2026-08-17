"""Graceful shutdown for sweep workers.

On Spot, interruption is the normal case rather than an exception, and the worker gets
about two minutes' notice. A 200-video batch will not finish in that window, so the
correct response is to stop claiming, hand the untouched remainder back to the queue,
and exit — not to attempt a drain that will be killed halfway through anyway.

Releasing rather than draining also keeps attempt accounting honest: preemption is not
a failure of the work, so it must not consume a retry.
"""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterable, Sequence
from types import FrameType

STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class ShutdownGuard:
    """Records a shutdown request without acting inside the signal handler.

    The handler only sets a flag. Doing database work from a signal handler risks
    re-entering a connection mid-statement, and it isn't necessary: if the process is
    killed before it can release, the lease expires and the work returns anyway. The
    release path is an optimisation over lease expiry, not a correctness requirement.
    """

    def __init__(self) -> None:
        self._requested = False
        self._previous: dict[int, object] = {}

    @property
    def requested(self) -> bool:
        return self._requested

    def _handle(self, signum: int, frame: FrameType | None) -> None:  # noqa: ARG002
        self._requested = True

    def __enter__(self) -> ShutdownGuard:
        for sig in STOP_SIGNALS:
            self._previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)
        return self

    def __exit__(self, *exc: object) -> None:
        # Restore rather than reset to default: a sweep is a guest in this process and
        # must not leave its signal disposition altered.
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)  # type: ignore[arg-type]
        self._previous.clear()


def process_batch[T](
    items: Iterable[T],
    *,
    handler: Callable[[T], None],
    release: Callable[[list[T]], None],
    on_error: Callable[[T, Exception], None] | None = None,
) -> list[T]:
    """Process a claimed batch, releasing the remainder if shutdown is requested.

    Returns the released remainder. Every item is either handled or released — losing
    one silently means a video that never gets analysed and never shows up as failed.

    An item that raises is reported and skipped rather than aborting the batch: one
    corrupt video must not cost the other 199 their 30-60s model load.
    """
    pending: Sequence[T] = list(items)
    processed = 0

    with ShutdownGuard() as guard:
        for index, item in enumerate(pending):
            if guard.requested:
                break
            processed = index + 1
            try:
                handler(item)
            except Exception as exc:  # noqa: BLE001 - one bad item must not stop the batch
                if on_error is None:
                    raise
                on_error(item, exc)

    remainder = list(pending[processed:])
    if remainder:
        release(remainder)
    return remainder
