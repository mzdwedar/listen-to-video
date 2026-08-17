"""The work queue: batch claim, leases, retries, dead-letter, fan-out.

Hand-written SQL rather than an ORM (invariant 11). The whole hot path here is
`FOR UPDATE SKIP LOCKED` batch claiming, conditional state transitions and array
updates — the things ORMs express worst.

Two properties this module exists to guarantee:

1. **Two workers never claim the same video.** Overlap means two GPUs doing identical
   work, and at ~11,000 GPU-hours per million videos that is real money.
2. **Nothing is lost when a worker dies.** Leases expire and the work returns. On Spot
   this is the normal case, not the exception.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import psycopg

from vl.domain.barrier import ready_to_claim
from vl.domain.jobs import JobState, StageState
from vl.domain.stages import Stage

CLAIMABLE = ("ready", "retry")

DEFAULT_LEASE_S = 1800
DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class Claim:
    job_id: int
    video_id: str
    stage: Stage
    model_id: str
    attempts: int


def default_model_resolver(stage: Stage) -> str:
    """Placeholder model identity for fanned-out work.

    Real identities come from `models.yaml` once `vl models prepare` exists; until then
    this keeps the idempotency key well-formed without inventing version numbers that
    would later look authoritative.
    """
    return f"{stage.value}@unset"


# --- registration ----------------------------------------------------------------


def register_video(
    conn: psycopg.Connection,
    *,
    video_id: str,
    source_uri: str,
    platform: str | None = None,
    published_at: object | None = None,
    phash: int | None = None,
) -> bool:
    """Register a video and create its barrier rows. False if already known.

    Barrier rows are created for every stage up front so the barrier can be read for a
    video that has not started yet, without special-casing missing rows.
    """
    result = conn.execute(
        """
        INSERT INTO videos (id, source_uri, platform, published_at, phash)
        VALUES (%(id)s, %(source_uri)s, %(platform)s, %(published_at)s, %(phash)s)
        ON CONFLICT (id) DO NOTHING
        """,
        {
            "id": video_id,
            "source_uri": source_uri,
            "platform": platform,
            "published_at": published_at,
            "phash": phash,
        },
    )
    if result.rowcount == 0:
        return False

    conn.execute(
        """
        INSERT INTO video_stages (video_id, stage)
        SELECT %(id)s, unnest(%(stages)s::text[])
        ON CONFLICT DO NOTHING
        """,
        {"id": video_id, "stages": [s.value for s in Stage]},
    )
    return True


def enqueue(
    conn: psycopg.Connection,
    *,
    video_id: str,
    stage: Stage,
    model_id: str,
) -> bool:
    """Make a stage claimable. False if this exact unit of work already exists."""
    result = conn.execute(
        """
        INSERT INTO jobs (video_id, stage, model_id, state)
        VALUES (%(video_id)s, %(stage)s, %(model_id)s, 'ready')
        ON CONFLICT (video_id, stage, model_id) DO NOTHING
        """,
        {"video_id": video_id, "stage": stage.value, "model_id": model_id},
    )
    return result.rowcount > 0


# --- claiming --------------------------------------------------------------------


def claim_batch(
    conn: psycopg.Connection,
    *,
    stage: Stage,
    worker_id: str,
    batch_size: int,
    lease_s: int = DEFAULT_LEASE_S,
) -> list[Claim]:
    """Claim up to `batch_size` videos for one stage.

    Batched, not per-video: model load is 30-60s, so a worker loads once and processes
    the whole batch. `SKIP LOCKED` is what makes concurrent workers take disjoint sets
    instead of blocking on each other.
    """
    rows = conn.execute(
        """
        WITH claimable AS (
            SELECT id
            FROM jobs
            WHERE stage = %(stage)s
              AND state = ANY(%(claimable)s)
            ORDER BY created_at
            LIMIT %(batch_size)s
            FOR UPDATE SKIP LOCKED
        )
        UPDATE jobs j
        SET state       = 'leased',
            worker_id   = %(worker_id)s,
            lease_until = now() + make_interval(secs => %(lease_s)s),
            attempts    = j.attempts + 1,
            updated_at  = now()
        FROM claimable c
        WHERE j.id = c.id
        RETURNING j.id, j.video_id, j.stage, j.model_id, j.attempts
        """,
        {
            "stage": stage.value,
            "claimable": list(CLAIMABLE),
            "batch_size": batch_size,
            "worker_id": worker_id,
            "lease_s": lease_s,
        },
    ).fetchall()

    claims = [
        Claim(job_id=r[0], video_id=r[1], stage=Stage(r[2]), model_id=r[3], attempts=r[4])
        for r in rows
    ]
    if claims:
        _set_stage_state(conn, [(c.video_id, c.stage) for c in claims], StageState.RUNNING)
    return claims


def heartbeat(
    conn: psycopg.Connection,
    *,
    job_ids: list[int],
    worker_id: str,
    lease_s: int = DEFAULT_LEASE_S,
) -> int:
    """Extend leases for work still in progress. Returns rows affected.

    Scoped to the holding worker: a worker whose lease already expired and was
    reclaimed must not be able to take it back by heartbeating.
    """
    return conn.execute(
        """
        UPDATE jobs
        SET lease_until = now() + make_interval(secs => %(lease_s)s),
            updated_at  = now()
        WHERE id = ANY(%(job_ids)s)
          AND worker_id = %(worker_id)s
          AND state = 'leased'
        """,
        {"job_ids": job_ids, "worker_id": worker_id, "lease_s": lease_s},
    ).rowcount


def release(conn: psycopg.Connection, *, job_ids: list[int], worker_id: str) -> int:
    """Hand work back without consuming a retry attempt (invariant 16).

    The SIGTERM path. A 200-video batch cannot finish inside Spot's two-minute warning,
    so the worker releases rather than drains. Preemption is not a failure of the work,
    and charging it to the retry budget would let infrastructure churn dead-letter
    perfectly good videos.
    """
    rows = conn.execute(
        """
        UPDATE jobs
        SET state       = 'ready',
            worker_id   = NULL,
            lease_until = NULL,
            attempts    = GREATEST(attempts - 1, 0),
            updated_at  = now()
        WHERE id = ANY(%(job_ids)s)
          AND worker_id = %(worker_id)s
          AND state = 'leased'
        RETURNING video_id, stage
        """,
        {"job_ids": job_ids, "worker_id": worker_id},
    ).fetchall()

    if rows:
        _set_stage_state(conn, [(r[0], Stage(r[1])) for r in rows], StageState.PENDING)
    return len(rows)


def reclaim_expired(
    conn: psycopg.Connection,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    model_resolver: Callable[[Stage], str] = default_model_resolver,
) -> int:
    """Return work from dead workers, timing out stages that ran out of attempts.

    This is what bounds the barrier: a stage cannot stay unsettled forever, because a
    worker that vanishes has its lease expire and its attempts eventually exhausted.
    """
    rows = conn.execute(
        """
        UPDATE jobs
        SET state       = CASE WHEN attempts >= %(max_attempts)s THEN 'dead' ELSE 'retry' END,
            worker_id   = NULL,
            lease_until = NULL,
            error       = COALESCE(error, 'lease expired'),
            updated_at  = now()
        WHERE state = 'leased'
          AND lease_until < now()
        RETURNING video_id, stage, state
        """,
        {"max_attempts": max_attempts},
    ).fetchall()

    timed_out = [(r[0], Stage(r[1])) for r in rows if r[2] == JobState.DEAD]
    retrying = [(r[0], Stage(r[1])) for r in rows if r[2] == JobState.RETRY]

    if retrying:
        _set_stage_state(conn, retrying, StageState.PENDING)
    if timed_out:
        _set_stage_state(conn, timed_out, StageState.TIMED_OUT)
        for video_id, _ in timed_out:
            fan_out(conn, video_id=video_id, model_resolver=model_resolver)

    return len(rows)


# --- completion and failure ------------------------------------------------------


def complete(
    conn: psycopg.Connection,
    *,
    job_id: int,
    spans_written: int,
    model_resolver: Callable[[Stage], str] = default_model_resolver,
) -> None:
    """Settle a stage as done and enqueue whatever it unblocked."""
    row = conn.execute(
        """
        UPDATE jobs
        SET state = 'done', worker_id = NULL, lease_until = NULL, updated_at = now()
        WHERE id = %(job_id)s
        RETURNING video_id, stage
        """,
        {"job_id": job_id},
    ).fetchone()
    if row is None:
        return

    video_id, stage = row[0], Stage(row[1])
    _set_stage_state(conn, [(video_id, stage)], StageState.DONE, spans_written=spans_written)
    fan_out(conn, video_id=video_id, model_resolver=model_resolver)


def fail(
    conn: psycopg.Connection,
    *,
    job_id: int,
    error: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    model_resolver: Callable[[Stage], str] = default_model_resolver,
) -> JobState:
    """Record a failure: retry if attempts remain, otherwise dead-letter.

    Dead-lettering settles the stage as `failed_permanent`, which unblocks its
    dependents — they run degraded rather than waiting on something that will never
    arrive.
    """
    row = conn.execute(
        """
        UPDATE jobs
        SET state       = CASE WHEN attempts >= %(max_attempts)s THEN 'dead' ELSE 'retry' END,
            worker_id   = NULL,
            lease_until = NULL,
            error       = %(error)s,
            updated_at  = now()
        WHERE id = %(job_id)s
        RETURNING video_id, stage, state
        """,
        {"job_id": job_id, "error": error, "max_attempts": max_attempts},
    ).fetchone()
    if row is None:
        raise LookupError(f"no such job: {job_id}")

    video_id, stage, state = row[0], Stage(row[1]), JobState(row[2])

    if state is JobState.DEAD:
        _set_stage_state(conn, [(video_id, stage)], StageState.FAILED_PERMANENT)
        fan_out(conn, video_id=video_id, model_resolver=model_resolver)
    else:
        _set_stage_state(conn, [(video_id, stage)], StageState.PENDING)

    return state


def fan_out(
    conn: psycopg.Connection,
    *,
    video_id: str,
    model_resolver: Callable[[Stage], str] = default_model_resolver,
) -> list[Stage]:
    """Enqueue every stage whose dependencies are now settled.

    Uses the domain barrier so the "terminal, not successful" rule lives in exactly one
    place. `fuse` is excluded deliberately: it owns the grace timer, so it is enqueued
    by the barrier evaluation rather than by plain dependency satisfaction.
    """
    states = stage_states(conn, video_id)
    enqueued = []
    for stage in Stage:
        if stage is Stage.FUSE:
            continue
        if states[stage].is_terminal or states[stage] is StageState.RUNNING:
            continue
        if not ready_to_claim(stage, states):
            continue
        if enqueue(conn, video_id=video_id, stage=stage, model_id=model_resolver(stage)):
            enqueued.append(stage)
    return enqueued


# --- reads -----------------------------------------------------------------------


def stage_states(conn: psycopg.Connection, video_id: str) -> dict[Stage, StageState]:
    rows = conn.execute(
        "SELECT stage, state FROM video_stages WHERE video_id = %s",
        (video_id,),
    ).fetchall()
    return {Stage(r[0]): StageState(r[1]) for r in rows}


def stage_state(conn: psycopg.Connection, video_id: str, stage: Stage) -> StageState:
    row = conn.execute(
        "SELECT state FROM video_stages WHERE video_id = %s AND stage = %s",
        (video_id, stage.value),
    ).fetchone()
    if row is None:
        raise LookupError(f"no barrier row for {video_id}/{stage.value}")
    return StageState(row[0])


def job_state(conn: psycopg.Connection, video_id: str, stage: Stage) -> JobState:
    row = conn.execute(
        "SELECT state FROM jobs WHERE video_id = %s AND stage = %s ORDER BY id DESC LIMIT 1",
        (video_id, stage.value),
    ).fetchone()
    if row is None:
        raise LookupError(f"no job for {video_id}/{stage.value}")
    return JobState(row[0])


def lease_until(conn: psycopg.Connection, job_id: int):  # noqa: ANN201 - datetime
    return conn.execute("SELECT lease_until FROM jobs WHERE id = %s", (job_id,)).fetchone()[0]


def attempts(conn: psycopg.Connection, job_id: int) -> int:
    return conn.execute("SELECT attempts FROM jobs WHERE id = %s", (job_id,)).fetchone()[0]


def queue_depth(conn: psycopg.Connection) -> dict[Stage, int]:
    """Claimable work per stage — the backpressure signal.

    A growing VLM depth while ASR drains means the VLM pool is undersized. Published to
    CloudWatch on AWS to drive per-pool autoscaling.
    """
    rows = conn.execute(
        """
        SELECT stage, count(*)
        FROM jobs
        WHERE state = ANY(%s)
        GROUP BY stage
        """,
        (list(CLAIMABLE),),
    ).fetchall()
    return {Stage(r[0]): r[1] for r in rows}


# --- internals -------------------------------------------------------------------


def _set_stage_state(
    conn: psycopg.Connection,
    pairs: list[tuple[str, Stage]],
    state: StageState,
    *,
    spans_written: int | None = None,
) -> None:
    conn.execute(
        """
        UPDATE video_stages vs
        SET state         = %(state)s,
            terminal_at   = CASE WHEN %(terminal)s THEN now() ELSE NULL END,
            spans_written = COALESCE(%(spans)s, vs.spans_written)
        FROM unnest(%(video_ids)s::text[], %(stages)s::text[]) AS t(video_id, stage)
        WHERE vs.video_id = t.video_id AND vs.stage = t.stage
        """,
        {
            "state": state.value,
            "terminal": state.is_terminal,
            "spans": spans_written,
            "video_ids": [p[0] for p in pairs],
            "stages": [p[1].value for p in pairs],
        },
    )
