"""The work queue: batch claim, leases, retries, dead-letter, SIGTERM release.

Every test here needs a real Postgres. The behaviours under test — `SKIP LOCKED`
disjointness, lease expiry, attempt accounting — are the ones that fail only under
genuine contention, which is exactly why they are not unit tests with a fake.
"""

from __future__ import annotations

import pytest

from vl.domain.jobs import JobState, StageState
from vl.domain.stages import Stage
from vl.io.db import queue

pytestmark = pytest.mark.integration

WORKER_A = "worker-a"
WORKER_B = "worker-b"
MODEL = "whisper-large-v3@int8"


def seed(conn, count: int, stage: Stage = Stage.NORMALIZE) -> list[str]:
    """Register `count` videos and make `stage` claimable for each."""
    ids = []
    for i in range(count):
        video_id = f"vid{i:04d}"
        queue.register_video(conn, video_id=video_id, source_uri=f"file:///{video_id}.mp4")
        queue.enqueue(conn, video_id=video_id, stage=stage, model_id=MODEL)
        ids.append(video_id)
    return ids


# --- migrations and registration -------------------------------------------------


def test_migrations_are_idempotent(conn) -> None:
    from vl.io.db.migrate import apply_migrations

    apply_migrations(conn)  # already applied by the fixture
    apply_migrations(conn)


def test_duplicate_videos_are_registered_once(conn) -> None:
    """Reposted video is endemic on social platforms; the same bytes are one row."""
    assert queue.register_video(conn, video_id="abc", source_uri="file:///a.mp4") is True
    assert queue.register_video(conn, video_id="abc", source_uri="file:///b.mp4") is False

    count = conn.execute("SELECT count(*) FROM videos").fetchone()[0]
    assert count == 1


def test_enqueue_is_idempotent_per_model(conn) -> None:
    """Idempotency key is (video_id, stage, model_id) — invariant 2.

    Re-enqueueing must not create a second unit of work, or a retried sweep would
    double the GPU bill.
    """
    queue.register_video(conn, video_id="abc", source_uri="file:///a.mp4")
    queue.enqueue(conn, video_id="abc", stage=Stage.ASR, model_id=MODEL)
    queue.enqueue(conn, video_id="abc", stage=Stage.ASR, model_id=MODEL)

    count = conn.execute("SELECT count(*) FROM jobs").fetchone()[0]
    assert count == 1


def test_a_different_model_is_a_different_unit_of_work(conn) -> None:
    """Re-quantising a model must be able to re-extract without fighting the old row."""
    queue.register_video(conn, video_id="abc", source_uri="file:///a.mp4")
    queue.enqueue(conn, video_id="abc", stage=Stage.ASR, model_id="whisper@fp16")
    queue.enqueue(conn, video_id="abc", stage=Stage.ASR, model_id="whisper@int8")

    count = conn.execute("SELECT count(*) FROM jobs").fetchone()[0]
    assert count == 2


# --- batch claim -----------------------------------------------------------------


def test_claim_returns_at_most_the_batch_size(conn) -> None:
    """Batched because model load is 30-60s and must be amortised, not paid per video."""
    seed(conn, 10)
    claimed = queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=4)
    assert len(claimed) == 4


def test_claim_returns_everything_available_when_under_batch_size(conn) -> None:
    seed(conn, 3)
    claimed = queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=200)
    assert len(claimed) == 3


def test_claim_is_scoped_to_one_stage(conn) -> None:
    """Pools are stage-specific; a cpu worker must never be handed VLM work."""
    seed(conn, 2, stage=Stage.NORMALIZE)
    assert queue.claim_batch(conn, stage=Stage.VLM, worker_id=WORKER_A, batch_size=10) == []


def test_concurrent_workers_claim_disjoint_batches(conn, second_conn) -> None:
    """The core guarantee. Overlap here means two GPUs doing identical work.

    Both claims run inside *open* transactions on purpose. Under autocommit the first
    claim would commit and release its row locks before the second began, so the
    claims would be sequential and `SKIP LOCKED` would never be exercised — the test
    would pass whether or not the locking were correct.

    With the transactions overlapping, a missing `SKIP LOCKED` shows up as either
    overlapping batches (no locking) or a hang (plain `FOR UPDATE`).
    """
    seed(conn, 10)

    with conn.transaction(), second_conn.transaction():
        a = queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=5)
        b = queue.claim_batch(second_conn, stage=Stage.NORMALIZE, worker_id=WORKER_B, batch_size=5)

        ids_a = {c.video_id for c in a}
        ids_b = {c.video_id for c in b}

    assert len(ids_a) == 5
    assert len(ids_b) == 5, "second worker starved — SKIP LOCKED is not doing its job"
    assert ids_a.isdisjoint(ids_b)


def test_claimed_work_is_invisible_to_later_claims(conn) -> None:
    seed(conn, 4)
    queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=4)
    assert queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_B, batch_size=4) == []


def test_claim_marks_the_stage_running(conn) -> None:
    seed(conn, 1)
    queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1)
    assert queue.stage_state(conn, "vid0000", Stage.NORMALIZE) is StageState.RUNNING


# --- leases ----------------------------------------------------------------------


def test_heartbeat_extends_a_lease(conn) -> None:
    seed(conn, 1)
    [claim] = queue.claim_batch(
        conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1, lease_s=10
    )
    before = queue.lease_until(conn, claim.job_id)
    queue.heartbeat(conn, job_ids=[claim.job_id], worker_id=WORKER_A, lease_s=600)
    assert queue.lease_until(conn, claim.job_id) > before


def test_heartbeat_from_the_wrong_worker_is_ignored(conn) -> None:
    """A worker that lost its lease must not be able to reclaim it by heartbeating."""
    seed(conn, 1)
    [claim] = queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1)
    assert queue.heartbeat(conn, job_ids=[claim.job_id], worker_id=WORKER_B, lease_s=600) == 0


def test_expired_lease_is_reclaimed(conn) -> None:
    """A killed worker's work must come back, or the corpus stalls silently."""
    seed(conn, 1)
    queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1, lease_s=-1)
    assert queue.reclaim_expired(conn) == 1
    assert (
        len(queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_B, batch_size=1)) == 1
    )


def test_live_leases_are_left_alone(conn) -> None:
    seed(conn, 1)
    queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1, lease_s=3600)
    assert queue.reclaim_expired(conn) == 0


def test_reclaim_times_out_a_stage_once_attempts_are_exhausted(conn) -> None:
    """Bounded retries are what make the barrier reachable (invariant 5)."""
    seed(conn, 1)
    for _ in range(3):
        queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1, lease_s=-1)
        queue.reclaim_expired(conn, max_attempts=3)

    assert queue.stage_state(conn, "vid0000", Stage.NORMALIZE) is StageState.TIMED_OUT
    assert queue.job_state(conn, "vid0000", Stage.NORMALIZE) is JobState.DEAD


# --- failure and dead-letter -----------------------------------------------------


def test_failure_with_attempts_remaining_is_retried(conn) -> None:
    seed(conn, 1)
    [claim] = queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1)
    queue.fail(conn, job_id=claim.job_id, error="ffmpeg exited 1", max_attempts=3)

    assert queue.job_state(conn, "vid0000", Stage.NORMALIZE) is JobState.RETRY
    assert (
        len(queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1)) == 1
    )


def test_failure_with_attempts_exhausted_dead_letters(conn) -> None:
    seed(conn, 1)
    for _ in range(3):
        [claim] = queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1)
        queue.fail(conn, job_id=claim.job_id, error="ffmpeg exited 1", max_attempts=3)

    assert queue.job_state(conn, "vid0000", Stage.NORMALIZE) is JobState.DEAD
    assert queue.stage_state(conn, "vid0000", Stage.NORMALIZE) is StageState.FAILED_PERMANENT


def test_dead_letter_keeps_the_error(conn) -> None:
    """A failed batch must be diagnosable without re-running extraction."""
    seed(conn, 1)
    [claim] = queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1)
    queue.fail(conn, job_id=claim.job_id, error="moov atom not found", max_attempts=1)

    error = conn.execute("SELECT error FROM jobs WHERE video_id = 'vid0000'").fetchone()[0]
    assert "moov atom" in error


# --- SIGTERM release -------------------------------------------------------------


def test_release_returns_work_without_consuming_an_attempt(conn) -> None:
    """Spot preemption is not a failure of the work (invariant 16).

    A 200-video batch cannot finish inside a two-minute warning, so the worker
    releases its leases. Charging that to the retry budget would let infrastructure
    churn dead-letter perfectly good videos.
    """
    seed(conn, 1)
    [claim] = queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1)
    assert queue.attempts(conn, claim.job_id) == 1

    queue.release(conn, job_ids=[claim.job_id], worker_id=WORKER_A)

    assert queue.attempts(conn, claim.job_id) == 0
    assert queue.job_state(conn, "vid0000", Stage.NORMALIZE) is JobState.READY
    assert queue.stage_state(conn, "vid0000", Stage.NORMALIZE) is StageState.PENDING


def test_release_only_affects_the_holding_worker(conn) -> None:
    seed(conn, 1)
    [claim] = queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1)
    assert queue.release(conn, job_ids=[claim.job_id], worker_id=WORKER_B) == 0


# --- completion and fan-out ------------------------------------------------------


def test_completion_settles_the_stage_and_enqueues_dependents(conn) -> None:
    seed(conn, 1)
    [claim] = queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1)
    queue.complete(conn, job_id=claim.job_id, spans_written=0)

    assert queue.stage_state(conn, "vid0000", Stage.NORMALIZE) is StageState.DONE
    # ASR depends only on normalize, so it becomes claimable immediately.
    assert len(queue.claim_batch(conn, stage=Stage.ASR, worker_id=WORKER_A, batch_size=1)) == 1


def test_dependents_with_unmet_dependencies_are_not_claimable(conn) -> None:
    """Keyframes needs normalize AND asr settled; one is not enough."""
    seed(conn, 1)
    [claim] = queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1)
    queue.complete(conn, job_id=claim.job_id, spans_written=0)

    assert queue.claim_batch(conn, stage=Stage.KEYFRAMES, worker_id=WORKER_A, batch_size=1) == []


def test_a_permanently_failed_dependency_still_unblocks_dependents(conn) -> None:
    """Soft dependencies, end to end: ASR dying must not strand keyframe selection."""
    seed(conn, 1)
    [norm] = queue.claim_batch(conn, stage=Stage.NORMALIZE, worker_id=WORKER_A, batch_size=1)
    queue.complete(conn, job_id=norm.job_id, spans_written=0)

    [asr] = queue.claim_batch(conn, stage=Stage.ASR, worker_id=WORKER_A, batch_size=1)
    queue.fail(conn, job_id=asr.job_id, error="no audio stream", max_attempts=1)

    claimed = queue.claim_batch(conn, stage=Stage.KEYFRAMES, worker_id=WORKER_A, batch_size=1)
    assert len(claimed) == 1


def test_queue_depth_is_reported_per_stage(conn) -> None:
    """Backpressure signal: a growing VLM depth while ASR drains means undersized pool."""
    seed(conn, 7)
    depth = queue.queue_depth(conn)
    assert depth[Stage.NORMALIZE] == 7
    assert depth.get(Stage.VLM, 0) == 0
