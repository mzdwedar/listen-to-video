-- Videos, the fan-in barrier, and the work queue.
--
-- Evidence and interpretation tables land in a later migration, alongside the code
-- that writes them. Creating them now would mean guessing at columns for tables
-- nothing reads.

CREATE TABLE IF NOT EXISTS videos (
    -- Content hash, not a surrogate key: the same bytes must be one row no matter how
    -- many times they are reposted, and reposting is endemic on social platforms.
    id                  TEXT PRIMARY KEY,
    source_uri          TEXT        NOT NULL,
    platform            TEXT,
    duration_s          DOUBLE PRECISION,
    -- Canonical frame rate after normalise forces CFR. Every timestamp in the system
    -- is relative to this, never to the source container (invariant 4).
    fps                 DOUBLE PRECISION,
    language            TEXT,
    published_at        TIMESTAMPTZ,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Perceptual hash catches near-duplicates that content hashing misses: the same
    -- video re-encoded, watermarked, or trimmed by a frame.
    phash               BIGINT,
    dedup_of            TEXT        REFERENCES videos (id),
    -- Bumped on any new evidence write. Fusion records which generation it consumed,
    -- so a stage that succeeds on repair hours later triggers re-interpretation of
    -- just that video, with no GPU re-extraction.
    evidence_generation INT         NOT NULL DEFAULT 0,
    repair_state        TEXT        NOT NULL DEFAULT 'none',
    repair_round        INT         NOT NULL DEFAULT 0,
    next_repair_at      TIMESTAMPTZ,
    -- Set when the text requirement is first satisfied. On expiry the video fuses
    -- provisionally with whatever exists rather than blocking the corpus.
    grace_expires_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS videos_phash_idx ON videos (phash) WHERE phash IS NOT NULL;
CREATE INDEX IF NOT EXISTS videos_repair_idx ON videos (next_repair_at)
    WHERE repair_state = 'needs_repair';


-- The fan-in barrier. One row per (video, stage), holding whether that stage is
-- settled. Separate from `jobs` because a job tracks one *attempt* under lease while
-- this tracks whether the stage is decided — and the barrier needs the latter.
CREATE TABLE IF NOT EXISTS video_stages (
    video_id      TEXT NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
    stage         TEXT NOT NULL,
    state         TEXT NOT NULL DEFAULT 'pending',
    terminal_at   TIMESTAMPTZ,
    spans_written INT  NOT NULL DEFAULT 0,
    PRIMARY KEY (video_id, stage)
);


CREATE TABLE IF NOT EXISTS jobs (
    id          BIGSERIAL PRIMARY KEY,
    video_id    TEXT        NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
    stage       TEXT        NOT NULL,
    -- Part of the idempotency key so re-quantising a model is a distinct unit of work
    -- rather than a conflict with the old one.
    model_id    TEXT        NOT NULL,
    state       TEXT        NOT NULL DEFAULT 'ready',
    attempts    INT         NOT NULL DEFAULT 0,
    lease_until TIMESTAMPTZ,
    worker_id   TEXT,
    -- Retained on dead-letter with the artifact URIs, so a failed batch is
    -- diagnosable without re-running extraction.
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Invariant 2: (video_id, stage, model_id) is the idempotency key. Without this,
    -- a retried sweep doubles the GPU bill.
    UNIQUE (video_id, stage, model_id)
);

-- Supports the claim query: one stage, oldest first, only claimable rows.
CREATE INDEX IF NOT EXISTS jobs_claimable_idx ON jobs (stage, created_at)
    WHERE state IN ('ready', 'retry');

-- Supports the lease reaper.
CREATE INDEX IF NOT EXISTS jobs_lease_idx ON jobs (lease_until) WHERE state = 'leased';
