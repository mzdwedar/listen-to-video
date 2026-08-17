"""Shared fixtures.

The database fixtures use a real Postgres in a container (invariant 17). SQLite cannot
stand in: `FOR UPDATE SKIP LOCKED`, declarative partitioning and pgvector are the
things under test, and none of them exist there.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

# pgvector rather than plain postgres — later slices index embeddings in this database,
# and discovering the extension is missing then is a worse time to find out.
POSTGRES_IMAGE = "pgvector/pgvector:pg16"


@pytest.fixture(scope="session")
def postgres_dsn() -> Iterator[str]:
    """A throwaway Postgres for the session. Skips cleanly when Docker is unavailable."""
    docker = pytest.importorskip("docker", reason="docker SDK not installed")
    try:
        docker.from_env().ping()
    except Exception as exc:  # noqa: BLE001 - any docker failure means skip, not fail
        pytest.skip(f"Docker is not available: {exc}")

    try:  # package layout moved; support both
        from testcontainers.community.postgres import PostgresContainer
    except ImportError:
        from testcontainers.postgres import PostgresContainer

    with PostgresContainer(POSTGRES_IMAGE, driver=None) as container:
        yield container.get_connection_url()


@pytest.fixture
def conn(postgres_dsn: str) -> Iterator[psycopg.Connection]:  # type: ignore[name-defined] # noqa: F821
    """A connection to a freshly migrated, empty database.

    Truncating between tests rather than recreating the schema keeps the suite quick
    while still isolating each test.
    """
    import psycopg

    from vl.io.db.migrate import apply_migrations

    with psycopg.connect(postgres_dsn, autocommit=True) as connection:
        apply_migrations(connection)
        connection.execute("TRUNCATE videos, video_stages, jobs CASCADE")
        yield connection


@pytest.fixture
def second_conn(postgres_dsn: str) -> Iterator[psycopg.Connection]:  # type: ignore[name-defined] # noqa: F821
    """A second, independent connection — needed to test real lock contention.

    Two workers claiming from one pool is the behaviour that matters; a single
    connection cannot exercise `SKIP LOCKED` at all.

    `lock_timeout` is set so that a regression which makes the queue *block* under
    contention — dropping `SKIP LOCKED` and leaving a bare `FOR UPDATE`, say — fails
    in seconds with a clear error instead of hanging the suite indefinitely.
    """
    import psycopg

    with psycopg.connect(
        postgres_dsn, autocommit=True, options="-c lock_timeout=5000"
    ) as connection:
        yield connection
