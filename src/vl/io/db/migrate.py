"""Numbered SQL migrations, applied in filename order.

Deliberately not Alembic. Migrations here do things Alembic's autogeneration handles
badly anyway — declarative partitioning, partial indexes, pgvector index types — so an
ORM-shaped migration tool would mostly be something to work around.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[3].parent / "migrations"

_TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def migration_files(directory: Path | None = None) -> list[Path]:
    """Migrations in application order.

    Filenames are zero-padded so lexical order is numeric order; a file named `10_x`
    sorting before `2_x` is a classic way to corrupt a schema.
    """
    return sorted((directory or MIGRATIONS_DIR).glob("*.sql"))


def apply_migrations(conn: psycopg.Connection, directory: Path | None = None) -> list[str]:
    """Apply any unapplied migrations. Returns the filenames applied this call.

    Idempotent, so it is safe to run on every worker start rather than needing a
    separate deploy step.
    """
    conn.execute(_TRACKING_TABLE)
    applied = {row[0] for row in conn.execute("SELECT filename FROM schema_migrations")}

    just_applied = []
    for path in migration_files(directory):
        if path.name in applied:
            continue
        # One transaction per migration: a failure leaves earlier migrations intact
        # rather than rolling back a partially-migrated schema into an unknown state.
        with conn.transaction():
            conn.execute(path.read_text())  # type: ignore[arg-type]
            conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES (%s)",
                (path.name,),
            )
        just_applied.append(path.name)

    return just_applied
