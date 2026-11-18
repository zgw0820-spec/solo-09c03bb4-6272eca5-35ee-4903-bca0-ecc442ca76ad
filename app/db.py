"""SQLite schema and connection helpers.

One database file on a shared Docker volume, mounted into the API and both
worker containers.  Keeping jobs, checkpoints, leases and heartbeats in one
file means a checkpoint commit and the lease check that authorises it are a
single atomic ``BEGIN IMMEDIATE`` transaction.

SQLite runs in WAL mode with ``synchronous=FULL``: a committed batch survives a
process crash and a container restart.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    feature_dim     INTEGER NOT NULL,
    epochs          INTEGER NOT NULL,
    batch_size      INTEGER NOT NULL,
    record_count    INTEGER NOT NULL,
    total_batches   INTEGER NOT NULL,
    learning_rate   REAL NOT NULL,
    momentum        REAL NOT NULL,
    features_json   TEXT NOT NULL,
    labels_json     TEXT NOT NULL,
    weights_json    TEXT NOT NULL,
    velocity_json   TEXT NOT NULL,
    loss            REAL NOT NULL,
    completed_steps INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    finished_at     REAL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    job_id        TEXT NOT NULL,
    step          INTEGER NOT NULL,
    weights_json  TEXT NOT NULL,
    velocity_json TEXT NOT NULL,
    loss          REAL NOT NULL,
    batch_loss    REAL NOT NULL,
    committed_at  REAL NOT NULL,
    worker_id     TEXT NOT NULL,
    lease_epoch   INTEGER NOT NULL,
    PRIMARY KEY (job_id, step),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS job_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT NOT NULL,
    event      TEXT NOT NULL,
    detail     TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS workers (
    worker_id    TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    started_at   REAL NOT NULL,
    heartbeat_at REAL NOT NULL,
    -- Wall-clock heartbeat_at stays for display only; liveness is decided by
    -- boot_id + mono_at so host clock adjustments cannot expire a live worker.
    boot_id      TEXT NOT NULL DEFAULT '',
    mono_at      REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS leases (
    job_id       TEXT PRIMARY KEY,
    worker_id    TEXT NOT NULL,
    epoch        INTEGER NOT NULL,
    heartbeat_at REAL NOT NULL,
    -- Boot cycle in which the heartbeat was written, plus a monotonic-clock
    -- timestamp.  A different boot_id means the heartbeat predates a reboot
    -- and is stale at once; otherwise expiry is measured on the monotonic
    -- clock, which never jumps when the host time is corrected.
    boot_id      TEXT NOT NULL DEFAULT '',
    mono_at      REAL NOT NULL DEFAULT 0
);

-- Monotonic per-job lease counter.  Leases themselves are deleted on release,
-- but the counter is never reset, so a *new* lease can never reuse an epoch a
-- previous holder still holds a reference to.
CREATE TABLE IF NOT EXISTS lease_epochs (
    job_id TEXT PRIMARY KEY,
    epoch  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_job ON job_events(job_id, id);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(path: Path | str) -> Iterator[sqlite3.Connection]:
    """A write transaction that takes the write lock immediately.

    ``BEGIN IMMEDIATE`` makes writers in separate containers queue on the busy
    timeout instead of failing on a deferred-to-exclusive lock upgrade.
    """
    conn = connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        conn.close()


@contextmanager
def reader(path: Path | str) -> Iterator[sqlite3.Connection]:
    """A consistent read snapshot (WAL readers do not block writers)."""
    conn = connect(path)
    try:
        conn.execute("BEGIN")
        try:
            yield conn
        finally:
            conn.execute("COMMIT")
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add monotonic-timing columns to databases created by older versions.

    Rows written before the upgrade carry an empty ``boot_id`` / zero
    ``mono_at``; the staleness predicate treats them as expired immediately,
    so a leftover lease from the old wall-clock implementation can never block
    a takeover after the upgrade.
    """
    for table in ("leases", "workers"):
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if "boot_id" not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN boot_id TEXT NOT NULL DEFAULT ''")
        if "mono_at" not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN mono_at REAL NOT NULL DEFAULT 0")


def init_db(path: Path | str) -> None:
    """Create the schema if missing and migrate older databases in place.

    ``executescript`` commits implicitly, so it must not run inside our
    hand-managed transaction; concurrent starts serialise on SQLite's schema
    lock and every statement is idempotent.
    """
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
    finally:
        conn.close()
