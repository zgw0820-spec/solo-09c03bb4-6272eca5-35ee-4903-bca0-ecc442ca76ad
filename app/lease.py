"""Lease ownership in the shared database.

Exactly one worker may hold a job's lease.  A lease is only taken over when its
holder has not renewed it for ``lease_seconds`` of *monotonic* time within the
same host boot.  Ownership changes increment the lease ``epoch``, and a
worker's identity is the pair ``(worker_id, epoch)``.

Timing model
------------

Every heartbeat records two timing columns:

* ``boot_id`` - the kernel boot identifier of the host that wrote it;
* ``mono_at`` - a :func:`time.monotonic` reading at write time.

A held lease is stale when either:

1. the heartbeat's ``boot_id`` differs from the current boot, meaning the
   holder wrote it before a host reboot - the lease is reclaimable instantly so
   a job never stalls behind a lease a restarted host left behind; or
2. the boot matches but ``monotonic() - mono_at >= lease_seconds`` of real
   elapsed time.  Because the monotonic clock is unaffected by wall-clock
   correction, neither a backward nor a forward time jump can shorten or extend
   a live holder's lease within the same boot.

Wall-clock ``heartbeat_at`` is still stored for display and auditing, but it
never participates in an expiry decision.

The epoch is the fence for checkpoint commits: ``advance_job`` verifies it
inside the same ``BEGIN IMMEDIATE`` transaction that writes the checkpoint.  A
worker whose lease was stolen therefore cannot write anything, even if its
process is alive and still believes it owns the job - it cannot roll the
breakpoint back, and it cannot report a false completion.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import clock
from .db import connect, transaction


@dataclass(frozen=True)
class Lease:
    job_id: str
    worker_id: str
    epoch: int


class LeaseLost(Exception):
    """This worker is no longer allowed to train the job."""


def _now() -> tuple[float, float, str]:
    """Return (wall time, monotonic time, current boot id) for a heartbeat."""
    return clock.wall(), clock.mono(), clock.boot_id()


def is_stale(row: sqlite3.Row, lease_seconds: float) -> bool:
    """Whether a lease/worker heartbeat row is currently expired.

    This is the single predicate used both for takeover decisions and for the
    ``/workers`` API, so the displayed expiry always matches what a worker is
    actually allowed to do.  A row from a previous boot (including one written
    by an older wall-clock-only release, whose ``boot_id`` is empty) is stale
    immediately; same-boot rows expire on monotonic elapsed time only.
    """
    if row["boot_id"] != clock.boot_id():
        return True
    return clock.mono() - float(row["mono_at"]) >= lease_seconds


def register_worker(path: Path | str, worker_id: str) -> None:
    wall_now, mono_now, boot = _now()
    with transaction(path) as conn:
        conn.execute(
            """
            INSERT INTO workers(worker_id, status, started_at, heartbeat_at, boot_id, mono_at)
            VALUES(?, 'starting', ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                status='starting', started_at=excluded.started_at,
                heartbeat_at=excluded.heartbeat_at,
                boot_id=excluded.boot_id, mono_at=excluded.mono_at
            """,
            (worker_id, wall_now, wall_now, boot, mono_now),
        )


def heartbeat(path: Path | str, worker_id: str, status: str) -> None:
    wall_now, mono_now, boot = _now()
    with transaction(path) as conn:
        conn.execute(
            "UPDATE workers SET heartbeat_at=?, status=?, boot_id=?, mono_at=? "
            "WHERE worker_id=?",
            (wall_now, status, boot, mono_now, worker_id),
        )


def _allocate_epoch(conn: sqlite3.Connection, job_id: str) -> int:
    """Bump the job's monotonic lease counter and return the new epoch."""
    conn.execute(
        """
        INSERT INTO lease_epochs(job_id, epoch) VALUES(?, 1)
        ON CONFLICT(job_id) DO UPDATE SET epoch=lease_epochs.epoch + 1
        """,
        (job_id,),
    )
    row = conn.execute("SELECT epoch FROM lease_epochs WHERE job_id=?", (job_id,)).fetchone()
    return int(row["epoch"])


def acquire(
    path: Path | str, job_id: str, worker_id: str, lease_seconds: float
) -> Lease | None:
    """Try to own ``job_id``.

    Returns a lease when the job is unowned or its holder's heartbeat is stale.
    If this worker already owns it, the same epoch is renewed and returned, so a
    worker can call this on every poll.  ``None`` means another live worker owns
    the job.  Every new ownership takes a fresh, never-reused epoch.
    """
    wall_now, mono_now, boot = _now()
    with transaction(path) as conn:
        row = conn.execute("SELECT * FROM leases WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            epoch = _allocate_epoch(conn, job_id)
            conn.execute(
                "INSERT INTO leases(job_id, worker_id, epoch, heartbeat_at, boot_id, mono_at) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (job_id, worker_id, epoch, wall_now, boot, mono_now),
            )
            return Lease(job_id, worker_id, epoch)
        if row["worker_id"] == worker_id and not is_stale(row, lease_seconds):
            conn.execute(
                "UPDATE leases SET heartbeat_at=?, boot_id=?, mono_at=? WHERE job_id=?",
                (wall_now, boot, mono_now, job_id),
            )
            return Lease(job_id, worker_id, int(row["epoch"]))
        if not is_stale(row, lease_seconds):
            return None
        epoch = _allocate_epoch(conn, job_id)
        conn.execute(
            "UPDATE leases SET worker_id=?, epoch=?, heartbeat_at=?, boot_id=?, mono_at=? "
            "WHERE job_id=?",
            (worker_id, epoch, wall_now, boot, mono_now, job_id),
        )
        return Lease(job_id, worker_id, epoch)


def renew(path: Path | str, lease: Lease, lease_seconds: float) -> bool:
    """Renew if we still match the lease; ``False`` means it moved on.

    A lease that has become stale (silent for the lease term, or left over from
    a previous boot) cannot be renewed by its old holder even if the row still
    names that worker; only a fresh :func:`acquire` may reclaim it.
    """
    wall_now, mono_now, boot = _now()
    with transaction(path) as conn:
        row = conn.execute(
            "SELECT worker_id, epoch, boot_id, mono_at FROM leases WHERE job_id=?",
            (lease.job_id,),
        ).fetchone()
        if row is None or row["worker_id"] != lease.worker_id or int(row["epoch"]) != lease.epoch:
            return False
        if is_stale(row, lease_seconds):
            return False
        conn.execute(
            "UPDATE leases SET heartbeat_at=?, boot_id=?, mono_at=? WHERE job_id=?",
            (wall_now, boot, mono_now, lease.job_id),
        )
        return True


def owner_ok(conn: sqlite3.Connection, lease: Lease) -> bool:
    """Check ownership inside an already-open write transaction."""
    row = conn.execute(
        "SELECT worker_id, epoch FROM leases WHERE job_id=?", (lease.job_id,)
    ).fetchone()
    return bool(row) and row["worker_id"] == lease.worker_id and int(row["epoch"]) == lease.epoch


def release(path: Path | str, lease: Lease) -> None:
    """Give up a lease, but only if it is still ours.

    A stale caller's DELETE matches nothing, so release is always safe.
    """
    with transaction(path) as conn:
        conn.execute(
            "DELETE FROM leases WHERE job_id=? AND worker_id=? AND epoch=?",
            (lease.job_id, lease.worker_id, lease.epoch),
        )


def holder(path: Path | str, job_id: str) -> tuple[str, int, float] | None:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT worker_id, epoch, heartbeat_at FROM leases WHERE job_id=?", (job_id,)
        ).fetchone()
    if row is None:
        return None
    return (row["worker_id"], int(row["epoch"]), float(row["heartbeat_at"]))
