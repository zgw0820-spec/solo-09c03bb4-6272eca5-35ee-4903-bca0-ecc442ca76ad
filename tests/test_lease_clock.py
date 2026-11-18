"""Lease timing tests: host clock correction and host reboot.

The production lease decision must be driven by a monotonic clock scoped to a
host boot:

* a wall-clock jump (the on-call engineer correcting the time) must neither
  expire a live holder's lease early (forward jump) nor keep a dead holder's
  lease alive (backward jump);
* a host reboot changes the kernel boot id, so any heartbeat written by the
  previous boot is stale immediately and the surviving worker resumes from the
  last committed checkpoint without waiting for the nominal lease term;
* the ``/workers`` view must report exactly the staleness a takeover attempt
  uses.

Real ``settimeofday`` cannot be exercised from a test process, so the clock
facade in :mod:`app.clock` is driven through a fake; the monotonic clock still
advances via real sleeps where genuine elapsed time is required.
"""
from __future__ import annotations

import numpy as np
import pytest

from app import clock, lease as lm, store
from app.config import Settings
from app.db import init_db


class FakeClock:
    """Deterministic stand-in for app.clock with independent wall/boot state."""

    def __init__(self, mono_start: float = 1000.0, wall_start: float = 1_700_000_000.0):
        self._mono = mono_start
        self._wall = wall_start
        self._boot = "boot-A"

    def mono(self) -> float:
        return self._mono

    def wall(self) -> float:
        return self._wall

    def boot_id(self) -> str:
        return self._boot

    def advance_mono(self, seconds: float) -> None:
        self._mono += seconds

    def jump_wall(self, seconds: float) -> None:
        self._wall += seconds

    def reboot(self, new_boot: str = "boot-B") -> None:
        self._boot = new_boot


@pytest.fixture()
def fake(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(clock, "mono", fake.mono)
    monkeypatch.setattr(clock, "wall", fake.wall)
    monkeypatch.setattr(clock, "boot_id", fake.boot_id)
    return fake


@pytest.fixture()
def settings(tmp_path):
    s = Settings(
        db_path=tmp_path / "training.db",
        lease_seconds=0.6,
        heartbeat_interval=0.1,
        poll_interval=0.01,
        batch_delay_seconds=0.0,
    )
    init_db(s.db_path)
    return s


def sample(records: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 3)).tolist()
    labels = [int(i) % 2 for i in range(records)]
    return features, labels


def _two_workers(settings):
    lm.register_worker(settings.db_path, "worker-a")
    lm.register_worker(settings.db_path, "worker-b")


def test_forward_wall_jump_does_not_expire_a_live_lease(settings, fake):
    job = store.create_job(settings.db_path, *sample(), 3)
    _two_workers(settings)
    lease = lm.acquire(settings.db_path, job["job_id"], "worker-a", settings.lease_seconds)
    assert lease is not None and lease.epoch == 1

    # Operator jumps the wall clock forward by three hours.  Almost no real
    # (monotonic) time has passed, so the lease must remain owned by worker-a.
    fake.jump_wall(3 * 3600)
    assert lm.acquire(settings.db_path, job["job_id"], "worker-b", settings.lease_seconds) is None
    assert lm.renew(settings.db_path, lease, settings.lease_seconds) is True
    # And the holder can still commit checkpoints.
    result = store.advance_job(settings.db_path, lease)
    assert result["step"] == 1


def test_backward_wall_jump_does_not_keep_a_dead_holders_lease_fresh(settings, fake):
    job = store.create_job(settings.db_path, *sample(), 3)
    _two_workers(settings)
    lease = lm.acquire(settings.db_path, job["job_id"], "worker-a", settings.lease_seconds)
    store.advance_job(settings.db_path, lease)

    # Let the full lease term elapse in real (monotonic) time: worker-a is dead
    # and stops heartbeating.  Then correct the wall clock backwards by a day.
    fake.advance_mono(settings.lease_seconds + 0.1)
    fake.jump_wall(-24 * 3600)

    # The stale lease must still be reclaimable now, not in ~24 hours.
    taken = lm.acquire(settings.db_path, job["job_id"], "worker-b", settings.lease_seconds)
    assert taken is not None
    assert taken.epoch == lease.epoch + 1


def test_lease_expires_on_monotonic_elapsed_time_within_one_boot(settings, fake):
    job = store.create_job(settings.db_path, *sample(), 3)
    _two_workers(settings)
    lease = lm.acquire(settings.db_path, job["job_id"], "worker-a", settings.lease_seconds)

    # Before the term, even with a big forward wall jump, nothing is reclaimable.
    fake.advance_mono(settings.lease_seconds - 0.2)
    fake.jump_wall(3600)
    assert lm.acquire(settings.db_path, job["job_id"], "worker-b", settings.lease_seconds) is None

    # After the term of monotonic time, takeover succeeds with a fresh epoch.
    fake.advance_mono(0.3)
    taken = lm.acquire(settings.db_path, job["job_id"], "worker-b", settings.lease_seconds)
    assert taken is not None and taken.epoch == lease.epoch + 1


def test_reboot_makes_leftover_lease_stale_immediately(settings, fake):
    job = store.create_job(settings.db_path, *sample(), 3)
    _two_workers(settings)
    old = lm.acquire(settings.db_path, job["job_id"], "worker-a", settings.lease_seconds)
    for _ in range(3):
        store.advance_job(settings.db_path, old)
    checkpoint = store.get_job(settings.db_path, job["job_id"])
    assert checkpoint["completed_steps"] == 3

    # Host reboots: kernel boot id changes and monotonic time restarts low.
    fake.reboot("boot-B")
    fake._mono = 5.0

    # No waiting for lease_seconds: the previous boot's lease is stale at once.
    new = lm.acquire(settings.db_path, job["job_id"], "worker-b", settings.lease_seconds)
    assert new is not None
    assert new.epoch == old.epoch + 1

    # The old holder (a process from the previous boot) is fenced out and can
    # neither write nor roll the checkpoint back.
    with pytest.raises(lm.LeaseLost):
        store.advance_job(settings.db_path, old)
    assert store.get_job(settings.db_path, job["job_id"])["completed_steps"] == 3

    # The survivor resumes exactly from the last confirmed checkpoint and
    # finishes the job.
    while True:
        result = store.advance_job(settings.db_path, new)
        if result["status"] == "completed":
            break
    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    assert final["completed_steps"] == final["total_steps"]
    steps = [c["step"] for c in store.list_checkpoints(settings.db_path, job["job_id"])]
    assert steps == list(range(1, final["total_steps"] + 1))


def test_reboot_lets_same_worker_id_reclaim_with_a_new_epoch(settings, fake):
    """A worker container restarted under the same id must not trust its old lease."""
    job = store.create_job(settings.db_path, *sample(records=16), 2)
    lm.register_worker(settings.db_path, "worker-1")
    old = lm.acquire(settings.db_path, job["job_id"], "worker-1", settings.lease_seconds)
    store.advance_job(settings.db_path, old)

    fake.reboot("boot-C")
    fake._mono = 1.0
    lm.register_worker(settings.db_path, "worker-1")  # same id, new boot
    new = lm.acquire(settings.db_path, job["job_id"], "worker-1", settings.lease_seconds)
    assert new is not None and new.epoch == old.epoch + 1
    with pytest.raises(lm.LeaseLost):
        store.advance_job(settings.db_path, old)


def test_workers_endpoint_matches_actual_takeover(settings, fake, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main as main_module

    cfg = Settings(
        db_path=settings.db_path,
        lease_seconds=settings.lease_seconds,
        heartbeat_interval=0.1,
        poll_interval=0.01,
        batch_delay_seconds=0.0,
    )
    monkeypatch.setattr(main_module, "settings", cfg)
    with TestClient(main_module.app) as client:
        job = client.post(
            "/jobs",
            json={
                "features": np.random.default_rng(0).normal(size=(16, 2)).tolist(),
                "labels": [i % 2 for i in range(16)],
                "epochs": 2,
            },
        ).json()
        _two_workers(settings)
        lease = lm.acquire(settings.db_path, job["job_id"], "worker-a", settings.lease_seconds)

        def lease_view():
            leases = client.get("/workers").json()["leases"]
            assert len(leases) == 1
            return leases[0]

        view = lease_view()
        assert view["expired"] is False
        assert view["same_boot"] is True
        assert view["worker_id"] == "worker-a"

        # Forward wall jump: displayed state must still say "not expired", and
        # an actual takeover attempt must agree and fail.
        fake.jump_wall(5 * 3600)
        assert lease_view()["expired"] is False
        assert (
            lm.acquire(settings.db_path, job["job_id"], "worker-b", settings.lease_seconds)
            is None
        )

        # Monotonic expiry: display and takeover must flip together.
        fake.advance_mono(settings.lease_seconds + 0.1)
        assert lease_view()["expired"] is True
        taken = lm.acquire(settings.db_path, job["job_id"], "worker-b", settings.lease_seconds)
        assert taken is not None and taken.epoch == lease.epoch + 1
        assert lease_view()["worker_id"] == "worker-b"
        assert lease_view()["expired"] is False

        # Reboot: a heartbeat from the old boot is shown and treated as stale
        # immediately even though monotonic time restarts near zero.
        fake.reboot("boot-D")
        fake._mono = 2.0
        view = lease_view()
        assert view["same_boot"] is False
        assert view["expired"] is True
        assert (
            lm.acquire(settings.db_path, job["job_id"], "worker-a", settings.lease_seconds)
            is not None
        )


def test_legacy_wallclock_only_rows_are_treated_as_stale(settings):
    """A database upgraded from the old release must not stall on old leases."""
    import sqlite3

    job = store.create_job(settings.db_path, *sample(), 2)
    lm.register_worker(settings.db_path, "old")
    # Hand-write a row in the pre-fix shape: empty boot_id, zero mono_at, and a
    # wall timestamp far in the future, which the old code would call fresh.
    with sqlite3.connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO lease_epochs(job_id, epoch) VALUES(?, 1)", (job["job_id"],)
        )
        conn.execute(
            "INSERT INTO leases(job_id, worker_id, epoch, heartbeat_at, boot_id, mono_at) "
            "VALUES(?, 'old', 1, 9999999999.0, '', 0)",
            (job["job_id"],),
        )
    lm.register_worker(settings.db_path, "new")
    taken = lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds)
    assert taken is not None and taken.epoch == 2


def test_schema_migration_adds_timing_columns_to_old_database(tmp_path):
    """init_db upgrades an old wall-clock schema without losing existing rows."""
    import sqlite3

    db_path = tmp_path / "legacy.db"
    # Build a database with the original, pre-fix schema.
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE workers (
            worker_id TEXT PRIMARY KEY, status TEXT NOT NULL,
            started_at REAL NOT NULL, heartbeat_at REAL NOT NULL
        );
        CREATE TABLE leases (
            job_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL,
            epoch INTEGER NOT NULL, heartbeat_at REAL NOT NULL
        );
        CREATE TABLE lease_epochs (job_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL);
        INSERT INTO workers VALUES ('w1', 'training', 1.0, 2.0);
        INSERT INTO lease_epochs VALUES ('job1', 1);
        INSERT INTO leases VALUES ('job1', 'w1', 1, 2.0);
        """
    )
    conn.commit()
    conn.close()

    init_db(db_path)  # must migrate, not fail

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    worker_cols = {r["name"] for r in conn.execute("PRAGMA table_info(workers)")}
    lease_cols = {r["name"] for r in conn.execute("PRAGMA table_info(leases)")}
    assert {"boot_id", "mono_at"} <= worker_cols
    assert {"boot_id", "mono_at"} <= lease_cols
    row = conn.execute("SELECT boot_id, mono_at FROM leases WHERE job_id='job1'").fetchone()
    assert row["boot_id"] == "" and row["mono_at"] == 0.0  # stale after upgrade
    conn.close()


def test_unknown_boot_fails_closed_when_procfs_is_missing(settings, monkeypatch):
    """Without a kernel boot id, each process uses its own token.

    A heartbeat written by another process can never be mistaken for the
    current boot, so a stale lease is always reclaimable rather than trusted.
    """
    monkeypatch.delenv("BOOT_ID_PLACEHOLDER", raising=False)
    real_read_text = __import__("pathlib").Path.read_text

    def missing_proc(self, *args, **kwargs):
        if str(self) == str(clock.BOOT_ID_PATH):
            raise OSError("no procfs")
        return real_read_text(self, *args, **kwargs)

    clock.reset_caches()
    monkeypatch.setattr(__import__("pathlib").Path, "read_text", missing_proc)

    job = store.create_job(settings.db_path, *sample(records=16), 2)
    lm.register_worker(settings.db_path, "a")
    lm.register_worker(settings.db_path, "b")
    first = lm.acquire(settings.db_path, job["job_id"], "a", settings.lease_seconds)
    assert first is not None
    # Simulate another process/boot: a distinct fallback token, same host.
    token = clock.boot_id()
    assert token.startswith("fallback-")
    monkeypatch.setenv(clock.BOOT_ID_ENV, "other-process-boot")
    assert lm.acquire(settings.db_path, job["job_id"], "b", settings.lease_seconds) is not None
    with pytest.raises(lm.LeaseLost):
        store.advance_job(settings.db_path, first)
    clock.reset_caches()
