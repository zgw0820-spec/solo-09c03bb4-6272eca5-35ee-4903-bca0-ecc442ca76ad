"""Clock primitives that are safe across host time adjustments and reboots.

Lease timing must never depend on the wall clock (``time.time()``): an operator
correcting a skewed host clock can jump it backwards or forwards by hours.  A
backward jump would keep a dead holder's lease "fresh" for hours and block
takeover; a forward jump would make a live, training holder look instantly
stale and get it fenced out mid-job.

Two primitives replace wall time for all lease decisions:

* :func:`mono` wraps :func:`time.monotonic`, a clock that only ever advances
  and is never touched by ``settimeofday``/NTP step corrections.  Elapsed-time
  comparisons inside one boot are therefore immune to clock adjustment in
  either direction.
* :func:`boot_id` returns the kernel boot identifier.  It is constant for the
  whole uptime of a host and changes on every reboot.  Every container shares
  the host kernel, so the API and both workers observe the same value.  A lease
  heartbeat carrying a different ``boot_id`` was written by a previous boot and
  is stale immediately, no matter what its timestamps say - this is what keeps
  a job from stalling behind a lease left over from a restart.

The ``TRAINING_BOOT_ID`` environment variable overrides the kernel value.  It
exists for deterministic tests of the reboot case (a real reboot cannot be
simulated inside one test process) and is not needed in normal operation.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
BOOT_ID_ENV = "TRAINING_BOOT_ID"

_cached_boot_id: str | None = None
_fallback_boot_id: str | None = None


def mono() -> float:
    """Monotonic elapsed-time clock; unaffected by wall-clock adjustments."""
    return time.monotonic()


def wall() -> float:
    """Wall-clock time, for human display and audit only.

    Never use this for lease expiry: it can jump in both directions when the
    host clock is corrected.
    """
    return time.time()


def boot_id() -> str:
    """Stable identifier of the current host boot cycle.

    Constant within one boot, different after every reboot.  An explicit
    environment override always wins so tests can model a reboot by changing
    it.  On platforms without ``/proc`` a per-process random token is used,
    which still makes stale cross-process leases fail closed.
    """
    override = os.environ.get(BOOT_ID_ENV)
    if override is not None:
        return override
    global _cached_boot_id, _fallback_boot_id
    if _cached_boot_id is not None:
        return _cached_boot_id
    try:
        value = BOOT_ID_PATH.read_text().strip()
    except OSError:
        value = ""
    if not value:
        if _fallback_boot_id is None:
            _fallback_boot_id = f"fallback-{uuid.uuid4().hex}"
        return _fallback_boot_id
    _cached_boot_id = value
    return value


def reset_caches() -> None:
    """Forget cached boot identity. Test-only: forces re-reading the source."""
    global _cached_boot_id, _fallback_boot_id
    _cached_boot_id = None
    _fallback_boot_id = None
