"""Where this fork's SIGNALS.lock.json lives, and the one call every AXIS signal loader makes.

`axis_signals_lock.py` beside this file is a BYTE COPY of the AXIS-QVLA repository's
axis_qvla/core/signals_lock.py (the release gate compares the sha256s) and is never edited
here; this module is the fork-side glue, so the lock is resolved in exactly one place. The guard
only REFUSES: it never changes an array a loader returns.
"""

from __future__ import annotations

import os
from pathlib import Path

from lerobot.utils import axis_signals_lock as _lock

LOCK_PATH = Path(__file__).with_name("SIGNALS.lock.json")


def guard(path: str | os.PathLike[str], *, kind: str, rowspace_key: object) -> dict | None:
    """The locked row for these bytes, None if the lock does not know them, or SignalLockError."""
    return _lock.guard_signal(path, kind=kind, rowspace_key=rowspace_key, lock_path=LOCK_PATH)
