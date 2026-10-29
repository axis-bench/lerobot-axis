"""Refuse a quality-signal file whose bytes are not the ones a paper arm trained on.

CORE TIER: stdlib only, Python >= 3.10 syntax, no module-level paths. These exact bytes are
vendored into the LeRobot line (src/lerobot/utils/axis_signals_lock.py), and `release check`
compares the two sha256s. So the module never imports `axis` and never knows where a lock lives:
every caller passes it.

WHY. A quality-signal file is identified by its bytes alone: two generations of one file can share
dtype, n_rows, reward_id and even the AWR corpus fingerprint, so only a byte check tells them apart.
A locked file passes; other bytes at a locked published path (or at a legacy name, when the lock
lists legacy names) over the same row space are refused, and so is a copy the lock lists as
deprecated; a file the lock does not mention passes as "unlocked" (other corpora, fixtures), so
nothing outside the lock changes. A file at a locked published path must carry exactly THAT row's
bytes: another row's locked bytes there (v2 tags at the approach-aware tags path) are a silent arm
swap, refused.

CALLER CONTRACT. `kind` is what the caller will use the file as (one of KINDS). `rowspace_key`
is the n_rows (int) of the dataset the file is applied to, or the corpus fingerprint (str) the
caller computed. It is REQUIRED for every file the lock knows -- by sha256, by published path
(a path ending in a row's `path`) or by legacy name: None, or a key that is not that row
space's, is refused (fail closed). Only a file the lock does not know at all passes with None.
A missing or unreadable lock is a SignalLockError, never a bare OSError/ValueError.

LOCK FORMS. The published lock lists each file's identity, digests, kind, row space, consumers,
generator digests and trained arms (FILE_KEYS). A lock may also list `legacy_names` and
`reproduces` per file and a `deprecated` list of copies to refuse; each is checked when present.

HASH, THEN LOAD. Callers hash a path, then np.load/json.load it: the guarded files are immutable
published artifacts, so the bytes hashed are the bytes loaded, and every loader keeps its own
code path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import posixpath
import re
from collections.abc import Mapping

SCHEMA = "qvla.signals_lock/1"
KINDS: frozenset[str] = frozenset(
    {
        "quality_tags",
        "awr_weights",
        "schedule",
        "lerobot_rows",
        "lerobot_awr_flat",
        "lerobot_schedule",
        "rowspace",
    }
)
FILE_KEYS = (
    "id",
    "path",
    "kind",
    "reward_id",
    "granularity",
    "rowspace",
    "size",
    "sha256",
    "md5",
    "generator",
    "derived_from",
    "consumers",
    "trained",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX32 = re.compile(r"[0-9a-f]{32}")
_LOG = logging.getLogger(
    "qvla_signals"
)  # rename-invariant name, see test_guard_source_has_no_renameable_axis_token
_CACHE: dict[str, dict] = {}


class SignalLockError(RuntimeError):
    """A signal file is deprecated, mis-kinded, or a different generation of a locked file."""


def file_digests(path: str | os.PathLike[str]) -> tuple[str, str, int]:
    """(sha256, md5, size) in one read; md5 only because older provenance recorded md5."""
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


def verify_file(path: str | os.PathLike[str], expected_sha256: str) -> None:
    got = file_digests(path)[0]
    if got != expected_sha256:
        raise SignalLockError(f"{os.fspath(path)}: sha256 {got} != expected {expected_sha256}")


def _check_file_row(row: Mapping, spaces: Mapping, seen: set, p: list[str]) -> None:
    where = f"file {row['id']}"
    if row["id"] in seen:
        p.append(f"{where}: duplicate id")
    seen.add(row["id"])
    if row["kind"] not in KINDS:
        p.append(f"{where}: kind {row['kind']!r} is not one of {sorted(KINDS)}")
    if not _HEX64.fullmatch(str(row["sha256"])):
        p.append(f"{where}: sha256 must be 64 lowercase hex")
    if not _HEX32.fullmatch(str(row["md5"])):
        p.append(f"{where}: md5 must be 32 lowercase hex")
    if row["rowspace"] not in spaces:
        p.append(f"{where}: unknown rowspace {row['rowspace']!r}")
    if not isinstance(row["size"], int) or row["size"] < 0:
        p.append(f"{where}: size must be a non-negative int")
    names = row.get("legacy_names", [])
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        p.append(f"{where}: legacy_names must be a list of strings")
    if row.get("reproduces") not in (True, False, None):
        p.append(f"{where}: reproduces must be true, false or null")


def validate_lock(doc: Mapping[str, object]) -> list[str]:
    """Every problem found, as text; [] means the lock can be trusted by verify_signal."""
    if not isinstance(doc, Mapping):
        return ["lock is not a JSON object"]
    p: list[str] = []
    if doc.get("schema") != SCHEMA:
        p.append(f"schema is {doc.get('schema')!r}, expected {SCHEMA!r}")
    if not isinstance(doc.get("release"), str):
        p.append("release must be a string")
    spaces = doc.get("rowspaces")
    if not isinstance(spaces, Mapping) or not spaces:
        p.append("rowspaces must be a non-empty object")
        spaces = {}
    for name, sp in spaces.items():
        if not isinstance(sp, Mapping) or not isinstance(sp.get("n_rows"), int) or sp["n_rows"] <= 0:
            p.append(f"rowspace {name}: n_rows must be a positive int")
    files = doc.get("files")
    if not isinstance(files, list):
        p.append("files must be a list")
        files = []
    ids: set = set()
    shas: set = set()
    digests: set = set()
    for i, row in enumerate(files):
        if not isinstance(row, Mapping):
            p.append(f"files[{i}]: not an object")
            continue
        missing = [k for k in FILE_KEYS if k not in row]
        if missing:
            p.append(f"files[{i}]: missing {missing}")
            continue
        _check_file_row(row, spaces, ids, p)
        if row["sha256"] in shas:
            p.append(f"file {row['id']}: duplicate sha256")
        shas.add(row["sha256"])
        digests.update((row["sha256"], row["md5"]))
    deprecated = doc.get("deprecated", [])
    if not isinstance(deprecated, list):
        p.append("deprecated must be a list when present")
        deprecated = []
    for i, d in enumerate(deprecated):
        where = f"deprecated[{i}]"
        if not isinstance(d, Mapping):
            p.append(f"{where}: not an object")
            continue
        if not _HEX32.fullmatch(str(d.get("md5", ""))):
            p.append(f"{where}: md5 must be 32 lowercase hex")
        if "sha256" in d and not _HEX64.fullmatch(str(d["sha256"])):
            p.append(f"{where}: sha256 must be 64 lowercase hex")
        if not isinstance(d.get("legacy_name"), str):
            p.append(f"{where}: legacy_name must be a string")
        if d.get("superseded_by") is not None and d["superseded_by"] not in ids:
            p.append(f"{where}: superseded_by {d['superseded_by']!r} is not a locked file id")
        for key in ("sha256", "md5"):
            if d.get(key) in digests:
                p.append(f"{where}: {key} is also a locked file's digest")
    return p


def load_lock(path: str | os.PathLike[str]) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        raise SignalLockError(
            f"cannot read signal lock {os.fspath(path)}: {type(exc).__name__}: {exc}"
        ) from exc
    problems = validate_lock(doc)
    if problems:
        raise SignalLockError(f"invalid signal lock {os.fspath(path)}: {problems[:3]}")
    return doc


def verify_signal(
    path: str | os.PathLike[str], *, kind: str, rowspace_key: object, lock: Mapping[str, object]
) -> dict | None:
    """The locked row for these bytes, None for a file the lock does not know, or a refusal."""
    if kind not in KINDS:
        raise SignalLockError(f"unknown signal kind {kind!r}")
    shown = os.fspath(path)
    sha, md5, _ = file_digests(path)
    for d in lock.get("deprecated", ()):
        if d.get("sha256") == sha or d.get("md5") == md5:
            raise SignalLockError(
                f"{shown} is a deprecated {d.get('legacy_name')} (sha256 {sha[:12]}, md5 "
                f"{md5[:8]}): {d.get('reason')}. Use {d.get('superseded_by')} instead."
            )
    spaces, files = lock.get("rowspaces", {}), lock.get("files", ())
    name, posix = os.path.basename(shown), shown.replace(os.sep, posixpath.sep)
    at = [(r, how) for r in files if (how := _at(r, posix, name))]
    owner = next((r for r in files if r["sha256"] == sha), None)
    if owner is not None and at and all(r["id"] != owner["id"] for r, _ in at):
        raise SignalLockError(
            f"{shown} holds the locked bytes of {owner['id']} at the locked {at[0][1]} of "
            f"{', '.join(r['id'] for r, _ in at)}: a swapped file is refused"
        )
    if owner is not None:
        if owner["kind"] != kind:
            raise SignalLockError(
                f"{shown} is locked as {owner['id']} of kind {owner['kind']!r}, but is being "
                f"loaded as kind {kind!r}"
            )
        if rowspace_key is None or rowspace_key not in _keys(spaces, owner):
            raise SignalLockError(
                f"{shown} is locked as {owner['id']} over row space {owner['rowspace']!r}; "
                f"rowspace_key {rowspace_key!r} is not that row space's"
            )
        return dict(owner)
    for row, how in at:
        if rowspace_key is None:
            raise SignalLockError(
                f"{shown} has the locked {how} of {row['id']} but other bytes (sha256 "
                f"{sha[:12]}); a rowspace_key is required to tell it from another corpus's file"
            )
        if rowspace_key in _keys(spaces, row):
            raise SignalLockError(
                f"{shown} is a different generation of {row['id']}: same {how}, same row space, "
                f"sha256 {sha[:12]} is not the locked {row['sha256'][:12]}"
            )
    return None


def _at(row: Mapping, posix: str, name: str) -> str | None:
    """'path' if the file sits at the row's published path, 'name' at a legacy name, else None."""
    if posix == row["path"] or posix.endswith(posixpath.sep + row["path"]):
        return "path"
    return "name" if name in row.get("legacy_names", ()) else None


def _keys(spaces: Mapping, row: Mapping) -> set:
    space = spaces.get(row["rowspace"], {})
    return {space.get("n_rows"), space.get("corpus_fingerprint")} - {None}


def guard_signal(
    path: str | os.PathLike[str], *, kind: str, rowspace_key: object, lock_path: str | os.PathLike[str]
) -> dict | None:
    """verify_signal with the lock loaded once per process and the outcome logged."""
    key = os.fspath(lock_path)
    lock = _CACHE.get(key)
    if lock is None:
        lock = _CACHE[key] = load_lock(key)
    row = verify_signal(path, kind=kind, rowspace_key=rowspace_key, lock=lock)
    if row is None:
        _LOG.warning("unlocked %s", os.fspath(path))
    else:
        _LOG.info("signal %s sha256 %s", row["id"], row["sha256"])
    return row
