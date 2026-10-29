"""SIGNALS.lock guard at the four AXIS loaders of this fork: it refuses, and changes nothing.

Each loader must refuse deprecated bytes and a different generation of a locked legacy name, pass
unknown files, and -- for accepted bytes -- return exactly what it returned without the guard.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging

import numpy as np
import pytest
import torch

from lerobot.utils import axis_signals, axis_signals_lock as sl

N_BINS = 5


def _tags(path, variant=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 0 = NO_TAG: AxisQualityTags refuses a file with no unconditional row, so every fixture has one.
    tag = (np.arange(8) % (N_BINS + 1)).astype(np.uint8)
    meta = {
        "reward_id": "phase",
        "n_rows": 8,
        "bin_row_counts": {str(b): int((tag == b).sum()) for b in range(1, N_BINS + 1)},
    }
    np.savez(
        path, tag=tag, prompts=np.array([f"pick up the bowl {variant}"]), meta=np.array(json.dumps(meta))
    )
    return path


def _schedule(path, variant=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ((np.arange(64 * 32, dtype=np.int64) * 7919 + variant) % 100).reshape(64, 32)
    np.savez(path, rows=rows, meta=np.array(json.dumps({"mode": "drop_top", "n_rows": 100})))
    return path


def _flat(path, variant=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weights=np.linspace(0.5, 2.0 + variant, 16, dtype=np.float32))
    return path


N_FRAMES, N_TRAINABLE = 100, 77  # a rows file lists a SUBSET of the frames: len(rows) != num_frames


def _rows(path, variant=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, rows=(np.arange(N_TRAINABLE, dtype=np.int64) + variant) % N_FRAMES)
    return path


def _load_tags(p):
    from lerobot.utils.axis_quality import AxisQualityTags

    t = AxisQualityTags(str(p), expected_frames=8)
    return (t.tag.tolist(), t.prompts)


def _load_schedule(p):
    from lerobot.datasets.axis_sampler import AxisScheduleSampler

    return list(itertools.islice(iter(AxisScheduleSampler(str(p), expected_frames=100)), 2048))


def _load_flat(p):
    from lerobot.utils.axis_sample_weighting import AxisFlatWeighter
    from lerobot.utils.sample_weighting import SampleWeightingConfig

    cfg = SampleWeightingConfig(
        type="axis_flat", extra_params={"weights_path": str(p), "expected_frames": 16}
    )
    return AxisFlatWeighter(cfg, torch.device("cpu"))._weights.tolist()


def _load_rows(p):
    from lerobot.scripts.lerobot_train import _make_axis_sampler

    return list(itertools.islice(iter(_make_axis_sampler(str(p), 0, N_FRAMES)), N_TRAINABLE))


LOADERS = {  # name: (writer, loader, kind, rowspace n_rows, legacy file name)
    "tags": (_tags, _load_tags, "quality_tags", 8, "quality_phase.npz"),
    "schedule": (
        _schedule,
        _load_schedule,
        "lerobot_schedule",
        100,
        "lerobot_schedule_drop_top_phase_smolvla.npz",
    ),
    "flat": (_flat, _load_flat, "lerobot_awr_flat", 16, "lerobot_awr_flat_phase.npz"),
    "rows": (_rows, _load_rows, "lerobot_rows", N_FRAMES, "lerobot_rows.npz"),
}


def _digests(path):
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest(), len(data)


def _lock(tmp_path, n_rows, files=(), deprecated=()):
    rows = []
    for fid, path, kind, legacy in files:
        sha, md5, size = _digests(path)
        rows.append(
            {
                "id": fid,
                "path": f"signals/v1/{fid}",
                "legacy_names": [legacy],
                "kind": kind,
                "reward_id": "phase",
                "granularity": "chunk",
                "rowspace": "rs",
                "size": size,
                "sha256": sha,
                "md5": md5,
                "generator": {},
                "derived_from": [],
                "consumers": {},
                "trained": [],
                "evidence": [],
                "reproduces": None,
            }
        )
    dep = []
    for path, legacy in deprecated:
        sha, md5, _ = _digests(path)
        dep.append(
            {
                "sha256": sha,
                "md5": md5,
                "legacy_name": legacy,
                "hosted": [],
                "superseded_by": None,
                "reason": "fixture stale copy",
            }
        )
    doc = {
        "schema": sl.SCHEMA,
        "release": "test",
        "rowspaces": {"rs": {"n_rows": n_rows}},
        "files": rows,
        "deprecated": dep,
    }
    p = tmp_path / "SIGNALS.lock.json"
    p.write_text(json.dumps(doc))
    return p


@pytest.fixture(autouse=True)
def _fresh_cache():
    sl._CACHE.clear()
    yield
    sl._CACHE.clear()


@pytest.mark.parametrize("name", sorted(LOADERS))
def test_locked_bytes_load_identically_with_the_guard_on_and_off(tmp_path, monkeypatch, caplog, name):
    write, load, kind, n_rows, legacy = LOADERS[name]
    p = write(tmp_path / "study" / legacy)
    monkeypatch.setattr(axis_signals, "LOCK_PATH", _lock(tmp_path, n_rows, files=[("f", p, kind, legacy)]))
    caplog.set_level(logging.INFO, logger="qvla_signals")
    on = load(p)
    assert "signal f sha256" in caplog.text
    monkeypatch.setattr(sl, "guard_signal", lambda *a, **k: None)
    assert load(p) == on


@pytest.mark.parametrize("name", sorted(LOADERS))
def test_deprecated_bytes_are_refused(tmp_path, monkeypatch, name):
    write, load, _, n_rows, legacy = LOADERS[name]
    p = write(tmp_path / "stale" / legacy)
    monkeypatch.setattr(axis_signals, "LOCK_PATH", _lock(tmp_path, n_rows, deprecated=[(p, legacy)]))
    with pytest.raises(sl.SignalLockError, match="deprecated"):
        load(p)


@pytest.mark.parametrize("name", sorted(LOADERS))
def test_another_generation_of_a_locked_name_is_refused(tmp_path, monkeypatch, name):
    write, load, kind, n_rows, legacy = LOADERS[name]
    study = write(tmp_path / "study" / legacy)
    other = write(tmp_path / "other" / legacy, variant=1)
    monkeypatch.setattr(
        axis_signals, "LOCK_PATH", _lock(tmp_path, n_rows, files=[("f", study, kind, legacy)])
    )
    with pytest.raises(sl.SignalLockError, match="different generation of f"):
        load(other)


def test_unknown_files_pass_unlocked(tmp_path, monkeypatch, caplog):
    p = _flat(tmp_path / "x" / "weights_randcam.npz")
    monkeypatch.setattr(axis_signals, "LOCK_PATH", _lock(tmp_path, 16))
    caplog.set_level(logging.INFO, logger="qvla_signals")
    _load_flat(p)
    assert "unlocked" in caplog.text


def test_the_vendored_lock_is_valid():
    """The public v1.0 lock: 30 files, and no deprecation list."""
    lock = sl.load_lock(axis_signals.LOCK_PATH)
    assert lock["release"] == "v1.0" and len(lock["files"]) == 30 and not lock.get("deprecated")
