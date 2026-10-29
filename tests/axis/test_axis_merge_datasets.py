"""axis_merge_datasets: a plain merge concatenates its inputs; with --ranges it keeps exactly the
spans' rows, one episode per span, in task-id order whatever the argument order, the stated action
column, recomputed statistics, and the source's values and pixels."""

import hashlib
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.axis_merge_datasets import check_merged, main

CAMS = ("observation.images.third_person", "observation.images.wrist")
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "action": {"dtype": "float32", "shape": (8,), "names": None},
    "observation.ee_state": {"dtype": "float32", "shape": (7,), "names": None},
    **{k: {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channels"]} for k in CAMS},
    "action_eef": {"dtype": "float32", "shape": (7,), "names": None},
}
# Task ids whose string order ("task_10" < "task_2") is not their numeric order (2 < 10).
A, B = 2, 10


def _value(task: int, ep: int, frame: int, width: int) -> np.ndarray:
    return (task * 1000 + ep * 100 + frame + np.arange(width) / 10).astype(np.float32)


def _root(tmp_path, task: int, lengths: list[int]):
    ds = LeRobotDataset.create(
        f"local/task_{task}", fps=15, features=FEATURES, root=tmp_path / f"task_{task}"
    )
    for ep, n in enumerate(lengths):
        for fr in range(n):
            img = np.full((64, 64, 3), (task * 37 + ep * 11 + fr * 9) % 256, dtype=np.uint8)
            ds.add_frame(
                {
                    "observation.state": _value(task, ep, fr, 8),
                    "action": -_value(task, ep, fr, 8),
                    "observation.ee_state": _value(task, ep, fr, 7),
                    "action_eef": _value(task, ep, fr, 7),
                    CAMS[0]: img,
                    CAMS[1]: 255 - img,
                    "task": "pick",
                }
            )
        ds.save_episode()
    ds.finalize()
    return ds.root


def _load(root):
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version", return_value="v3.0"),
        patch("lerobot.datasets.dataset_metadata.snapshot_download", return_value=str(root)),
    ):
        return LeRobotDataset("local/merged", root=root, video_backend="pyav")


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("src")
    return [_root(tmp, A, [7, 5]), _root(tmp, B, [6])]


SPANS = {f"task_{A}--0": [[1, 3], [4, 7]], f"task_{A}--1": [[0, 2]], f"task_{B}--0": [[2, 5]]}
# (task, source episode, start, stop) of each output episode, in order.
PLAN = [(A, 0, 1, 3), (A, 0, 4, 7), (A, 1, 0, 2), (B, 0, 2, 5)]


def _ranges(tmp_path, spans=SPANS):
    p = tmp_path / "ranges.json"
    p.write_text(json.dumps(spans))
    return p


def _cut(out, ranges, inputs, action_source="action"):
    main(["--out", str(out), "--ranges", str(ranges), "--action-source", action_source, *map(str, inputs)])


def _build_dirs(tmp_path, name="m"):
    return [p for p in tmp_path.iterdir() if p.name.startswith(f".{name}.build-")]


def test_a_plain_merge_concatenates_the_inputs(tmp_path, roots):
    main(["--out", str(tmp_path / "m"), *map(str, roots)])
    info = json.loads((tmp_path / "m" / "meta" / "info.json").read_text())
    assert (info["total_frames"], info["total_episodes"]) == (18, 3)
    assert "action_eef" in info["features"]


def test_a_plain_merge_is_in_task_id_order_whatever_the_argument_order(tmp_path, roots):
    main(["--out", str(tmp_path / "ordered"), *map(str, roots)])
    main(["--out", str(tmp_path / "reversed"), *map(str, reversed(roots))])
    assert _files(tmp_path / "reversed") == _files(tmp_path / "ordered")
    first = _load(tmp_path / "ordered")[0]  # task A's first frame leads, though "task_10" < "task_2"
    assert torch.equal(first["observation.state"], torch.from_numpy(_value(A, 0, 0, 8)))


def test_a_plain_merge_refuses_inputs_that_are_not_one_directory_per_task(tmp_path, roots):
    other = tmp_path / "elsewhere"
    (other / "scene").mkdir(parents=True)
    with pytest.raises(ValueError, match="not a task directory name"):
        main(["--out", str(tmp_path / "m"), *map(str, roots), str(other / "scene")])
    (other / roots[0].name).mkdir()
    with pytest.raises(ValueError, match=f"two inputs are task {A}"):
        main(["--out", str(tmp_path / "m"), *map(str, roots), str(other / roots[0].name)])
    assert not (tmp_path / "m").exists()


def test_the_copy_records_the_ranges_file_and_inputs_it_was_cut_from(tmp_path, roots):
    ranges = _ranges(tmp_path)
    _cut(tmp_path / "m", ranges, list(reversed(roots)))
    record = json.loads((tmp_path / "m" / "axis_spans.json").read_text())
    assert record["ranges_sha256"] == hashlib.sha256(ranges.read_bytes()).hexdigest()
    assert record["action_source"] == "action"
    assert record["inputs"] == [
        {"root": r.name, "info_sha256": hashlib.sha256((r / "meta" / "info.json").read_bytes()).hexdigest()}
        for r in roots
    ]
    assert len(record["episodes"]) == len(PLAN)


def test_ranges_keep_exactly_the_spans_one_episode_per_span(tmp_path, roots):
    out = tmp_path / "m"
    _cut(out, _ranges(tmp_path), roots)
    info = json.loads((out / "meta" / "info.json").read_text())
    assert (info["total_frames"], info["total_episodes"]) == (10, 4)
    assert list(info["features"]) == [
        *CAMS,
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    plan = json.loads((out / "axis_spans.json").read_text())["episodes"]
    assert [
        (int(p["root"][5:]), p["src_episode_index"], p["src_frame_start"], p["src_frame_stop"]) for p in plan
    ] == PLAN
    assert [p["episode_index"] for p in plan] == [0, 1, 2, 3]
    ds = _load(out)
    rows = [(t, e, f) for t, e, s, stop in PLAN for f in range(s, stop)]
    for j, (t, e, f) in enumerate(rows):
        item = ds[j]
        assert torch.equal(item["action"], torch.from_numpy(-_value(t, e, f, 8)))
        assert torch.equal(item["observation.state"], torch.from_numpy(_value(t, e, f, 8)))
    assert [int(ds[j]["episode_index"]) for j in range(len(rows))] == [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
    assert [int(ds[j]["frame_index"]) for j in range(len(rows))] == [0, 1, 0, 1, 2, 0, 1, 0, 1, 2]
    kept = np.stack([-_value(t, e, f, 8) for t, e, f in rows]).astype(np.float64)
    stats = json.loads((out / "meta" / "stats.json").read_text())
    np.testing.assert_allclose(stats["action"]["mean"], kept.mean(0), rtol=1e-6)
    np.testing.assert_allclose(stats["action"]["min"], kept.min(0), rtol=0)


def _files(root):
    """Every file of a copy: JSON compared as values (its key order comes from set iteration),
    everything else as bytes."""
    out = {}
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = f.relative_to(root).as_posix()
        out[rel] = json.loads(f.read_text()) if f.suffix == ".json" else f.read_bytes()
    return out


def test_the_row_order_is_task_id_order_whatever_the_argument_order(tmp_path, roots):
    """A shell glob gives "task_10" before "task_2"; the copy is the same as the task-id-ordered one."""
    ranges = _ranges(tmp_path)
    _cut(tmp_path / "ordered", ranges, roots)
    _cut(tmp_path / "reversed", ranges, list(reversed(roots)))
    assert sorted(r.name for r in roots) == [f"task_{B}", f"task_{A}"]  # what a glob would give
    assert _files(tmp_path / "reversed") == _files(tmp_path / "ordered")


def test_ranges_keep_the_source_pixels(tmp_path, roots):
    out = tmp_path / "m"
    _cut(out, _ranges(tmp_path), roots)
    merged, src = _load(out), {r.name: _load(r) for r in roots}
    offsets = {f"task_{A}": [0, 7], f"task_{B}": [0]}
    rows = [(t, e, f) for t, e, s, stop in PLAN for f in range(s, stop)]
    for j in (0, 2, 5, 9):
        t, e, f = rows[j]
        a, b = merged[j], src[f"task_{t}"][offsets[f"task_{t}"][e] + f]
        for cam in CAMS:
            assert torch.equal(a[cam], b[cam])


def test_the_action_column_is_the_one_stated(tmp_path, roots):
    out = tmp_path / "m"
    _cut(out, _ranges(tmp_path, {f"task_{A}--0": [[0, 2]], f"task_{B}--0": [[0, 1]]}), roots, "action_eef")
    assert json.loads((out / "meta" / "info.json").read_text())["features"]["action"]["shape"] == [7]
    assert torch.equal(_load(out)[2]["action"], torch.from_numpy(_value(B, 0, 0, 7)))


@pytest.mark.parametrize(
    "spans, match",
    [
        ({f"task_{A}--0": [[0, 8]], f"task_{B}--0": [[0, 1]]}, "outside its 7 frames"),
        ({f"task_{A}--0": [[0, 4], [3, 5]], f"task_{B}--0": [[0, 1]]}, "unsorted or overlap"),
        ({f"task_{A}--9": [[0, 1]], f"task_{B}--0": [[0, 1]]}, "does not have"),
        ({f"task_{A}--0": [[0, 1]], f"task_{B}--0": [[0, 1]], "task_3--0": [[0, 1]]}, "not inputs"),
        ({f"task_{A}--0": [[0, 1]]}, "no spans for the inputs"),
        ({f"task_{A}--0": [[0, 1]], "scene--0": [[0, 1]]}, "not a task directory name"),
    ],
)
def test_spans_that_do_not_fit_are_refused_before_anything_is_merged(tmp_path, roots, spans, match):
    out = tmp_path / "m"
    with pytest.raises(ValueError, match=match):
        _cut(out, _ranges(tmp_path, spans), roots)
    assert not out.exists()
    assert not _build_dirs(tmp_path)


def test_inputs_that_are_not_one_directory_per_task_are_refused(tmp_path, roots):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "scene").mkdir()
    with pytest.raises(ValueError, match="not a task directory name"):
        _cut(tmp_path / "m", _ranges(tmp_path), [*roots, other / "scene"])
    twin = other / roots[0].name
    twin.mkdir()
    with pytest.raises(ValueError, match=f"two inputs are task {A}"):
        _cut(tmp_path / "m", _ranges(tmp_path), [*roots, twin])
    assert not (tmp_path / "m").exists()


def test_the_action_column_must_be_stated_and_exist(tmp_path, roots):
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "m"), "--ranges", str(_ranges(tmp_path)), *map(str, roots)])
    with pytest.raises(ValueError, match="declares no feature 'nope'"):
        _cut(tmp_path / "m", _ranges(tmp_path), roots, "nope")


def test_a_copy_that_fails_its_check_never_appears_at_out(tmp_path, roots, monkeypatch):
    import lerobot.scripts.axis_merge_datasets as m

    def refuse(*a, **kw):
        raise ValueError("planted check failure")

    monkeypatch.setattr(m, "check_merged", refuse)
    out = tmp_path / "m"
    with pytest.raises(ValueError, match="planted check failure"):
        _cut(out, _ranges(tmp_path), roots)
    assert not out.exists()
    assert not _build_dirs(tmp_path)


@pytest.mark.parametrize("plain", [True, False], ids=["plain", "ranges"])
def test_an_existing_out_is_refused_before_any_work(tmp_path, roots, plain):
    out = tmp_path / "m"
    out.mkdir()
    (out / "keep").write_text("x")
    with pytest.raises(FileExistsError, match="already exists"):
        if plain:
            main(["--out", str(out), *map(str, roots)])
        else:
            _cut(out, _ranges(tmp_path), roots)
    assert [p.name for p in out.iterdir()] == ["keep"]


def test_a_build_left_by_a_killed_run_is_refused_and_kept(tmp_path, roots):
    from lerobot.scripts.axis_merge_datasets import merge_cut

    left = tmp_path / ".m.build-abc123"
    left.mkdir()
    with pytest.raises(
        FileExistsError, match=r"killed or is still running: .*\.m\.build-abc123.*remove that"
    ):
        merge_cut(roots, tmp_path / "m", ranges=_ranges(tmp_path), action_source="action", repo_id="r")
    assert left.is_dir() and not (tmp_path / "m").exists()


@pytest.mark.parametrize("column, shift", [("length", 1), ("dataset_from_index", 1), ("from_timestamp", 0.1)])
def test_the_check_refuses_an_episode_table_that_is_not_the_spans(tmp_path, roots, column, shift):
    out = tmp_path / "m"
    _cut(out, _ranges(tmp_path), roots)
    plan = json.loads((out / "axis_spans.json").read_text())["episodes"]
    check_merged(out, roots, plan, action_source="action")  # the built copy passes
    (f,) = sorted((out / "meta" / "episodes").rglob("*.parquet"))
    eps = pd.read_parquet(f)
    key = f"videos/{CAMS[1]}/{column}" if column == "from_timestamp" else column
    eps.loc[eps["episode_index"] == 2, key] += shift
    eps.to_parquet(f)
    with pytest.raises(ValueError, match="episode 2's"):
        check_merged(out, roots, plan, action_source="action")
