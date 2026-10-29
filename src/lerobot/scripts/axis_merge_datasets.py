"""Merge AXIS task datasets into one, the merged frame index an AXIS row space indexes.

    python -m lerobot.scripts.axis_merge_datasets --out <dir> [--repo-id axis/merged] <task_<id> dir> ...
    python -m lerobot.scripts.axis_merge_datasets --out <dir> --ranges <ranges.json> \
        --action-source <column> [--repo-id axis/merged] <task_<id> dir> ...

In both modes the inputs are task directories `task_<id>`, one per task, and they are merged in
ascending task id -- the order every AXIS row space walks its roots in -- whatever order they are
given in: the copy's row order is a property of the inputs, never of the command line. A name
that is not `task_<id>`, or two inputs for one task, is refused.

Without --ranges, a thin command over `lerobot.datasets.aggregate.aggregate_datasets`: the merged
frame index is the concatenation of the inputs in that order.

With --ranges, the merged copy holds exactly the rows inside the file's spans, the rows an
AXIS-QVLA finetune trains on, in the row order that finetune's row space indexes:

- the inputs must be exactly the task directories the ranges file names (`task_<id>--<episode>`);
- each is cut to its spans, one output episode per span, keeping the video features,
  `observation.state` and the action read from `--action-source` (stored as `action`), with every
  vector statistic recomputed from the kept rows (`lerobot.datasets.axis_spans.cut_root`); videos
  are copied, one file per kept source episode, never re-encoded;
- the copy is built and checked beside `--out` and moved there only when every check passes: as
  many frames and episodes as the spans hold; each episode's length, row range and video windows
  the span's; every row's action and state bit-equal to its source row. `axis_spans.json` beside
  `meta/` records what the copy was cut from -- the ranges file's sha256 (`ranges_sha256`), each
  input's task directory and the sha256 of its `meta/info.json` -- and, for each output episode,
  its task directory, source episode and `[start, stop)`. Before a run trains on the copy, AXIS-QVLA
  compares every field of this record with what the run's row space names (its ranges file, action
  column and task directories), so a copy cut from other spans, inputs or action column is refused
  even when its frame count is the same.

`--out` must not exist, and a hidden build directory beside it (`.<out name>.build-*`, left by a
build that was killed or is still running) is refused rather than reused or removed.
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from lerobot.datasets.aggregate import aggregate_datasets

SPANS_FILE = "axis_spans.json"
# Video windows are float seconds written by the cut and copied by the merge unchanged.
WINDOW_TOLERANCE_S = 1e-9


def build_dirs(out: Path) -> list[Path]:
    """Hidden build directories of `out` beside it: a build in progress or one that was killed.
    A parent that cannot be listed raises (PermissionError), never reads as "none"."""
    try:
        names = os.listdir(out.parent)
    except FileNotFoundError:
        return []
    return sorted(out.parent / n for n in names if n.startswith(f".{out.name}.build-"))


def task_ordered(roots: list[Path]) -> list[Path]:
    """`roots` in ascending task id; refuses a name that is not `task_<id>` and two inputs for one task."""
    from lerobot.datasets.axis_spans import task_id

    by_id: dict[int, Path] = {}
    for r in roots:
        tid = task_id(r.name)
        if tid in by_id:
            raise ValueError(f"two inputs are task {tid}: {by_id[tid]} and {r}")
        by_id[tid] = r
    return [by_id[t] for t in sorted(by_id)]


def refuse_existing(out: Path) -> None:
    """Every entry point's first check: never write over an existing `--out` or a leftover build."""
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"--out {out} already exists; remove it or choose another --out")
    left = build_dirs(out)
    if left:
        raise FileExistsError(
            f"a build of {out} was killed or is still running: {[str(d) for d in left]}. If no build is "
            "running, remove that directory and rebuild"
        )


def ordered_inputs(roots: list[Path], spans: dict[str, dict], ranges: Path) -> list[Path]:
    """`roots` in ascending task id, the ranges walk's order; refuses a set that is not the file's."""
    ordered = task_ordered(roots)
    names = {r.name for r in roots}
    unknown, uncovered = sorted(set(spans) - names), sorted(names - set(spans))
    problems = [f"names roots that are not inputs: {unknown}"] if unknown else []
    problems += [f"has no spans for the inputs {uncovered}"] if uncovered else []
    if problems:
        raise ValueError(f"{ranges} " + "; ".join(problems))
    return ordered


def merge_cut(roots: list[Path], out: Path, *, ranges: Path, action_source: str, repo_id: str) -> list[dict]:
    """Cut every root to its spans in `ranges`, merge the cut roots in task-id order, check, move to `out`."""
    from lerobot.datasets.axis_spans import cut_root, read_ranges

    refuse_existing(out)
    spans = read_ranges(ranges)
    roots = ordered_inputs(roots, spans, ranges)
    out.parent.mkdir(parents=True, exist_ok=True)
    plan: list[dict] = []
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=f".{out.name}.build-") as work:
        cut = [Path(work) / "cut" / r.name for r in roots]
        for root, dst in zip(roots, cut, strict=True):
            first = len(plan)
            for p in cut_root(root, dst, action_source=action_source, spans=spans[root.name]):
                plan.append({"root": root.name, **p, "episode_index": first + p["episode_index"]})
        merged = Path(work) / "merged"
        aggregate_datasets(
            repo_ids=[f"local/{r.name}" for r in roots],
            aggr_repo_id=repo_id,
            roots=cut,
            aggr_root=merged,
            concatenate_videos=False,
        )
        check_merged(merged, roots, plan, action_source=action_source)
        record = {
            "ranges_sha256": _sha256(ranges),
            "action_source": action_source,
            "inputs": [{"root": r.name, "info_sha256": _sha256(r / "meta" / "info.json")} for r in roots],
            "episodes": plan,
        }
        (merged / SPANS_FILE).write_text(json.dumps(record, indent=1))
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"--out {out} appeared while the copy was being built; not replacing it")
        os.rename(merged, out)
    return plan


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _merged_rows(out: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.concat_tables([pq.read_table(f) for f in sorted((out / "data").rglob("*.parquet"))])
    order = np.argsort(table.column("index").to_numpy(), kind="stable")
    return table.take(pa.array(order))


def _episodes(root: Path):
    import pandas as pd

    eps = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))])
    return eps.sort_values("episode_index").set_index("episode_index", drop=False)


def _source_episodes(root: Path, action_source: str):
    """root -> {episode index: its rows (frame order) of the timestamp, action and state columns}."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    eps = _episodes(root)
    out = {}
    for e, c, f in zip(eps["episode_index"], eps["data/chunk_index"], eps["data/file_index"], strict=True):
        path = root / info["data_path"].format(chunk_index=int(c), file_index=int(f))
        cols = ["episode_index", "frame_index", "timestamp", action_source, "observation.state"]
        t = pq.read_table(path, columns=cols)
        t = t.filter(
            pc.equal(t.column("episode_index"), pa.scalar(int(e), t.schema.field("episode_index").type))
        )
        out[int(e)] = t.take(pa.array(np.argsort(t.column("frame_index").to_numpy(), kind="stable")))
    return out, eps


def check_merged(out: Path, roots: list[Path], plan: list[dict], *, action_source: str) -> None:
    """Refuse a merged copy whose counts, episode table, video windows or values are not the spans'
    rows of the inputs."""
    info = json.loads((out / "meta" / "info.json").read_text())
    n_rows = sum(p["src_frame_stop"] - p["src_frame_start"] for p in plan)
    if (info["total_frames"], info["total_episodes"]) != (n_rows, len(plan)):
        raise ValueError(
            f"{out}: {info['total_frames']} frames in {info['total_episodes']} episodes; the spans "
            f"hold {n_rows} rows in {len(plan)} episodes"
        )
    merged = _merged_rows(out)
    if not np.array_equal(merged.column("index").to_numpy(), np.arange(n_rows)):
        raise ValueError(f"{out}: the merged 'index' column is not 0..{n_rows - 1}")
    fps = float(info["fps"])
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    table = _episodes(out)
    if table["episode_index"].tolist() != list(range(len(plan))):
        raise ValueError(f"{out}: meta/episodes does not list episodes 0..{len(plan) - 1}")
    by_name = {r.name: r for r in roots}
    src_action, src_state, lengths, source, start = [], [], [], {}, 0
    for p in plan:
        if p["root"] not in source:
            source = {p["root"]: _source_episodes(by_name[p["root"]], action_source)}
        src_rows, src_eps = source[p["root"]]
        e, s, n = p["src_episode_index"], p["src_frame_start"], p["src_frame_stop"] - p["src_frame_start"]
        rows = src_rows[e].slice(s, n)
        row = table.loc[p["episode_index"]]
        if (int(row["length"]), int(row["dataset_from_index"]), int(row["dataset_to_index"])) != (
            n,
            start,
            start + n,
        ):
            raise ValueError(f"{out}: episode {p['episode_index']}'s length or row range is not its span's")
        for cam in cams:
            want_from = float(src_eps.at[e, f"videos/{cam}/from_timestamp"]) + float(
                src_rows[e].column("timestamp")[s].as_py()
            )
            got_from, got_to = (
                float(row[f"videos/{cam}/from_timestamp"]),
                float(row[f"videos/{cam}/to_timestamp"]),
            )
            if (
                abs(got_from - want_from) > WINDOW_TOLERANCE_S
                or abs(got_to - got_from - n / fps) > WINDOW_TOLERANCE_S
            ):
                raise ValueError(f"{out}: episode {p['episode_index']}'s {cam} window is not its span's")
        src_action.append(np.stack(rows.column(action_source).to_numpy(zero_copy_only=False)))
        src_state.append(np.stack(rows.column("observation.state").to_numpy(zero_copy_only=False)))
        lengths.append(len(rows))
        start += n
    got_lengths = np.bincount(merged.column("episode_index").to_numpy(), minlength=len(plan)).tolist()
    if got_lengths != lengths:
        raise ValueError(f"{out}: the merged episodes' lengths are not the spans' lengths, in order")
    for name, src in (("action", src_action), ("observation.state", src_state)):
        got = np.stack(merged.column(name).to_numpy(zero_copy_only=False))
        want = np.concatenate(src)
        if got.dtype != want.dtype or got.tobytes() != want.tobytes():
            raise ValueError(f"{out}: merged {name!r} is not bit-equal to the source rows the spans name")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-id", default="axis/merged")
    ap.add_argument("--ranges", type=Path, help="keep only the rows inside this file's spans")
    ap.add_argument("--action-source", help="with --ranges: the column the action is read from")
    ap.add_argument("roots", nargs="+", type=Path)
    a = ap.parse_args(argv)
    if a.ranges is None:
        if a.action_source is not None:
            ap.error("--action-source needs --ranges")
        refuse_existing(a.out)
        roots = task_ordered(a.roots)
        aggregate_datasets(
            repo_ids=[f"local/{r.name}" for r in roots],
            aggr_repo_id=a.repo_id,
            roots=roots,
            aggr_root=a.out,
        )
    else:
        if a.action_source is None:
            ap.error(
                "--ranges needs --action-source: the column the action is read from is stated, never guessed"
            )
        plan = merge_cut(a.roots, a.out, ranges=a.ranges, action_source=a.action_source, repo_id=a.repo_id)
        print(f"{len(plan)} episodes, {sum(p['src_frame_stop'] - p['src_frame_start'] for p in plan)} frames")
    print(a.out)


if __name__ == "__main__":
    main()
