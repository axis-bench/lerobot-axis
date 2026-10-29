"""Cut one LeRobot v3 root down to the rows of a ranges file: one output episode per span.

An AXIS-QVLA finetune trains on the rows whose within-episode frame falls inside a half-open span
`[start, stop)` of its ranges file (`{"task_<id>--<episode>": [[start, stop], ...]}`), the rows
the openpi line keeps from the same file (`openpi.training.axis_roots.plan_rows_from_roots`). A
LeRobot policy has no ranges hook, so its dataset must contain exactly those rows. `cut_root`
writes such a root:

- one output episode per span: splicing an episode's spans into one output episode would make
  an action chunk cross a cut the robot never executed;
- the columns a policy reads -- every video feature, `observation.state`, the action (read from
  the column the caller names, stored as `action`) and the bookkeeping columns -- with their
  values unchanged: rows are sliced on the arrow table, so the stored floats are the source's;
- every vector statistic recomputed in float64 from the kept rows (corpus-level `stats.json` and
  the per-episode `stats/*` columns); the video statistics are the source's;
- videos are linked, never re-encoded: each kept source episode's mp4 is linked once, and each of
  its spans addresses its window through `videos/*/from_timestamp`. A row is decoded at its rebased
  timestamp plus that offset, which is the source row's own timestamp, so the pixels are the
  source's.

The action column is a stated parameter, never inferred: a root may carry several action columns.
Every refusal (an unknown episode, a span out of range, unsorted or overlapping spans, a missing
action column) is raised before anything is written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

BOOKKEEPING = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
VECTOR_STAT_KEYS = ["observation.state", "action", *BOOKKEEPING]

Span = tuple[int, int]


def task_id(name: str) -> int:
    """The task id a root's directory name `task_<id>` carries (the ranges file's key prefix)."""
    prefix, sep, digits = name.partition("_")
    if (prefix, sep) != ("task", "_") or not digits.isdigit():
        raise ValueError(f"{name!r} is not a task directory name 'task_<id>'")
    return int(digits)


def read_ranges(path: str | os.PathLike[str]) -> dict[str, dict[int, list[Span]]]:
    """`{"task_<id>--<episode>": [[start, stop], ...]}` -> {"task_<id>": {episode: spans}}."""
    out: dict[str, dict[int, list[Span]]] = {}
    for key, spans in json.loads(Path(path).read_text()).items():
        name, sep, episode = key.rpartition("--")
        if not sep or not episode.isdigit():
            raise ValueError(f"{path}: key {key!r} is not 'task_<id>--<episode index>'")
        task_id(name)
        out.setdefault(name, {})[int(episode)] = [(int(s), int(t)) for s, t in spans]
    return out


def _stats_block(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return {
        "mean": a.mean(0).tolist(),
        "std": a.std(0).tolist(),
        "min": a.min(0).tolist(),
        "max": a.max(0).tolist(),
        "count": [int(len(a))],
    }


def _column(df, name: str) -> np.ndarray:
    col = df[name].to_numpy()
    return np.stack(col) if col.dtype == object else col


def _checked_spans(src: Path, spans: dict[int, list[Span]], lengths: dict[int, int]) -> dict[int, list[Span]]:
    """Spans that are inside their episode, sorted and disjoint; a span that would need clamping
    changes the kept-row count against the ranges file's own arithmetic, so it is refused."""
    norm: dict[int, list[Span]] = {}
    for ep in sorted(spans):
        if not spans[ep]:
            continue
        if ep not in lengths:
            raise ValueError(f"{src}: the ranges name episode {ep}, which the root does not have")
        prev_end = 0
        for s, t in spans[ep]:
            if not 0 <= s < t <= lengths[ep]:
                raise ValueError(f"{src}: episode {ep} span [{s}, {t}) is outside its {lengths[ep]} frames")
            if s < prev_end:
                raise ValueError(f"{src}: episode {ep} spans are unsorted or overlap at [{s}, {t})")
            prev_end = t
        norm[ep] = list(spans[ep])
    if not norm:
        raise ValueError(f"{src}: the ranges keep no row of this root")
    return norm


def cut_root(src: Path, dst: Path, *, action_source: str, spans: dict[int, list[Span]]) -> list[dict]:
    """Write at `dst` the rows of `src` that `spans` ({episode: [(start, stop), ...]}) keep.

    Returns the output episodes in order, each `{"episode_index", "src_episode_index",
    "src_frame_start", "src_frame_stop"}`.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise ValueError(f"refusing to overwrite {dst}")
    info = json.loads((src / "meta" / "info.json").read_text())
    src_feats = info["features"]
    if action_source not in src_feats:
        raise ValueError(
            f"{src}: meta/info.json declares no feature {action_source!r} to read the action from"
        )
    video_keys = [k for k, v in src_feats.items() if v.get("dtype") == "video"]
    kept = [*video_keys, "observation.state", "action", *BOOKKEEPING]
    feats = {k: v for k, v in src_feats.items() if k in kept and k != "action"}
    feats["action"] = dict(src_feats[action_source])
    missing = [k for k in kept if k not in feats]
    if missing:
        raise ValueError(f"{src}: meta/info.json declares no feature {missing}")
    columns = [k for k in kept if k not in video_keys]  # data columns, named as written
    src_columns = [action_source if k == "action" else k for k in columns]
    fps = float(info["fps"])

    ep_files = sorted((src / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise ValueError(f"{src}: no meta/episodes parquet")
    src_ep = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    src_ep = src_ep.sort_values("episode_index").set_index("episode_index", drop=False)
    lengths = {int(e): int(n) for e, n in zip(src_ep["episode_index"], src_ep["length"], strict=True)}
    norm = _checked_spans(src, spans, lengths)

    dst.mkdir(parents=True)
    (dst / "meta").mkdir()
    (dst / "meta" / "tasks.parquet").write_bytes((src / "meta" / "tasks.parquet").read_bytes())

    vid_file_of = {e: i for i, e in enumerate(norm)}
    for key in video_keys:
        for e in norm:
            c, f = (
                int(src_ep.at[e, f"videos/{key}/chunk_index"]),
                int(src_ep.at[e, f"videos/{key}/file_index"]),
            )
            src_mp4 = src / info["video_path"].format(video_key=key, chunk_index=c, file_index=f)
            out = dst / info["video_path"].format(video_key=key, chunk_index=0, file_index=vid_file_of[e])
            out.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src_mp4.resolve(), out)

    new_idx = global_from = 0
    parts: list = []
    ep_rows: list[dict] = []
    plan: list[dict] = []
    stale = [c for c in src_ep.columns if c.startswith("stats/")]
    for e in norm:
        f = src / info["data_path"].format(
            chunk_index=int(src_ep.at[e, "data/chunk_index"]), file_index=int(src_ep.at[e, "data/file_index"])
        )
        table = pq.read_table(f)
        if action_source not in table.column_names:
            raise ValueError(f"{f}: no {action_source!r} column, though meta/info.json declares it")
        table = table.select(src_columns).rename_columns(columns)
        where = np.flatnonzero(table.column("episode_index").to_numpy() == e)
        if len(where) != lengths[e] or (len(where) and where[-1] - where[0] + 1 != len(where)):
            raise ValueError(
                f"{f}: episode {e} has {len(where)} rows, or they are not contiguous; "
                f"its metadata says {lengths[e]}"
            )
        lo = int(where[0])
        if int(table.column("frame_index").to_numpy()[lo]) != 0:
            raise ValueError(f"{f}: episode {e} does not start at frame_index 0")
        ts = table.column("timestamp").to_numpy()
        ts_type = table.schema.field("timestamp").type
        for s, t in norm[e]:
            n = t - s
            sl = table.slice(lo + s, n)
            for name, arr in (
                ("frame_index", pa.array(np.arange(n, dtype=np.int64))),
                ("timestamp", pa.array(ts[lo + s : lo + t] - ts[lo + s], type=ts_type)),
                ("episode_index", pa.array(np.full(n, new_idx, dtype=np.int64))),
                ("index", pa.array(np.arange(global_from, global_from + n, dtype=np.int64))),
            ):
                i = sl.schema.get_field_index(name)
                sl = sl.set_column(i, sl.schema.field(name).with_nullable(True), arr)
            out = dst / info["data_path"].format(chunk_index=0, file_index=new_idx)
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(sl, out)
            parts.append(sl.to_pandas())
            row = src_ep.loc[e].drop(labels=stale).to_dict()
            row.update(
                {
                    "episode_index": new_idx,
                    "length": n,
                    "dataset_from_index": global_from,
                    "dataset_to_index": global_from + n,
                    "data/chunk_index": 0,
                    "data/file_index": new_idx,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                }
            )
            for key in video_keys:
                base = float(src_ep.at[e, f"videos/{key}/from_timestamp"]) + float(ts[lo + s])
                row.update(
                    {
                        f"videos/{key}/chunk_index": 0,
                        f"videos/{key}/file_index": vid_file_of[e],
                        f"videos/{key}/from_timestamp": base,
                        f"videos/{key}/to_timestamp": base + n / fps,
                    }
                )
            ep_rows.append(row)
            plan.append(
                {
                    "episode_index": new_idx,
                    "src_episode_index": int(e),
                    "src_frame_start": int(s),
                    "src_frame_stop": int(t),
                }
            )
            new_idx += 1
            global_from += n

    data = pd.concat(parts, ignore_index=True)
    info["features"] = {k: feats[k] for k in kept}
    info.update(total_episodes=new_idx, total_frames=global_from, splits={"train": f"0:{new_idx}"})
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    old_stats = json.loads((src / "meta" / "stats.json").read_text())
    stats = {k: v for k, v in old_stats.items() if k in video_keys}
    for k in VECTOR_STAT_KEYS:
        stats[k] = _stats_block(_column(data, k))
    if len(stats["action"]["mean"]) != info["features"]["action"]["shape"][0]:
        raise ValueError(
            f"{src}: the {action_source!r} column is {len(stats['action']['mean'])}-D but "
            f"meta/info.json declares shape {info['features']['action']['shape']}"
        )
    (dst / "meta" / "stats.json").write_text(json.dumps(stats))

    ep = pd.DataFrame(ep_rows)
    for k in VECTOR_STAT_KEYS:
        blocks = [
            _stats_block(_column(data.iloc[r["dataset_from_index"] : r["dataset_to_index"]], k))
            for r in ep_rows
        ]
        for stat in ("mean", "std", "min", "max", "count"):
            ep[f"stats/{k}/{stat}"] = [b[stat] for b in blocks]
    ep_out = dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_out.parent.mkdir(parents=True, exist_ok=True)
    ep.to_parquet(ep_out)
    return plan
