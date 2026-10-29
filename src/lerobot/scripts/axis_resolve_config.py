"""Resolve training configs exactly as `lerobot-train --config_path=<file>` does, without training.

    python -m lerobot.scripts.axis_resolve_config --out <new dir> <train_config.json> [...]

For each file this runs what `parser.wrap()` runs for `--config_path` -- the third-party plugin
registration and `TrainPipelineConfig.from_pretrained(<file>)` -- with output_dir set to
`<out>/<i>`, then `validate()`, and saves `<out>/<i>/train_config.json`: the file a checkpoint
records. AXIS-QVLA uses it to check, on a CPU and before any GPU time, that a config resolves to the
values a paper run recorded.
"""

import argparse
from pathlib import Path

import lerobot.policies  # noqa: F401  (registers every policy config class, as lerobot-train does)
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.import_utils import register_third_party_plugins


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("configs", nargs="+", type=Path)
    a = ap.parse_args(argv)
    register_third_party_plugins()
    for i, path in enumerate(a.configs):
        out = a.out / f"{i:03d}"
        cfg = TrainPipelineConfig.from_pretrained(path, cli_args=[f"--output_dir={out}"])
        cfg.validate()
        out.mkdir(parents=True)
        cfg._save_pretrained(out)
        print(f"{path} -> {out / 'train_config.json'}")


if __name__ == "__main__":
    main()
