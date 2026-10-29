"""The vendored SIGNALS.lock.json must ship in the built wheel and sdist, not only in a checkout.

`axis_signals.LOCK_PATH` resolves next to the installed module, and the guard refuses every AXIS
run when the lock cannot be read. setuptools packages only `*.py` unless package-data names a
file, so a non-editable install that dropped the JSON would refuse every run. The build below
uses this repository's own pyproject.toml, setup.py and MANIFEST.in verbatim, over a skeleton
`lerobot` package holding only the three vendored files: it tests the real packaging rules
without building every policy.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UTILS = REPO / "src" / "lerobot" / "utils"
VENDORED = ("axis_signals.py", "axis_signals_lock.py", "SIGNALS.lock.json")
LOCK_MEMBER = "lerobot/utils/SIGNALS.lock.json"


def _skeleton(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    utils = proj / "src" / "lerobot" / "utils"
    utils.mkdir(parents=True)
    for name in ("pyproject.toml", "setup.py", "README.md", "MANIFEST.in"):
        shutil.copy2(REPO / name, proj / name)
    (proj / "src" / "lerobot" / "__init__.py").write_text("")
    (utils / "__init__.py").write_text("")
    for name in VENDORED:
        shutil.copy2(UTILS / name, utils / name)
    return proj


def _build(tmp_path: Path) -> tuple[Path, Path]:
    proj, out = _skeleton(tmp_path), tmp_path / "dist"
    out.mkdir()
    code = (
        "import pathlib, sys, setuptools.build_meta as b\n"
        "out = sys.argv[1]\n"
        "pathlib.Path(out, 'NAMES').write_text(b.build_wheel(out) + '\\n' + b.build_sdist(out))\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", code, str(out)], cwd=proj, capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    wheel, sdist = (out / "NAMES").read_text().split()
    return out / wheel, out / sdist


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_the_wheel_and_sdist_carry_the_vendored_lock_byte_identically(tmp_path):
    wheel, sdist = _build(tmp_path)
    want = _sha((UTILS / "SIGNALS.lock.json").read_bytes())
    with zipfile.ZipFile(wheel) as z:
        assert LOCK_MEMBER in z.namelist(), f"{wheel.name} has no {LOCK_MEMBER}"
        assert _sha(z.read(LOCK_MEMBER)) == want
    member = f"{sdist.name.removesuffix('.tar.gz')}/src/{LOCK_MEMBER}"
    with tarfile.open(sdist) as t:
        assert member in t.getnames(), f"{sdist.name} has no {member}"
        assert _sha(t.extractfile(member).read()) == want


def test_an_installed_wheel_resolves_and_loads_its_own_lock(tmp_path):
    wheel, _ = _build(tmp_path)
    target = tmp_path / "site"
    with zipfile.ZipFile(wheel) as z:
        z.extractall(target)
    code = (
        "from lerobot.utils import axis_signals, axis_signals_lock as sl\n"
        "print(axis_signals.LOCK_PATH)\n"
        "print(sl.load_lock(axis_signals.LOCK_PATH)['release'])\n"
    )
    env = {**os.environ, "PYTHONPATH": str(target)}
    run = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stderr[-2000:]
    lock_path, release = run.stdout.split()
    assert Path(lock_path).resolve().is_relative_to(target.resolve())
    assert release == "v1.0"
