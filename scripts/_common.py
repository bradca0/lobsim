"""Shared plumbing for the experiment scripts: paths, seeds, JSON writing, timing."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "results" / "raw"
FIGURE_DIR = REPO_ROOT / "results" / "figures"

# The experiment protocol, frozen before any results were generated. Training and test seeds are
# disjoint; the learned policy never sees a test seed during fitting or model selection.
TRAIN_SEEDS: tuple[int, ...] = tuple(range(0, 200))
VALIDATION_SEEDS: tuple[int, ...] = tuple(range(500, 560))
TEST_SEEDS: tuple[int, ...] = tuple(range(1000, 1200))


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def provenance() -> dict[str, Any]:
    """Recorded alongside every result so a number can always be traced to the code that made it."""
    return {
        "git_revision": _git_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def write_json(name: str, payload: dict[str, Any]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    payload = {**payload, "_provenance": provenance()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_encode) + "\n")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return path


def read_json(name: str) -> dict[str, Any]:
    path = RAW_DIR / name
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT)} -- run `make reproduce` to regenerate results"
        )
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _encode(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"cannot serialise {type(obj)!r}")


@contextmanager
def stage(title: str) -> Iterator[None]:
    print(f"[{title}] starting")
    start = time.perf_counter()
    yield
    print(f"[{title}] done in {time.perf_counter() - start:.1f}s")


def progress(index: int, total: int, every: int = 25) -> None:
    if index % every == 0 or index == total - 1:
        print(f"    {index + 1}/{total}", end="\r", flush=True)
        if index == total - 1:
            print()
