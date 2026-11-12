"""Run BEAST2 on replicate directories (each holding a ``beast.xml``)."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
BEAST = [str(ROOT / "env/beast2/bin/beast"), "-packagedir", str(ROOT / "env/beast2-packages")]


def run_one(rep: Path, seed: int) -> str:
    """Run ``rep/beast.xml``; touch ``rep/done`` on success. Idempotent."""
    if (rep / "done").exists():
        return f"{rep}: already done"
    with open(rep / "beast.out", "w") as out:
        r = subprocess.run(BEAST + ["-overwrite", "-seed", str(seed), "-threads", "1", "beast.xml"],
                           cwd=rep, stdout=out, stderr=subprocess.STDOUT)
    if r.returncode == 0:
        (rep / "done").touch()
        return f"{rep}: ok"
    return f"{rep}: FAILED (exit {r.returncode})"


def run_all(reps: Iterable[Path], seed_of: Callable[[Path], int], jobs: int) -> None:
    todo = [r for r in reps if not (r / "done").exists()]
    print(f"{len(todo)} replicates to run with {jobs} jobs", flush=True)
    with ThreadPoolExecutor(jobs) as pool:
        for msg in pool.map(lambda r: run_one(r, seed_of(r)), todo):
            print(msg, flush=True)
