"""Simulate one SDollo dataset (no borrowing) and write it as CLDF + ground truth.

Usage:
    python scripts/simulate.py --out data/derived/sim/example --seed 1 \
        --n-tips 10 --root-age 4000 --n-concepts 200 --birth 2e-4 --loss 2e-4 \
        --gamma-shape 0.5 --clock-sd 0.3
    python scripts/simulate.py --out ... --tree my_tree.nwk   # fixed tree
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.cldf import write_simulation  # noqa: E402
from sim.sdollo import SDolloParams, simulate  # noqa: E402
from sim.tree import Tree, simulate_yule  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tree", help="Newick file with time-calibrated tree (overrides Yule)")
    ap.add_argument("--n-tips", type=int, default=10)
    ap.add_argument("--root-age", type=float, default=4000.0)
    ap.add_argument("--n-concepts", type=int, default=200)
    ap.add_argument("--birth", type=float, default=2e-4)
    ap.add_argument("--loss", type=float, default=2e-4)
    ap.add_argument("--gamma-shape", type=float, default=None)
    ap.add_argument("--gamma-categories", type=int, default=4)
    ap.add_argument("--clock-sd", type=float, default=0.0)
    a = ap.parse_args()

    # One master seed; independent streams for tree and characters.
    tree_ss, char_ss = np.random.SeedSequence(a.seed).spawn(2)
    if a.tree:
        tree = Tree.from_newick(Path(a.tree).read_text())
    else:
        tree = simulate_yule(a.n_tips, a.root_age, np.random.default_rng(tree_ss))
    params = SDolloParams(n_concepts=a.n_concepts, birth_rate=a.birth, loss_rate=a.loss,
                          gamma_shape=a.gamma_shape, gamma_categories=a.gamma_categories,
                          clock_sd=a.clock_sd)
    res = simulate(tree, params, rng=np.random.default_rng(char_ss))
    res.seed = a.seed
    md = write_simulation(res, a.out)
    obs = res.observed()
    print(f"{len(obs.leaf_names)} languages, {obs.n_classes} observed cognate classes, "
          f"root age {tree.root_age:g} -> {md}")


if __name__ == "__main__":
    main()
