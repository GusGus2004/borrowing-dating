"""Milestone 3c: does loan detection recover with longer word lists?

31 languages (the Mayan sample size), Mayan-like phonology, kappa = 0.5,
alternating uniform / distance contact; n_concepts in {40, 100, 200}. For
each, level 1 and level 3 are scored with the true cognate sets and with
LexStat-recoded sets (the real pipeline). Level 3 is the logistic
combination, 2-fold cross-validated over replicates.

Outputs data/derived/m3c/results.json and prints a table.
"""

from __future__ import annotations

import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmm.phylo import FEATURES, fit_logistic, predict_logistic  # noqa: E402
from infer.beastxml import load_prior  # noqa: E402
from scripts.m3b_detector import auc  # noqa: E402
from scripts.m5_detector import CONTACT, simulated  # noqa: E402

OUT = ROOT / "data/derived/m3c"
LENGTHS = (40, 100, 200)
N_REPS = 6


def job(args):
    n_concepts, i, pipeline = args
    prior = copy.deepcopy(load_prior(ROOT / "infer/beast/m1_strict_prior.json"))
    prior["n_tips"], prior["n_concepts"] = 31, n_concepts
    X, y = simulated(8000 + 10 * i, prior, CONTACT[i % 2], pipeline)
    return (n_concepts, i, pipeline), X, y


def cv(data):
    folds = [list(range(0, len(data), 2)), list(range(1, len(data), 2))]
    pred, ys, l1 = [], [], []
    for tr, te in (folds, folds[::-1]):
        w = fit_logistic(np.vstack([data[i][0] for i in tr]), np.concatenate([data[i][1] for i in tr]).astype(float))
        for i in te:
            pred.append(predict_logistic(w, data[i][0])); ys.append(data[i][1]); l1.append(data[i][0][:, 0])
    P, Y, L1 = np.concatenate(pred), np.concatenate(ys), np.concatenate(l1)
    X = np.vstack([d[0] for d in data])
    return {"level1": auc(L1, Y), "level3": auc(P, Y), "n": int(len(Y)), "borrowed": float(Y.mean()),
            "features": {f: auc(X[:, k], np.concatenate([d[1] for d in data])) for k, f in enumerate(FEATURES)}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(n, i, p) for n in LENGTHS for p in (False, True) for i in range(N_REPS)]
    jobs.sort(key=lambda j: -j[0])  # longest first
    res = {}
    with ProcessPoolExecutor(7) as pool:
        for key, X, y in pool.map(job, jobs):
            res[key] = (X, y)
            print(f"done {key}", flush=True)
    out = []
    print(f"\n{'concepts':>8s} {'cognates':>9s} {'level 1':>8s} {'level 3':>8s}  (31 languages, {N_REPS} replicates, held-out)")
    for n in LENGTHS:
        for p in (False, True):
            r = cv([res[(n, i, p)] for i in range(N_REPS)])
            r.update(n_concepts=n, cognates="LexStat" if p else "true")
            out.append(r)
            print(f"{n:8d} {r['cognates']:>9s} {r['level1']:8.3f} {r['level3']:8.3f}")
    (OUT / "results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
