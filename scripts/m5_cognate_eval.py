"""Whole-word vs morpheme-level (partial) cognate detection on simulated data with known cognates.

Simulated Mayan-shaped wordlists (31 languages, 40 concepts, milestone 1
priors, CVC roots and stable segments) with Mayan-style suffixation at
``suffix_rate`` per concept per lineage per year (0 = control). Each method
is scored by B-cubed precision, recall and F against the true cognate sets.

Outputs data/derived/m5_cognate_eval/results.json and prints a table.
"""

from __future__ import annotations

import collections
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer.beastxml import draw_truth, load_prior  # noqa: E402
from pipeline.asjp import lexstat_cognates, partial_cognates  # noqa: E402
from sim.borrowing import BorrowingParams, simulate_with_borrowing  # noqa: E402
from sim.phonology import PhonologyParams  # noqa: E402
from sim.sdollo import SDolloParams  # noqa: E402

OUT = ROOT / "data/derived/m5_cognate_eval"
PHON = dict(min_syllables=1, max_syllables=1, p_final_consonant=1.0, stable_segments="aiuhmnk", stable_weight=0.05)
RATES = (0.0, 3e-5, 7e-5)
METHODS = {"LexStat 0.75 (whole word)": ("lexstat", 0.75), "SCA 0.55 (whole word)": ("sca", 0.55),
           "partial SCA 0.35": ("partial", 0.35), "partial SCA 0.45": ("partial", 0.45),
           "partial SCA 0.55": ("partial", 0.55),
           "partial LexStat 0.55": ("partial-lexstat", 0.55), "partial LexStat 0.65": ("partial-lexstat", 0.65),
           "partial LexStat 0.75": ("partial-lexstat", 0.75), "partial LexStat 0.85": ("partial-lexstat", 0.85)}
N_REPS = 4


def bcubed(true: dict, pred: dict) -> tuple[float, float, float]:
    bt, bp = collections.defaultdict(set), collections.defaultdict(set)
    for k in true:
        bt[true[k]].add(k); bp[pred[k]].add(k)
    P = np.mean([len(bt[true[k]] & bp[pred[k]]) / len(bp[pred[k]]) for k in true])
    R = np.mean([len(bt[true[k]] & bp[pred[k]]) / len(bt[true[k]]) for k in true])
    return float(P), float(R), float(2 * P * R / (P + R))


def job(args):
    rate, i = args
    prior = copy.deepcopy(load_prior(ROOT / "infer/beast/m1_strict_prior.json"))
    prior["n_tips"], prior["n_concepts"] = 31, 40
    seed = 6000 + 10 * i
    t = draw_truth(prior, np.random.default_rng(seed))
    mu = t["loss_rate"]
    res = simulate_with_borrowing(t["tree"], SDolloParams(40, mu, mu), BorrowingParams(0.5 * mu),
                                  seed=seed + 1, phonology=PhonologyParams(**PHON, suffix_rate=rate))
    obs = res.observed()
    forms, truth = [], {}
    for r, l in enumerate(obs.leaf_names):
        for j in np.flatnonzero(obs.presence[r]):
            fid = f"{l}|{j}"
            w = obs.forms[r, j]
            forms.append({"ID": fid, "Language_ID": l, "Parameter_ID": str(obs.class_concept[j]),
                          "Form": w, "Segments": " ".join(w)})
            truth[fid] = str(j)
    suffixed = np.mean([any(f["Form"].endswith(x) for x in ("il", "al", "el", "ax")) for f in forms])
    out = {"rate": rate, "rep": i, "suffixed": float(suffixed)}
    for name, (m, thr) in METHODS.items():
        if m == "partial":
            cog = partial_cognates(forms, thr)
        elif m == "partial-lexstat":
            cog = partial_cognates(forms, thr, method="lexstat")
        else:
            cog = lexstat_cognates(forms, thr, method=m)
        out[name] = bcubed(truth, cog)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(r, i) for r in RATES for i in range(N_REPS)]
    with ProcessPoolExecutor(6) as pool:
        rows = list(pool.map(job, jobs))
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nB-cubed F (precision / recall), mean of {N_REPS} replicates, 31 languages x 40 concepts")
    print(f"{'method':28s} " + " ".join(f"{'suffix ' + format(r, 'g'):>22s}" for r in RATES))
    for name in METHODS:
        cells = []
        for r in RATES:
            v = np.array([x[name] for x in rows if x["rate"] == r])
            cells.append(f"{v[:, 2].mean():.3f} ({v[:, 0].mean():.2f}/{v[:, 1].mean():.2f})")
        print(f"{name:28s} " + " ".join(f"{c:>22s}" for c in cells))
    print("words ending in a suffix: " + ", ".join(
        f"rate {r:g}: {np.mean([x['suffixed'] for x in rows if x['rate'] == r]):.1%}" for r in RATES))


if __name__ == "__main__":
    main()
