"""Milestone 3b: a level-3 loan detector (forms + tree) against level 1.

Simulated datasets as in milestone 3 (milestone 1 prior, kappa = 0.5,
default sound-change rate), under two phonologies:

* ``default``: 2-3 syllable roots, all segments equally changeable;
* ``mayan``: CVC roots and stable segments (a, i, u, h, m, n, k at weight
  0.05), the setting where level 1 dropped to AUC 0.67.

Scores per entry (language, cognate set), see ``hmm.phylo``:
level 1 (mean pair z), set-centred form scores, Dollo patchiness on a UPGMA
tree from cognate sharing (realistic) or on the true tree (oracle), and a
logistic combination trained on half the replicates and tested on the other
half (and vice versa).

Outputs data/derived/m3b/results.json and figures/m3b_detector.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmm.pairhmm import pair_scores  # noqa: E402
from hmm.phylo import (FEATURES, feature_matrix, fit_logistic, heights_from_tree,  # noqa: E402
                       patchiness_scores, predict_logistic, set_centred_scores, transfer_scores, upgma_tree)
from infer.beastxml import draw_truth, load_prior  # noqa: E402
from sim.borrowing import BorrowingParams, _ancestors, _mrca_age, simulate_with_borrowing  # noqa: E402
from sim.phonology import PhonologyParams  # noqa: E402
from sim.sdollo import SDolloParams  # noqa: E402

OUT = ROOT / "data/derived/m3b"
FIG = ROOT / "figures"
CONTACT = {"uniform": dict(contact="uniform"), "distance": dict(contact="distance", distance_scale=1000.0)}
PHON = {"default": {}, "mayan": dict(min_syllables=1, max_syllables=1, p_final_consonant=1.0,
                                     stable_segments="aiuhmnk", stable_weight=0.05)}
SEP_BANDS = [(0, 500), (500, 1500), (1500, 1e9)]


def auc(s, lab) -> float:
    s = np.asarray(s, float); lab = np.asarray(lab, bool)
    if lab.all() or not lab.any():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def one_dataset(prior, seed, kappa, contact, phon):
    rng = np.random.default_rng(seed)
    t = draw_truth(prior, rng)
    mu = t["loss_rate"]
    tree = t["tree"]
    res = simulate_with_borrowing(
        tree, SDolloParams(prior["n_concepts"], mu * prior["classes_per_concept"], mu),
        BorrowingParams(kappa * mu, **CONTACT[contact]), seed=seed + 1,
        phonology=PhonologyParams(**PHON[phon]))
    keep = res.presence.any(axis=0)
    idx = np.flatnonzero(keep)
    obs = res.observed()
    B = res.extra["borrowed"][:, keep]
    wl = {l: {str(j): obs.forms[i, j] for j in np.flatnonzero(obs.presence[i])}
          for i, l in enumerate(obs.leaf_names)}
    concept_of = {str(j): str(c) for j, c in enumerate(obs.class_concept)}
    form = set_centred_scores(pair_scores(wl))
    Ht = heights_from_tree(tree, mu)
    Hu = upgma_tree(wl, concept_of)
    merge = lambda p, q: {e: {**p[e], **q[e]} for e in p}
    patch_true = merge(patchiness_scores(wl, concept_of, list(Ht)), transfer_scores(wl, concept_of, Ht))
    patch_upgma = merge(patchiness_scores(wl, concept_of, list(Hu)), transfer_scores(wl, concept_of, Hu))
    entries, labels, seps = [], [], []
    for i, l in enumerate(obs.leaf_names):
        anc = {n.name for n in _ancestors(tree.node_by_name(l))}
        for j in np.flatnonzero(obs.presence[i]):
            e = (l, str(j))
            if e not in form:
                continue  # singleton set: no form evidence
            sep = None
            if B[i, j]:
                ev = min((x for x in res.extra["borrow_events"]
                          if x["class"] == idx[j] and x["recipient"] in anc), key=lambda x: x["age"])
                sep = _mrca_age(tree.node_by_name(ev["recipient"]), tree.node_by_name(ev["donor"])) - ev["age"]
            entries.append(e); labels.append(bool(B[i, j])); seps.append(sep)
    n = len(obs.leaf_names)
    return {"X_upgma": feature_matrix(form, patch_upgma, entries, n),
            "X_true": feature_matrix(form, patch_true, entries, n),
            "y": np.array(labels), "sep": seps}


def evaluate(data: list[dict], key: str) -> dict:
    """Per-feature AUC plus a 2-fold (by replicate) logistic combination."""
    out = {}
    X = np.vstack([d[key] for d in data]); y = np.concatenate([d["y"] for d in data])
    for k, f in enumerate(FEATURES):
        out[f] = auc(X[:, k], y)
    # standardise with training-fold statistics, fit, predict on the other fold
    pred = [None] * len(data)
    half = [list(range(0, len(data), 2)), list(range(1, len(data), 2))]
    weights = []
    for tr, te in (half, half[::-1]):
        Xtr = np.vstack([data[i][key] for i in tr]); ytr = np.concatenate([data[i]["y"] for i in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        w = fit_logistic((Xtr - mu) / sd, ytr.astype(float))
        weights.append(w.tolist())
        for i in te:
            pred[i] = predict_logistic(w, (data[i][key] - mu) / sd)
    out["combined"] = auc(np.concatenate(pred), y)
    out["combined_per_rep"] = [auc(p, d["y"]) for p, d in zip(pred, data)]
    out["weights"] = weights
    # by separation, for level 1 and the combination
    seps = [s for d in data for s in d["sep"]]
    P = np.concatenate(pred); L1 = X[:, FEATURES.index("mean")]
    band = {}
    for lo, hi in SEP_BANDS:
        sel = np.array([(not lab) or (s is not None and lo <= s < hi) for lab, s in zip(y, seps)])
        lab = f">{lo:g}" if hi >= 1e9 else f"{lo:g}-{hi:g}"
        band[lab] = {"n": int((y & sel).sum()), "level1": auc(L1[sel], y[sel]), "combined": auc(P[sel], y[sel])}
    out["by_separation"] = band
    # precision / recall of the combination at the threshold flagging as many entries as there are loans
    thr = np.quantile(P, 1 - y.mean())
    flag = P >= thr
    out["precision_at_base_rate"] = float((flag & y).sum() / flag.sum())
    out["recall_at_base_rate"] = float((flag & y).sum() / y.sum())
    flag1 = L1 >= np.quantile(L1, 1 - y.mean())
    out["level1_precision_at_base_rate"] = float((flag1 & y).sum() / flag1.sum())
    out["_pred"] = P; out["_y"] = y; out["_l1"] = L1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--seed0", type=int, default=9000)
    a = ap.parse_args()
    prior = load_prior(ROOT / "infer/beast/m1_strict_prior.json")
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for contact in CONTACT:
        for phon in PHON:
            data = [one_dataset(prior, a.seed0 + 10 * k, a.kappa, contact, phon) for k in range(a.reps)]
            for key, tree in (("X_upgma", "UPGMA"), ("X_true", "true tree")):
                r = evaluate(data, key)
                r.update(contact=contact, phonology=phon, tree=tree,
                         n_entries=int(sum(len(d["y"]) for d in data)), n_borrowed=int(sum(d["y"].sum() for d in data)))
                results.append(r)
                feats = "  ".join(f"{f} {r[f]:.2f}" for f in FEATURES)
                bands = "  ".join(f"{k}: {v['level1']:.2f}->{v['combined']:.2f}" for k, v in r["by_separation"].items())
                print(f"{contact:8s} {phon:8s} {tree:9s} | {feats} | COMBINED {r['combined']:.3f} | "
                      f"prec {r['level1_precision_at_base_rate']:.2f}->{r['precision_at_base_rate']:.2f} | {bands}",
                      flush=True)
    (OUT / "results.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in results], indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3), sharey=True)
    for ax, (contact, phon) in zip(axes, [(c, p) for c in CONTACT for p in PHON]):
        for r, col in zip([x for x in results if x["contact"] == contact and x["phonology"] == phon],
                          ["#d1495b", "#8a5a9e"]):
            for s, ls, nm in ((r["_l1"], ":", "level 1"), (r["_pred"], "-", f"level 3 ({r['tree']})")):
                o = np.argsort(-s); y = r["_y"][o]
                tp = np.concatenate([[0], np.cumsum(y) / y.sum()]); fp = np.concatenate([[0], np.cumsum(~y) / (~y).sum()])
                if nm == "level 1" and col != "#d1495b":
                    continue
                ax.plot(fp, tp, ls=ls, color="#555555" if nm == "level 1" else col,
                        label=f"{nm}: AUC {auc(s, r['_y']):.2f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.6)
        ax.set_title(f"{contact} contact, {phon} phonology", fontsize=10)
        ax.set_xlabel("false positive rate")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("true positive rate")
    fig.suptitle(f"Milestone 3b: level 1 (forms only) vs level 3 (forms + tree), κ = {a.kappa}, {a.reps} replicates, held-out folds")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "m3b_detector.png", dpi=150)
    print(f"figure -> {FIG / 'm3b_detector.png'}")


if __name__ == "__main__":
    main()
