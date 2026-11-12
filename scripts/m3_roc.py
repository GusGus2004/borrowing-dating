"""Milestone 3: ROC for form-level (pair HMM) borrowing detection on simulated data.

Trees and loss rates are drawn from the milestone 1 strict prior; borrowing
at kappa * mu (default kappa = 0.5, ~10% of entries borrowed) under each
contact model; phoneme strings evolve with regular sound change at each rate
in --rates (changes per lineage per year) and sporadic change at 2e-5 per
word per year. Detection uses only the leaf forms and cognate sets (no tree).

Outputs data/derived/m3/results.json and figures/m3_roc.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmm.pairhmm import entry_scores  # noqa: E402
from infer.beastxml import draw_truth, load_prior  # noqa: E402
from sim.borrowing import BorrowingParams, _ancestors, _mrca_age, simulate_with_borrowing  # noqa: E402
from sim.phonology import PhonologyParams  # noqa: E402
from sim.sdollo import SDolloParams  # noqa: E402

OUT = ROOT / "data/derived/m3"
FIG = ROOT / "figures"
CONTACT = {"uniform": dict(contact="uniform"), "distance": dict(contact="distance", distance_scale=1000.0)}
SEP_BANDS = [(0, 500), (500, 1500), (1500, 1e9)]


def band_label(lo, hi) -> str:
    return f">{lo:g}" if hi >= 1e9 else f"{lo:g}-{hi:g}"


def auc(s, lab) -> float:
    s = np.asarray(s, float); lab = np.asarray(lab, bool)
    if lab.all() or not lab.any():
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum()))


def roc(s, lab):
    s = np.asarray(s, float); lab = np.asarray(lab, bool)
    o = np.argsort(-s)
    tp = np.concatenate([[0], np.cumsum(lab[o])]) / lab.sum()
    fp = np.concatenate([[0], np.cumsum(~lab[o])]) / (~lab).sum()
    return fp, tp


def one_dataset(prior, seed, kappa, contact, rate):
    rng = np.random.default_rng(seed)
    t = draw_truth(prior, rng)
    mu = t["loss_rate"]
    tree = t["tree"]
    res = simulate_with_borrowing(
        tree, SDolloParams(prior["n_concepts"], mu * prior["classes_per_concept"], mu),
        BorrowingParams(kappa * mu, **CONTACT[contact]), seed=seed + 1,
        phonology=PhonologyParams(change_rate=rate))
    keep = res.presence.any(axis=0)
    idx = np.flatnonzero(keep)
    obs = res.observed()
    B = res.extra["borrowed"][:, keep]
    wl = {l: {str(j): obs.forms[i, j] for j in np.flatnonzero(obs.presence[i])}
          for i, l in enumerate(obs.leaf_names)}
    sc = entry_scores(wl)
    rows = []
    for i, l in enumerate(obs.leaf_names):
        anc = {n.name for n in _ancestors(tree.node_by_name(l))}
        for j in np.flatnonzero(obs.presence[i]):
            v = sc[(l, str(j))]
            if not np.isfinite(v):
                continue
            sep = None
            if B[i, j]:
                ev = min((e for e in res.extra["borrow_events"]
                          if e["class"] == idx[j] and e["recipient"] in anc), key=lambda e: e["age"])
                sep = _mrca_age(tree.node_by_name(ev["recipient"]), tree.node_by_name(ev["donor"])) - ev["age"]
            rows.append((float(v), bool(B[i, j]), sep))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--rates", type=float, nargs="+", default=[1 / 400, 1 / 200, 1 / 100, 1 / 50])
    ap.add_argument("--seed0", type=int, default=9000)
    a = ap.parse_args()
    prior = load_prior(ROOT / "infer/beast/m1_strict_prior.json")
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for contact in CONTACT:
        for rate in a.rates:
            per_rep, pooled = [], []
            for k in range(a.reps):
                rows = one_dataset(prior, a.seed0 + 10 * k, a.kappa, contact, rate)
                pooled += rows
                per_rep.append(auc([r[0] for r in rows], [r[1] for r in rows]))
            neg = [r[0] for r in pooled if not r[1]]
            band = {}
            for lo, hi in SEP_BANDS:
                pos = [r[0] for r in pooled if r[1] and lo <= r[2] < hi]
                band[band_label(lo, hi)] = {"n": len(pos), "auc": auc(pos + neg, [1] * len(pos) + [0] * len(neg))}
            pr = np.array([x for x in per_rep if np.isfinite(x)])
            rec = {"contact": contact, "change_rate": rate, "kappa": a.kappa, "reps": a.reps,
                   "auc_mean": float(pr.mean()), "auc_se": float(pr.std(ddof=1) / np.sqrt(len(pr))),
                   "n_entries": len(pooled), "n_borrowed": int(sum(r[1] for r in pooled)),
                   "auc_by_separation": band,
                   "scores": [r[0] for r in pooled], "labels": [r[1] for r in pooled]}
            results.append(rec)
            bands = "  ".join(f"{k}y: {v['auc']:.2f} (n={v['n']})" for k, v in band.items())
            print(f"{contact:8s} 1/{1 / rate:<4.0f}  AUC {rec['auc_mean']:.3f} ± {rec['auc_se']:.3f}  "
                  f"borrowed {rec['n_borrowed']}/{rec['n_entries']}  | by separation {bands}", flush=True)
    (OUT / "results.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("scores", "labels")} for r in results], indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    cmap = plt.get_cmap("viridis")
    for ax, contact in zip(axes[:2], CONTACT):
        rs = [r for r in results if r["contact"] == contact]
        for n, r in enumerate(rs):
            fp, tp = roc(r["scores"], r["labels"])
            ax.plot(fp, tp, color=cmap(n / max(len(rs) - 1, 1)),
                    label=f"1 change / {1 / r['change_rate']:.0f} y: AUC {r['auc_mean']:.2f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.set_title(f"{contact} contact (pooled over {a.reps} replicates)")
        ax.legend(fontsize=8, loc="lower right")
    ax = axes[2]
    for contact, mk in zip(CONTACT, "os"):
        rs = [r for r in results if r["contact"] == contact]
        for (lo, hi), ls in zip(SEP_BANDS, ["-", "--", ":"]):
            key = band_label(lo, hi)
            ax.plot([r["change_rate"] * 1000 for r in rs], [r["auc_by_separation"][key]["auc"] for r in rs],
                    marker=mk, ls=ls, label=f"{contact}, separation {key} y")
    ax.axhline(0.5, color="k", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("regular sound changes per lineage per 1000 y")
    ax.set_ylabel("AUC"); ax.set_title("AUC by recipient-donor separation\n(MRCA age - loan age)")
    ax.legend(fontsize=7)
    fig.suptitle(f"Milestone 3: pair-HMM borrowing detection (κ = {a.kappa}, form evidence only)")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "m3_roc.png", dpi=150)
    print(f"figure -> {FIG / 'm3_roc.png'}")


if __name__ == "__main__":
    main()
