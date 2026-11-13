"""Milestone 2b: bias under a Mayan-like calibration design.

Milestone 2 inferred with the generating root prior N(4000, 500), so most of
the borrowing bias went into the loss rate. Real analyses (milestone 5) give
the root a vague prior and calibrate only a few recent internal clades. This
refits the milestone 2 replicates (same simulated data, no new simulation)
under that design:

* root ~ Uniform(1500, 9000) (as in milestone 5), Yule rate estimated;
* loss-rate prior = the generating prior (so only the calibration design changes);
* up to two internal calibrations: greedily the oldest non-nested clades with
  true age in [0.08, 0.40] x root, each Normal(true age, 0.2 x true age),
  monophyletic. Mayan: Yucatecan/root ~ 0.25, Huastecan/root ~ 0.17.

Subcommands: prepare | run [--jobs N] | summarise
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.mcmc import ess, hpd, read_log  # noqa: E402
from infer.beastrun import run_all  # noqa: E402
from infer.beastxml import load_prior, render_xml  # noqa: E402
from pipeline.cldf import read_cognate_matrix  # noqa: E402
from sim.tree import Tree  # noqa: E402

M2 = ROOT / "data/derived/m2"
OUT = ROOT / "data/derived/m2b"
FIG = ROOT / "figures"
BASE = load_prior(ROOT / "infer/beast/m1_strict_prior.json")
CELLS = ["kappa_0", "uniform_kappa_1", "uniform_kappa_2", "distance_kappa_2"]
WINDOW = (0.08, 0.40)
CAL_SD = 0.2


def pick_calibrations(tree: Tree, k: int = 2) -> list[dict]:
    r = tree.root_age
    cands = [n for n in tree.internal_nodes()
             if not n.is_root and WINDOW[0] * r <= n.age <= WINDOW[1] * r]
    cands.sort(key=lambda n: -n.age)
    chosen, used = [], set()
    for n in cands:
        tips = {l.name for l in _leaves(n)}
        if tips & used:
            continue
        chosen.append({"name": f"cal{len(chosen) + 1}", "taxa": sorted(tips), "monophyletic": True,
                       "dist": {"dist": "normal", "mean": n.age, "sd": CAL_SD * n.age},
                       "source": f"simulation truth {n.name}, age {n.age:.0f}"})
        used |= tips
        if len(chosen) == k:
            break
    return chosen


def _leaves(n):
    return [n] if n.is_leaf else [l for c in n.children for l in _leaves(c)]


def prior_for(cals: list[dict]) -> dict:
    p = copy.deepcopy(BASE)
    p["root_age"] = {"dist": "uniform", "lower": 1500.0, "upper": 9000.0}
    p["yule_birth_rate"] = {"dist": "lognormal", "meanlog": -6.9, "sdlog": 1.5}
    p["calibrations"] = cals
    return p


def cmd_prepare(a) -> None:
    n_cal = []
    for cell in CELLS:
        for src in sorted((M2 / cell).glob("rep_*")):
            rep = OUT / cell / src.name
            rep.mkdir(parents=True, exist_ok=True)
            tree = Tree.from_newick((src / "tree.nwk").read_text())
            cals = pick_calibrations(tree)
            n_cal.append(len(cals))
            prior = prior_for(cals)
            cm = read_cognate_matrix(src / "Wordlist-metadata.json", concept_level_missing=False)
            (rep / "beast.xml").write_text(render_xml(cm, prior, stem="beast"))
            (rep / "prior.json").write_text(json.dumps(prior, indent=2))
            (rep / "truth.json").write_text((src / "truth.json").read_text())
    print(f"prepared {len(n_cal)} fits; calibrations per fit: "
          f"{dict(zip(*np.unique(n_cal, return_counts=True)))}")


def cmd_run(a) -> None:
    reps = [r for c in CELLS for r in sorted((OUT / c).glob("rep_*"))]
    reps.sort(key=lambda r: (r.name, r.parent.name))
    run_all(reps, lambda r: 104729 + int(r.name.split("_")[1]), a.jobs)


def cmd_summarise(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    burn = BASE["mcmc"]["burnin_frac"]
    rows = []
    for cell in CELLS:
        for rep in sorted((OUT / cell).glob("rep_*")):
            if not (rep / "done").exists():
                continue
            t = json.loads((rep / "truth.json").read_text())
            log = read_log(rep / "beast.log", burnin_frac=burn)
            x = log["Tree.height"]
            lo, hi = hpd(x)
            rows.append({"cell": cell, "rep": rep.name, "borrowed_fraction": t.get("borrowed_fraction", 0.0),
                         "true_root": t["root_age"], "median": float(np.median(x)), "hpd_lo": lo, "hpd_hi": hi,
                         "covered": bool(lo <= t["root_age"] <= hi), "ess": ess(x),
                         "true_mu": t["params"]["loss_rate"], "mu_median": float(np.median(log["deathRate"]))})
    if not rows:
        sys.exit("no finished fits")
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    base = {r["rep"]: r for r in rows if r["cell"] == "kappa_0"}
    m2 = {(r["contact"] if r["kappa"] else "none", r["kappa"], r["rep"]): r
          for r in json.loads((M2 / "results.json").read_text())}
    print(f"{'cell':18s} {'n':>3s} {'borrowed':>8s} {'root rel.bias':>13s} {'±se':>6s} {'paired Δ':>9s} {'±se':>6s} "
          f"{'coverage':>8s} {'μ rel.bias':>10s} {'M2 root bias':>12s} {'minESS':>6s}")
    summary = []
    for cell in CELLS:
        rs = [r for r in rows if r["cell"] == cell]
        if not rs:
            continue
        rb = np.array([(r["median"] - r["true_root"]) / r["true_root"] for r in rs])
        dd = np.array([(r["median"] - base[r["rep"]]["median"]) / r["true_root"] for r in rs if r["rep"] in base])
        mb = np.array([(r["mu_median"] - r["true_mu"]) / r["true_mu"] for r in rs])
        contact, kappa = ("none", 0.0) if cell == "kappa_0" else (cell.split("_")[0], float(cell.split("_")[-1]))
        m2b = np.array([(m2[(contact, kappa, r["rep"])]["median"] - r["true_root"]) / r["true_root"] for r in rs])
        se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        s = {"cell": cell, "n": len(rs), "borrowed_fraction": float(np.mean([r["borrowed_fraction"] for r in rs])),
             "rel_bias": float(rb.mean()), "rel_bias_se": se(rb),
             "paired_delta": float(dd.mean()) if len(dd) else float("nan"), "paired_delta_se": se(dd),
             "coverage": float(np.mean([r["covered"] for r in rs])), "mu_rel_bias": float(mb.mean()),
             "m2_rel_bias": float(m2b.mean()), "min_ess": float(min(r["ess"] for r in rs))}
        summary.append(s)
        print(f"{cell:18s} {s['n']:3d} {s['borrowed_fraction']:8.3f} {s['rel_bias']:13.3f} {s['rel_bias_se']:6.3f} "
              f"{s['paired_delta']:9.3f} {s['paired_delta_se']:6.3f} {s['coverage']:8.2f} {s['mu_rel_bias']:10.3f} "
              f"{s['m2_rel_bias']:12.3f} {s['min_ess']:6.0f}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = np.arange(len(summary))
    ax.errorbar(xs - 0.1, [s["m2_rel_bias"] * 100 for s in summary], fmt="o", color="#999999",
                label="M2 design (root prior N(4000, 500))")
    ax.errorbar(xs + 0.1, [s["rel_bias"] * 100 for s in summary], yerr=[2 * s["rel_bias_se"] * 100 for s in summary],
                fmt="o", color="#d1495b", capsize=3, label="Mayan-like design (vague root, 2 recent calibrations)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs, [f"{s['cell']}\n({s['borrowed_fraction'] * 100:.0f}% borrowed)" for s in summary], fontsize=8)
    ax.set_ylabel("root age relative bias (%)")
    ax.legend(fontsize=8)
    ax.set_title("Root-age bias from borrowing under two calibration designs")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "m2b_calibration.png", dpi=150)
    print(f"figure -> {FIG / 'm2b_calibration.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("run"); p.add_argument("--jobs", type=int, default=6)
    sub.add_parser("summarise")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
