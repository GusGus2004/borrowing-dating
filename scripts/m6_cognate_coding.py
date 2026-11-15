"""Milestone 6: dating bias from automated cognate coding vs from borrowing, on known truth.

Milestone 5b found that, with identical calibrations, the real Mayan root is
about 2,000 years younger with ASJP 40 / LexStat cognates than with Kaufman's
expert etyma. This asks how large that effect is when the truth is known,
and how it compares with the borrowing bias of milestone 2b.

Design: Mayan-shaped simulations (31 languages, 40 concepts, milestone 1
root and loss-rate priors, Mayan-like CVC phonology as in milestone 5's
detector work), with no borrowing (kappa 0) or uniform borrowing at
kappa = 1 (rate = loss rate). Each simulated dataset is fitted twice:

* ``true``     the true cognate classes (borrowed words in the donor's class);
* ``lexstat``  cognate sets re-inferred from the simulated words by LexStat
               at threshold 0.75, exactly as for the real ASJP data.

Inference uses the milestone 2b (Mayan-like) design: root Uniform(1500,
9000), Yule estimated, up to two internal calibrations on clades aged 8-40%
of the root, each Normal(true age, 20%). Absent concepts are coded 0, as in
milestones 1-2. 5M states per fit.

Subcommands: prepare [--reps N] | run [--jobs N] | summarise
"""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analysis.mcmc import ess, hpd, read_log  # noqa: E402
from infer.beastrun import run_all  # noqa: E402
from infer.beastxml import load_prior, render_xml, simulate_replicate  # noqa: E402
from m2b_calibration import pick_calibrations, prior_for  # noqa: E402
from pipeline.cldf import CognateMatrix, read_cognate_matrix  # noqa: E402
from sim.tree import Tree  # noqa: E402

OUT = ROOT / "data/derived/m6"
FIG = ROOT / "figures"
PHON = dict(min_syllables=1, max_syllables=1, p_final_consonant=1.0, stable_segments="aiuhmnk", stable_weight=0.05)
KAPPAS = (0.0, 1.0)
CODINGS = ("true", "lexstat")
THRESHOLD = 0.75
CHAIN = 5_000_000


def gen_prior() -> dict:
    p = copy.deepcopy(load_prior(ROOT / "infer/beast/m1_strict_prior.json"))
    p["n_tips"], p["n_concepts"] = 31, 40
    return p


def bcubed(true: dict, pred: dict) -> tuple[float, float, float]:
    bt, bp = collections.defaultdict(set), collections.defaultdict(set)
    for k in true:
        bt[true[k]].add(k); bp[pred[k]].add(k)
    P = np.mean([len(bt[true[k]] & bp[pred[k]]) / len(bp[pred[k]]) for k in true])
    R = np.mean([len(bt[true[k]] & bp[pred[k]]) / len(bt[true[k]]) for k in true])
    return float(P), float(R), float(2 * P * R / (P + R))


def lexstat_matrix(sim: Path, languages: list[str]) -> tuple[CognateMatrix, dict]:
    from pipeline.asjp import lexstat_cognates
    forms = [{"ID": r["ID"], "Language_ID": r["Language_ID"], "Parameter_ID": r["Parameter_ID"],
              "Form": r["Form"], "Segments": r["Segments"] or " ".join(r["Form"])} for r in csv.DictReader(open(sim / "forms.csv"))]
    truth = {r["Form_ID"]: r["Cognateset_ID"] for r in csv.DictReader(open(sim / "cognates.csv"))}
    cog = lexstat_cognates(forms, threshold=THRESHOLD)
    concept = {f["ID"]: f["Parameter_ID"] for f in forms}
    lang = {f["ID"]: f["Language_ID"] for f in forms}
    sets = sorted(set(cog.values()))
    col = {s: j for j, s in enumerate(sets)}
    row = {l: i for i, l in enumerate(languages)}
    m = np.zeros((len(languages), len(sets)), dtype=np.int8)
    set_concept = {}
    for fid, s in cog.items():
        m[row[lang[fid]], col[s]] = 1
        set_concept[s] = concept[fid]
    P, R, F = bcubed(truth, cog)
    return (CognateMatrix(list(languages), sets, [set_concept[s] for s in sets], m),
            {"P": P, "R": R, "F": F, "n_sets_true": len(set(truth.values())), "n_sets_lexstat": len(sets)})


def prepare_one(args) -> str:
    kappa, i = args
    seed = 880000 + 10 * i  # same tree and parameters at both kappa
    sim = OUT / "sims" / f"kappa_{kappa:g}" / f"rep_{i:03d}"
    if not (sim / "truth.json").exists():
        sim.mkdir(parents=True, exist_ok=True)
        simulate_replicate(gen_prior(), seed, sim, borrowing=dict(contact="uniform", rate_over_loss=kappa),
                           phonology=PHON)
    tree = Tree.from_newick((sim / "tree.nwk").read_text())
    prior = prior_for(pick_calibrations(tree))
    prior["mcmc"] = dict(prior["mcmc"], chain_length=CHAIN, log_every=2500, tree_log_every=25000)
    true_cm = read_cognate_matrix(sim / "Wordlist-metadata.json", concept_level_missing=False)
    lex_cm, score = lexstat_matrix(sim, true_cm.languages)
    for coding, cm in (("true", true_cm), ("lexstat", lex_cm)):
        d = OUT / f"{coding}_kappa_{kappa:g}" / f"rep_{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "beast.xml").write_text(render_xml(cm, prior, stem="beast"))
        (d / "prior.json").write_text(json.dumps(prior, indent=2))
        truth = json.loads((sim / "truth.json").read_text())
        truth["n_cognatesets_coded"] = len(cm.cognatesets)
        truth["coding_score"] = score if coding == "lexstat" else {"P": 1.0, "R": 1.0, "F": 1.0}
        (d / "truth.json").write_text(json.dumps(truth, indent=2))
    return f"kappa {kappa:g} rep {i}: {len(prior['calibrations'])} calibrations; LexStat F {score['F']:.3f}, " \
           f"sets {score['n_sets_true']} true -> {score['n_sets_lexstat']} LexStat"


def cmd_prepare(a) -> None:
    jobs = [(k, i) for i in range(a.reps) for k in KAPPAS]
    with ProcessPoolExecutor(a.jobs) as pool:
        for msg in pool.map(prepare_one, jobs):
            print(msg, flush=True)


def cells() -> list[str]:
    return [f"{c}_kappa_{k:g}" for k in KAPPAS for c in CODINGS]


def cmd_run(a) -> None:
    reps = [r for c in cells() for r in sorted((OUT / c).glob("rep_*"))]
    reps.sort(key=lambda r: (r.name, r.parent.name))  # interleave cells so partial results are balanced
    run_all(reps, lambda r: 271828 + int(r.name.split("_")[1]), a.jobs)


def cmd_summarise(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    burn = load_prior(ROOT / "infer/beast/m1_strict_prior.json")["mcmc"]["burnin_frac"]
    rows = []
    for cell in cells():
        for rep in sorted((OUT / cell).glob("rep_*")):
            if not (rep / "done").exists():
                continue
            t = json.loads((rep / "truth.json").read_text())
            log = read_log(rep / "beast.log", burnin_frac=burn)
            x = log["Tree.height"]
            lo, hi = hpd(x)
            rows.append({"cell": cell, "rep": rep.name, "true_root": t["root_age"], "median": float(np.median(x)),
                         "hpd_lo": lo, "hpd_hi": hi, "covered": bool(lo <= t["root_age"] <= hi), "ess": ess(x),
                         "true_mu": t["params"]["loss_rate"], "mu_median": float(np.median(log["deathRate"])),
                         "borrowed_fraction": t.get("borrowed_fraction", 0.0), "F": t["coding_score"]["F"],
                         "n_sets": t["n_cognatesets_coded"]})
    if not rows:
        sys.exit("no finished fits")
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    base = {r["rep"]: r for r in rows if r["cell"] == "true_kappa_0"}
    se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    print(f"{'cell':18s} {'n':>3s} {'borrowed':>8s} {'B3 F':>5s} {'sets':>5s} {'root bias':>9s} {'±se':>5s} "
          f"{'Δ vs true,κ0':>12s} {'±se':>5s} {'coverage':>8s} {'μ bias':>7s} {'minESS':>6s}")
    summary = []
    for cell in cells():
        rs = [r for r in rows if r["cell"] == cell]
        if not rs:
            continue
        rb = np.array([(r["median"] - r["true_root"]) / r["true_root"] for r in rs])
        dd = np.array([(r["median"] - base[r["rep"]]["median"]) / r["true_root"] for r in rs if r["rep"] in base])
        mb = np.array([(r["mu_median"] - r["true_mu"]) / r["true_mu"] for r in rs])
        s = {"cell": cell, "n": len(rs), "borrowed_fraction": float(np.mean([r["borrowed_fraction"] for r in rs])),
             "F": float(np.mean([r["F"] for r in rs])), "n_sets": float(np.mean([r["n_sets"] for r in rs])),
             "rel_bias": float(rb.mean()), "rel_bias_se": se(rb),
             "paired_delta": float(dd.mean()) if len(dd) else float("nan"), "paired_delta_se": se(dd),
             "coverage": float(np.mean([r["covered"] for r in rs])), "mu_rel_bias": float(mb.mean()),
             "min_ess": float(min(r["ess"] for r in rs))}
        summary.append(s)
        print(f"{cell:18s} {s['n']:3d} {s['borrowed_fraction']:8.3f} {s['F']:5.3f} {s['n_sets']:5.0f} "
              f"{s['rel_bias']:9.3f} {s['rel_bias_se']:5.3f} {s['paired_delta']:12.3f} {s['paired_delta_se']:5.3f} "
              f"{s['coverage']:8.2f} {s['mu_rel_bias']:7.3f} {s['min_ess']:6.0f}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    lab = {"true_kappa_0": "true cognates\nno borrowing", "lexstat_kappa_0": "LexStat cognates\nno borrowing",
           "true_kappa_1": "true cognates\nborrowing κ=1", "lexstat_kappa_1": "LexStat cognates\nborrowing κ=1"}
    col = {"true": "#4a7bb7", "lexstat": "#d1495b"}
    for k, s in enumerate(summary):
        ax.errorbar(k, s["rel_bias"] * 100, yerr=2 * s["rel_bias_se"] * 100, fmt="o", capsize=4,
                    color=col[s["cell"].split("_")[0]])
        ax.annotate(f"cov {s['coverage']:.2f}", (k, s["rel_bias"] * 100), textcoords="offset points",
                    xytext=(8, -3), fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(summary)), [lab[s["cell"]] for s in summary], fontsize=8)
    ax.set_ylabel("root age relative bias (%), mean ± 2 se")
    ax.set_title("Mayan-shaped simulations (31 languages × 40 concepts), 2 recent calibrations")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "m6_cognate_coding.png", dpi=150)
    print(f"figure -> {FIG / 'm6_cognate_coding.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--reps", type=int, default=20)
    p.add_argument("--jobs", type=int, default=6)
    p = sub.add_parser("run"); p.add_argument("--jobs", type=int, default=7)
    sub.add_parser("summarise")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
