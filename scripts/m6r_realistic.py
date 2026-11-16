"""Milestone 6r: milestone 6 repeated with realistic simulated words on the real Mayan tree.

Milestone 6 found that LexStat errors move simulated roots by only -8%,
while on real Mayan words the coding moves the root by millennia (5c). The
simulated words were too easy: LexStat merged as often as it split.
``scripts/sim_realism.py`` calibrates a more realistic setting (MAYAN_V1)
against the real ASJP words with Kaufman's etyma: the PMED MCC tree (29
languages) as the true tree, loss rate 6e-5, one form per language and
concept, short CVC roots with skewed segment frequencies, compounding and
suffixation that keep cognacy, derived (non-cognate) words built on old
roots, sporadic change. LexStat then over-splits (sets x 1.38 vs 1.70 real)
with B-cubed F 0.79 (real 0.80).

Each replicate (same tree, new characters and words) is fitted with the
true cognate sets and with LexStat 0.75 sets, without borrowing and with
uniform borrowing at kappa = 1. Inference as for the real data
(infer/beast/m5_mayan_prior.json: root Uniform(1500, 9000), Yule, loss rate
LogNormal(-9, 1), strict clock), with the real calibration nodes set at their
true ages in the simulation tree: Yucatecan and Cholan-Tzeltalan crowns,
each Normal(true age, 20%). A concept with no form in a language is missing.

Subcommands: prepare [--reps N] | run [--jobs N] | summarise
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from analysis.mcmc import ess, hpd, read_log  # noqa: E402
from infer.beastrun import run_all  # noqa: E402
from infer.beastxml import load_prior, render_xml  # noqa: E402
from m5c_asjp_expert import matrix  # noqa: E402
from sim_realism import MAYAN_V1, bcubed, simulate_forms  # noqa: E402

OUT = ROOT / "data/derived/m6r"
FIG = ROOT / "figures"
BASE = load_prior(ROOT / "infer/beast/m5_mayan_prior.json")
CALS = {"Yucatecan": ["YUK", "ITZ", "LAK", "MOP"], "CholanTzeltalan": ["CHL", "CHR", "TZE", "TZO"]}
KAPPAS = (0.0, 1.0)
CODINGS = ("true", "lexstat")
CHAIN = 5_000_000


def mrca_age(tree, taxa: list[str]) -> float:
    want = set(taxa)
    best = None
    for n in tree.internal_nodes():
        tips = {l.name for l in _leaves(n)}
        if want <= tips and (best is None or n.age < best.age):
            best = n
    return best.age


def _leaves(n):
    return [n] if n.is_leaf else [l for c in n.children for l in _leaves(c)]


def prepare_one(args) -> str:
    kappa, i = args
    from pipeline.asjp import lexstat_cognates
    forms, gold, tree, mu, bf = simulate_forms({**MAYAN_V1, "kappa": kappa}, 7700 + 10 * i)
    with contextlib.redirect_stderr(io.StringIO()):
        lex = lexstat_cognates(forms, 0.75)
    languages = sorted(l.name for l in tree.leaves())
    concept = {f["ID"]: f["concept"] for f in forms}
    prior = copy.deepcopy(BASE)
    prior["calibrations"] = [{"name": n, "taxa": t, "monophyletic": True,
                              "dist": {"dist": "normal", "mean": mrca_age(tree, t), "sd": 0.2 * mrca_age(tree, t)},
                              "source": "simulation truth"} for n, t in CALS.items()]
    prior["mcmc"] = dict(prior["mcmc"], chain_length=CHAIN, log_every=2500, tree_log_every=25000)
    P, R, F = bcubed(gold, lex)
    truth = {"root_age": tree.root_age, "loss_rate": mu, "kappa": kappa, "borrowed_fraction": bf,
             "lexstat": {"P": P, "R": R, "F": F, "sets_true": len(set(gold.values())), "sets_lexstat": len(set(lex.values()))}}
    for coding, cog in (("true", gold), ("lexstat", lex)):
        cm = matrix(languages, forms, concept, cog)
        d = OUT / f"{coding}_kappa_{kappa:g}" / f"rep_{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "beast.xml").write_text(render_xml(cm, prior, stem="beast"))
        (d / "prior.json").write_text(json.dumps(prior, indent=2))
        (d / "truth.json").write_text(json.dumps({**truth, "coding": coding, "n_sets": len(cm.cognatesets)}, indent=2))
    return (f"kappa {kappa:g} rep {i}: borrowed {bf:.2f}; LexStat F {F:.3f}, sets {truth['lexstat']['sets_true']} -> "
            f"{truth['lexstat']['sets_lexstat']}")


def cells() -> list[str]:
    return [f"{c}_kappa_{k:g}" for k in KAPPAS for c in CODINGS]


def cmd_prepare(a) -> None:
    with ProcessPoolExecutor(a.jobs) as pool:
        for msg in pool.map(prepare_one, [(k, i) for i in range(a.reps) for k in KAPPAS]):
            print(msg, flush=True)


def cmd_run(a) -> None:
    reps = sorted((r for c in cells() for r in (OUT / c).glob("rep_*")), key=lambda r: (r.name, r.parent.name))
    run_all(reps, lambda r: 161803 + int(r.name.split("_")[1]), a.jobs)


def cmd_summarise(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for cell in cells():
        for rep in sorted((OUT / cell).glob("rep_*")):
            if not (rep / "done").exists():
                continue
            t = json.loads((rep / "truth.json").read_text())
            log = read_log(rep / "beast.log", burnin_frac=BASE["mcmc"]["burnin_frac"])
            x = log["Tree.height"]
            lo, hi = hpd(x)
            rows.append({"cell": cell, "rep": rep.name, "true_root": t["root_age"], "median": float(np.median(x)),
                         "covered": bool(lo <= t["root_age"] <= hi), "ess": ess(x), "true_mu": t["loss_rate"],
                         "mu_median": float(np.median(log["deathRate"])), "borrowed": t["borrowed_fraction"],
                         "F": t["lexstat"]["F"] if t["coding"] == "lexstat" else 1.0, "n_sets": t["n_sets"]})
    if not rows:
        sys.exit("no finished fits")
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    base = {r["rep"]: r for r in rows if r["cell"] == "true_kappa_0"}
    se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    print(f"true root {rows[0]['true_root']:.0f} BP (fixed tree)")
    print(f"{'cell':16s} {'n':>3s} {'borrowed':>8s} {'F':>5s} {'sets':>5s} {'root median':>11s} {'bias':>7s} {'±se':>5s} "
          f"{'Δ vs true,κ0':>12s} {'±se':>5s} {'cover':>5s} {'μ bias':>7s} {'minESS':>6s}")
    summary = []
    for cell in cells():
        rs = [r for r in rows if r["cell"] == cell]
        if not rs:
            continue
        rb = np.array([(r["median"] - r["true_root"]) / r["true_root"] for r in rs])
        dd = np.array([(r["median"] - base[r["rep"]]["median"]) / r["true_root"] for r in rs if r["rep"] in base])
        s = {"cell": cell, "n": len(rs), "borrowed": float(np.mean([r["borrowed"] for r in rs])),
             "F": float(np.mean([r["F"] for r in rs])), "n_sets": float(np.mean([r["n_sets"] for r in rs])),
             "root_median": float(np.mean([r["median"] for r in rs])), "rel_bias": float(rb.mean()),
             "rel_bias_se": se(rb), "paired_delta": float(dd.mean()) if len(dd) else float("nan"),
             "paired_delta_se": se(dd), "coverage": float(np.mean([r["covered"] for r in rs])),
             "mu_rel_bias": float(np.mean([(r["mu_median"] - r["true_mu"]) / r["true_mu"] for r in rs])),
             "min_ess": float(min(r["ess"] for r in rs))}
        summary.append(s)
        print(f"{cell:16s} {s['n']:3d} {s['borrowed']:8.3f} {s['F']:5.2f} {s['n_sets']:5.0f} {s['root_median']:11.0f} "
              f"{s['rel_bias']:7.3f} {s['rel_bias_se']:5.3f} {s['paired_delta']:12.3f} {s['paired_delta_se']:5.3f} "
              f"{s['coverage']:5.2f} {s['mu_rel_bias']:7.2f} {s['min_ess']:6.0f}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    col = {"true": "#4a7bb7", "lexstat": "#d1495b"}
    lab = {"true_kappa_0": "true cognates\nno borrowing", "lexstat_kappa_0": "LexStat cognates\nno borrowing",
           "true_kappa_1": "true cognates\nborrowing κ=1", "lexstat_kappa_1": "LexStat cognates\nborrowing κ=1"}
    for k, s in enumerate(summary):
        ax.errorbar(k, s["rel_bias"] * 100, yerr=2 * s["rel_bias_se"] * 100, fmt="o", capsize=4,
                    color=col[s["cell"].split("_")[0]])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(summary)), [lab[s["cell"]] for s in summary], fontsize=8)
    ax.set_ylabel("root age relative bias (%), mean ± 2 se")
    ax.set_title("Realistic words on the Mayan tree (29 languages × 40 concepts)")
    fig.tight_layout()
    fig.savefig(FIG / "m6r_realistic.png", dpi=150)
    print(f"figure -> {FIG / 'm6r_realistic.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--reps", type=int, default=10)
    p.add_argument("--jobs", type=int, default=6)
    p = sub.add_parser("run"); p.add_argument("--jobs", type=int, default=7)
    sub.add_parser("summarise")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
