"""Milestone 6s: does the shape of cognate-set splits decide the direction of the dating bias?

V2 and V3 of milestone 6r share the same true histories, but LexStat's errors make the
root older in V2 (+17%, sets cut into subgroup-sized pieces) and younger in V3 (-12%,
single languages peeled off). Here the true cognate sets of those same replicates are
split on purpose, adding exactly as many sets as LexStat did in V3 (per replicate):

  tip:   repeatedly pick a form whose set has 2+ languages and make it a singleton set;
  clade: repeatedly pick a set with 3+ languages and a random internal node of the true
         tree under which 2+ (but not all) of its languages lie; move those to a new set.

No borrowing. Inference, priors, calibrations and seeds as in m6r. The baseline (true
cognates) is m6r2/true_kappa_0, which has the same matrices and seeds.

Subcommands: prepare | run [--jobs N] | summarise
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from analysis.mcmc import ess, hpd, read_log  # noqa: E402
from infer.beastrun import run_all  # noqa: E402
from infer.beastxml import render_xml  # noqa: E402
from m5c_asjp_expert import matrix  # noqa: E402
from m6r_realistic import BASE, CALS, CHAIN, _leaves, mrca_age  # noqa: E402
from sim_realism import MAYAN_V2, simulate_forms  # noqa: E402

OUT = ROOT / "data/derived/m6s"
V3 = ROOT / "data/derived/m6r3/lexstat_kappa_0"
BASELINE = ROOT / "data/derived/m6r2/true_kappa_0"
SHAPES = ("tip", "clade")
REPS = 10


def split(gold: dict, lang: dict, tree, n_extra: int, shape: str, rng) -> dict:
    cog = dict(gold)
    clades = [{l.name for l in _leaves(n)} for n in tree.internal_nodes() if n is not tree.root]
    added, tries = 0, 0
    while added < n_extra:
        tries += 1
        if tries > 100 * n_extra:
            raise RuntimeError(f"could only add {added} of {n_extra} sets")
        members = collections.defaultdict(list)
        for i, s in cog.items():
            members[s].append(i)
        langs = {s: {lang[i] for i in ids} for s, ids in members.items()}
        if shape == "tip":
            pool = [i for i, s in cog.items() if len(langs[s]) >= 2]
            i = pool[rng.integers(len(pool))]
            cog[i] = f"tip{added}"
        else:
            pool = [s for s in members if len(langs[s]) >= 3]
            s = pool[rng.integers(len(pool))]
            ok = [c for c in clades if 2 <= len(c & langs[s]) < len(langs[s])]
            if not ok:
                continue
            c = ok[rng.integers(len(ok))]
            for i in members[s]:
                if lang[i] in c:
                    cog[i] = f"clade{added}"
        added += 1
    return cog


def prepare_one(i: int) -> str:
    forms, gold, tree, mu, _ = simulate_forms({**MAYAN_V2, "kappa": 0.0}, 7700 + 10 * i)
    v3 = json.loads((V3 / f"rep_{i:03d}" / "truth.json").read_text())["lexstat"]
    n_extra = v3["sets_lexstat"] - v3["sets_true"]
    languages = sorted(l.name for l in tree.leaves())
    concept = {f["ID"]: f["concept"] for f in forms}
    lang = {f["ID"]: f["Language_ID"] for f in forms}
    prior = json.loads((BASELINE / f"rep_{i:03d}" / "prior.json").read_text())
    msg = [f"rep {i}: {len(set(gold.values()))} true sets, +{n_extra}"]
    for k, shape in enumerate(SHAPES):
        cog = split(gold, lang, tree, n_extra, shape, np.random.default_rng(5500 + 10 * i + k))
        cm = matrix(languages, forms, concept, cog)
        d = OUT / shape / f"rep_{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "beast.xml").write_text(render_xml(cm, prior, stem="beast"))
        (d / "prior.json").write_text(json.dumps(prior, indent=2))
        (d / "truth.json").write_text(json.dumps({"root_age": tree.root_age, "loss_rate": mu, "shape": shape,
                                                  "n_extra": n_extra, "n_sets": len(cm.cognatesets)}, indent=2))
        msg.append(f"{shape} -> {len(cm.cognatesets)} sets")
    return "; ".join(msg)


def cmd_prepare(a) -> None:
    with ProcessPoolExecutor(6) as pool:
        for m in pool.map(prepare_one, range(REPS)):
            print(m, flush=True)


def cmd_run(a) -> None:
    reps = sorted((r for s in SHAPES for r in (OUT / s).glob("rep_*")), key=lambda r: (r.name, r.parent.name))
    run_all(reps, lambda r: 161803 + int(r.name.split("_")[1]), a.jobs)


def fit(rep: Path) -> dict:
    t = json.loads((rep / "truth.json").read_text())
    log = read_log(rep / "beast.log", burnin_frac=BASE["mcmc"]["burnin_frac"])
    x = log["Tree.height"]
    lo, hi = hpd(x)
    return {"median": float(np.median(x)), "covered": bool(lo <= t["root_age"] <= hi), "ess": ess(x),
            "mu_median": float(np.median(log["deathRate"])), "true_root": t["root_age"], "true_mu": t["loss_rate"],
            "n_sets": t.get("n_sets")}


def cmd_summarise(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = {"true (baseline)": BASELINE, **{s: OUT / s for s in SHAPES}, "LexStat V3": V3}
    res = {c: {r.name: fit(r) for r in sorted(p.glob("rep_*")) if (r / "done").exists()} for c, p in cells.items()}
    base = res["true (baseline)"]
    se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v)))
    print(f"{'cell':16s} {'n':>3s} {'sets':>5s} {'bias':>7s} {'±se':>5s} {'Δ paired':>8s} {'±se':>5s} {'cover':>5s} "
          f"{'μ bias':>7s} {'minESS':>6s}")
    summary = []
    for c, rs in res.items():
        v = list(rs.values())
        rb = np.array([(r["median"] - r["true_root"]) / r["true_root"] for r in v])
        dd = np.array([(rs[k]["median"] - base[k]["median"]) / rs[k]["true_root"] for k in rs if k in base])
        s = {"cell": c, "n": len(v), "n_sets": float(np.mean([r["n_sets"] for r in v])), "rel_bias": float(rb.mean()),
             "rel_bias_se": se(rb), "paired_delta": float(dd.mean()), "paired_delta_se": se(dd),
             "coverage": float(np.mean([r["covered"] for r in v])),
             "mu_rel_bias": float(np.mean([(r["mu_median"] - r["true_mu"]) / r["true_mu"] for r in v])),
             "min_ess": float(min(r["ess"] for r in v))}
        summary.append(s)
        print(f"{c:16s} {s['n']:3d} {s['n_sets']:5.0f} {s['rel_bias']:7.3f} {s['rel_bias_se']:5.3f} "
              f"{s['paired_delta']:8.3f} {s['paired_delta_se']:5.3f} {s['coverage']:5.2f} {s['mu_rel_bias']:7.2f} "
              f"{s['min_ess']:6.0f}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    col = ["#4a7bb7", "#d1495b", "#edae49", "#7a7a7a"]
    for k, s in enumerate(summary):
        ax.errorbar(k, s["rel_bias"] * 100, yerr=2 * s["rel_bias_se"] * 100, fmt="o", capsize=4, color=col[k])
    ax.axhline(0, color="k", lw=0.8)
    lab = {"true (baseline)": "true cognates", "tip": "split off\nsingle languages", "clade": "split off\nsubgroups",
           "LexStat V3": "LexStat\n(V3 words)"}
    ax.set_xticks(range(len(summary)), [lab[s["cell"]] for s in summary], fontsize=8)
    ax.set_ylabel("root age relative bias (%), mean ± 2 se")
    ax.set_title("Same histories, same number of extra sets, different split shape")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/m6s_split_shape.png", dpi=150)
    print("figure -> figures/m6s_split_shape.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("run"); p.add_argument("--jobs", type=int, default=7)
    sub.add_parser("summarise")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
