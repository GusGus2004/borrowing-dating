"""Milestone 1: well-calibrated simulation study for the no-borrowing SDollo model.

Each replicate draws root age, tree, loss rate and clock sd from the prior in
``infer/beast/m1_prior.json``, simulates cognate data, and runs BEAST2 with
exactly that prior. If simulator and inference agree, the 95% HPD contains the
true value in ~95% of replicates for every parameter.

Subcommands:
    prepare     --n 100 [--seed0 1000]   simulate replicates + XML
    run         [--jobs 6]               run BEAST2 on every unfinished replicate
    prior-check [--draws 20000]          BEAST2 -sampleFromPrior vs simulator prior
    summarise                            coverage table + figures
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.mcmc import ess, hpd, read_log  # noqa: E402
from infer.beastrun import BEAST, run_all  # noqa: E402
from infer.beastxml import draw_truth, is_relaxed, load_prior, render_xml, simulate_replicate  # noqa: E402
from pipeline.cldf import read_cognate_matrix  # noqa: E402

# set in main() from --clock
PRIOR: Path
OUT: Path
FIG = ROOT / "figures"

# log column -> truth key
PARAMS = {
    "Tree.height": ("root_age", "Root age (years)"),
    "Tree.treeLength": ("tree_length", "Tree length (years)"),
    "deathRate": ("loss_rate", "Loss rate μ (per year)"),
    "ucldStdev": ("clock_sd", "Relaxed-clock σ"),
}


def params_for(prior: dict) -> dict:
    return {k: v for k, v in PARAMS.items() if k != "ucldStdev" or is_relaxed(prior)}


def rep_dirs() -> list[Path]:
    return sorted(p for p in OUT.glob("rep_*") if p.is_dir())


def truth_values(rep: Path) -> dict:
    t = json.loads((rep / "truth.json").read_text())
    return {
        "root_age": t["root_age"],
        "tree_length": t["tree_length"],
        "loss_rate": t["params"]["loss_rate"],
        "clock_sd": t["params"]["clock_sd"],
    }


# ---------------------------------------------------------------------- prepare


def cmd_prepare(a) -> None:
    prior = load_prior(PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        rep = OUT / f"rep_{i:03d}"
        if (rep / "beast.xml").exists():
            continue
        rec = simulate_replicate(prior, a.seed0 + i, rep)
        print(f"{rep.name}: seed {a.seed0 + i}, root {rec['root_age']:.0f}, "
              f"{rec['n_cognatesets']} cognate sets")


# ---------------------------------------------------------------------- run


def cmd_run(a) -> None:
    run_all(rep_dirs(), lambda r: 7919 + int(r.name.split("_")[1]), a.jobs)


# ---------------------------------------------------------------------- prior check


def cmd_prior_check(a) -> None:
    """Run BEAST2 with -sampleFromPrior and compare marginals to the simulator."""
    from scipy import stats

    prior = load_prior(PRIOR)
    d = OUT / "prior_check"
    d.mkdir(parents=True, exist_ok=True)
    # any replicate's data will do: the likelihood is ignored
    rep = OUT / "rep_000"
    if not rep.exists():
        simulate_replicate(prior, 1000, rep)
    cm = read_cognate_matrix(rep / "Wordlist-metadata.json", concept_level_missing=False)
    mcmc = dict(prior["mcmc"], chain_length=10_000_000, log_every=500)
    (d / "beast.xml").write_text(render_xml(cm, prior, stem="beast", mcmc=mcmc))
    if not (d / "beast.log").exists() or a.rerun:
        with open(d / "beast.out", "w") as out:
            subprocess.run(BEAST + ["-overwrite", "-seed", "1", "-sampleFromPrior", "beast.xml"],
                           cwd=d, stdout=out, stderr=subprocess.STDOUT, check=True)
    log = read_log(d / "beast.log", burnin_frac=0.1)

    rng = np.random.default_rng(a.seed)
    params = params_for(prior)
    sims = {k: [] for k in ("root_age", "tree_length", "loss_rate", "clock_sd")}
    for _ in range(a.draws):
        t = draw_truth(prior, rng)
        sims["root_age"].append(t["root_age"])
        sims["tree_length"].append(t["tree"].tree_length())
        sims["loss_rate"].append(t["loss_rate"])
        sims["clock_sd"].append(t["clock_sd"])

    rows = []
    for col, (key, label) in params.items():
        b, s = log[col], np.array(sims[key])
        # thin BEAST samples to roughly independent draws for the KS test
        n_eff = max(int(ess(b)), 10)
        b_thin = b[:: max(len(b) // n_eff, 1)]
        ks = stats.ks_2samp(b_thin, s)
        rows.append({"parameter": col, "beast_mean": float(b.mean()), "sim_mean": float(s.mean()),
                     "beast_sd": float(b.std()), "sim_sd": float(s.std()),
                     "beast_ess": float(ess(b)), "ks_p": float(ks.pvalue)})
    (d / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"{'parameter':18s} {'BEAST mean':>12s} {'sim mean':>12s} {'BEAST sd':>11s} "
          f"{'sim sd':>11s} {'ESS':>7s} {'KS p':>7s}")
    for r in rows:
        print(f"{r['parameter']:18s} {r['beast_mean']:12.5g} {r['sim_mean']:12.5g} "
              f"{r['beast_sd']:11.4g} {r['sim_sd']:11.4g} {r['beast_ess']:7.0f} {r['ks_p']:7.3f}")


# ---------------------------------------------------------------------- summarise


def cmd_summarise(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prior = load_prior(PRIOR)
    PARAMS = params_for(prior)
    rows = []
    for rep in rep_dirs():
        if not (rep / "done").exists():
            continue
        log = read_log(rep / "beast.log", burnin_frac=prior["mcmc"]["burnin_frac"])
        tv = truth_values(rep)
        row = {"rep": rep.name}
        for col, (key, _) in PARAMS.items():
            x = log[col]
            lo, hi = hpd(x)
            row[col] = {"true": tv[key], "mean": float(x.mean()), "median": float(np.median(x)),
                        "hpd_lo": lo, "hpd_hi": hi, "covered": bool(lo <= tv[key] <= hi),
                        "ess": ess(x), "rank": float((x < tv[key]).mean())}
        rows.append(row)
    if not rows:
        sys.exit("no finished replicates")
    n = len(rows)
    (OUT / "coverage.json").write_text(json.dumps(rows, indent=2))

    summary = {"n_replicates": n, "parameters": {}}
    print(f"{n} finished replicates\n")
    print(f"{'parameter':18s} {'coverage':>9s} {'±2se':>6s} {'min ESS':>8s} {'#ESS<200':>9s} "
          f"{'rel.bias':>9s} {'HPD/prior':>10s}")
    prior_sd = {"Tree.height": prior["root_age"]["sd"]}
    for col in PARAMS:
        cov = np.mean([r[col]["covered"] for r in rows])
        se = np.sqrt(cov * (1 - cov) / n) if 0 < cov < 1 else np.sqrt(0.95 * 0.05 / n)
        esss = np.array([r[col]["ess"] for r in rows])
        relbias = np.mean([(r[col]["median"] - r[col]["true"]) / r[col]["true"] for r in rows])
        width = np.mean([r[col]["hpd_hi"] - r[col]["hpd_lo"] for r in rows])
        shrink = width / (2 * 1.96 * prior_sd[col]) if col in prior_sd else float("nan")
        summary["parameters"][col] = {"coverage": cov, "coverage_se": se, "min_ess": float(esss.min()),
                                      "n_ess_below_200": int((esss < 200).sum()),
                                      "mean_relative_bias_of_median": relbias,
                                      "mean_hpd_width": width,
                                      "hpd_width_over_prior_95_width": shrink}
        print(f"{col:18s} {cov:9.3f} {2 * se:6.3f} {esss.min():8.0f} {(esss < 200).sum():9d} "
              f"{relbias:9.3f} {shrink:10.3f}")
    (OUT / "coverage_summary.json").write_text(json.dumps(summary, indent=2))

    # Figure: truth vs posterior median with HPD bars, plus rank histograms
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, len(PARAMS), figsize=(4 * len(PARAMS), 7.5))
    for j, (col, (_, label)) in enumerate(PARAMS.items()):
        ax = axes[0, j]
        tr = np.array([r[col]["true"] for r in rows])
        md = np.array([r[col]["median"] for r in rows])
        lo = np.array([r[col]["hpd_lo"] for r in rows])
        hi = np.array([r[col]["hpd_hi"] for r in rows])
        cov = np.array([r[col]["covered"] for r in rows])
        for mask, color in ((cov, "#4a7bb7"), (~cov, "#d1495b")):
            ax.errorbar(tr[mask], md[mask], yerr=[md[mask] - lo[mask], hi[mask] - md[mask]],
                        fmt="o", ms=3, lw=0.7, color=color, alpha=0.8)
        lim = [min(tr.min(), lo.min()), max(tr.max(), hi.max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel("posterior median, 95% HPD")
        ax.set_title(f"{col}: coverage {cov.mean():.2f}")
        ax = axes[1, j]
        ax.hist([r[col]["rank"] for r in rows], bins=10, range=(0, 1), color="#888", edgecolor="w")
        ax.axhline(n / 10, color="k", ls="--", lw=0.8)
        ax.set_xlabel("posterior quantile of truth (should be uniform)")
    clock = "UCLN" if is_relaxed(prior) else "strict clock"
    fig.suptitle(f"Milestone 1: SDollo + {clock} + Yule, BEAST2, {n} replicates drawn from the prior")
    fig.tight_layout()
    name = f"m1_coverage_{OUT.name.removeprefix('m1_')}.png"
    fig.savefig(FIG / name, dpi=150)
    print(f"\nfigure -> {FIG / name}")


def main() -> None:
    global PRIOR, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock", choices=["strict", "relaxed"], default="strict")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--n", type=int, default=100); p.add_argument("--seed0", type=int, default=1000)
    p = sub.add_parser("run"); p.add_argument("--jobs", type=int, default=6)
    p = sub.add_parser("prior-check"); p.add_argument("--draws", type=int, default=20000)
    p.add_argument("--seed", type=int, default=2); p.add_argument("--rerun", action="store_true")
    sub.add_parser("summarise")
    a = ap.parse_args()
    PRIOR = ROOT / "infer" / "beast" / f"m1_{a.clock}_prior.json"
    OUT = ROOT / "data" / "derived" / f"m1_{a.clock}"
    {"prepare": cmd_prepare, "run": cmd_run, "prior-check": cmd_prior_check,
     "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
