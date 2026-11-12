"""Milestone 2: bias in BEAST2 root-age estimates caused by unmodelled borrowing.

Design in ``infer/beast/m2_design.json``. Cells are (contact model, kappa),
where the borrowing rate is kappa * mu; kappa = 0 is a single shared cell.

Subcommands:
    prepare      simulate every replicate in every cell + XML
    run          [--jobs 6] run BEAST2 on unfinished replicates
    summarise    bias table (with Monte Carlo SE) + figure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.mcmc import ess, hpd, read_log  # noqa: E402
from infer.beastrun import run_all  # noqa: E402
from infer.beastxml import load_prior, simulate_replicate  # noqa: E402

DESIGN = json.loads((ROOT / "infer/beast/m2_design.json").read_text())
PRIOR = load_prior(ROOT / "infer/beast" / DESIGN["prior"])
OUT = ROOT / "data/derived/m2"
FIG = ROOT / "figures"


def cells() -> list[tuple[str, float]]:
    out = [("none", 0.0)]
    for c in DESIGN["contact"]:
        out += [(c, k) for k in DESIGN["kappa"] if k > 0]
    return out


def cell_dir(contact: str, kappa: float) -> Path:
    return OUT / ("kappa_0" if kappa == 0 else f"{contact}_kappa_{kappa:g}")


def cmd_prepare(a) -> None:
    for contact, kappa in cells():
        d = cell_dir(contact, kappa)
        b = None if kappa == 0 else dict(DESIGN["contact"][contact], rate_over_loss=kappa,
                                          mode=DESIGN["mode"])
        fb = []
        for i in range(DESIGN["n_reps"]):
            rep = d / f"rep_{i:03d}"
            if not (rep / "beast.xml").exists():
                if b is None:
                    # same simulator as the borrowing cells (Gillespie, beta = 0)
                    simulate_replicate(PRIOR, DESIGN["seed0"] + i, rep, borrowing={"rate_over_loss": 0.0})
                else:
                    simulate_replicate(PRIOR, DESIGN["seed0"] + i, rep, borrowing=b)
            fb.append(json.loads((rep / "truth.json").read_text()).get("borrowed_fraction", 0.0))
        print(f"{d.name:24s} {len(fb)} reps, mean borrowed fraction {np.mean(fb):.3f}", flush=True)


def cmd_run(a) -> None:
    reps = [r for c, k in cells() for r in sorted(cell_dir(c, k).glob("rep_*"))]
    # interleave cells so partial results cover the whole grid
    reps.sort(key=lambda r: (r.name, r.parent.name))
    run_all(reps, lambda r: 104729 + int(r.name.split("_")[1]), a.jobs)


def cmd_summarise(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    burn = PRIOR["mcmc"]["burnin_frac"]
    rows = []
    for contact, kappa in cells():
        for rep in sorted(cell_dir(contact, kappa).glob("rep_*")):
            if not (rep / "done").exists():
                continue
            t = json.loads((rep / "truth.json").read_text())
            log = read_log(rep / "beast.log", burnin_frac=burn)
            x = log["Tree.height"]
            lo, hi = hpd(x)
            rows.append({"contact": contact, "kappa": kappa, "rep": rep.name,
                         "borrowed_fraction": t.get("borrowed_fraction", 0.0),
                         "true_root": t["root_age"], "median": float(np.median(x)),
                         "hpd_lo": lo, "hpd_hi": hi, "covered": bool(lo <= t["root_age"] <= hi),
                         "ess": ess(x), "true_mu": t["params"]["loss_rate"],
                         "mu_median": float(np.median(log["deathRate"])),
                         "true_tl": t["tree_length"], "tl_median": float(np.median(log["Tree.treeLength"]))})
    if not rows:
        sys.exit("no finished replicates")
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))

    base = {r["rep"]: r for r in rows if r["kappa"] == 0}
    summary = []
    print(f"{'contact':9s} {'kappa':>5s} {'n':>3s} {'borrowed':>8s} {'rel.bias':>9s} {'±se':>6s} "
          f"{'paired Δ':>9s} {'±se':>6s} {'coverage':>8s} {'μ rel.bias':>10s} {'TL rel.bias':>11s} {'minESS':>6s}")
    for contact, kappa in cells():
        rs = [r for r in rows if r["contact"] == contact and r["kappa"] == kappa]
        if not rs:
            continue
        rb = np.array([(r["median"] - r["true_root"]) / r["true_root"] for r in rs])
        # paired difference against the kappa = 0 fit of the same replicate (common random numbers)
        dd = np.array([(r["median"] - base[r["rep"]]["median"]) / r["true_root"]
                       for r in rs if r["rep"] in base])
        mb = np.array([(r["mu_median"] - r["true_mu"]) / r["true_mu"] for r in rs])
        tb = np.array([(r["tl_median"] - r["true_tl"]) / r["true_tl"] for r in rs])
        s = {"contact": contact, "kappa": kappa, "n": len(rs),
             "borrowed_fraction": float(np.mean([r["borrowed_fraction"] for r in rs])),
             "rel_bias": float(rb.mean()), "rel_bias_se": float(rb.std(ddof=1) / np.sqrt(len(rb))) if len(rb) > 1 else float("nan"),
             "paired_delta": float(dd.mean()) if len(dd) else float("nan"),
             "paired_delta_se": float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else float("nan"),
             "coverage": float(np.mean([r["covered"] for r in rs])),
             "mu_rel_bias": float(mb.mean()), "mu_rel_bias_se": float(mb.std(ddof=1) / np.sqrt(len(mb))) if len(mb) > 1 else float("nan"),
             "tl_rel_bias": float(tb.mean()), "tl_rel_bias_se": float(tb.std(ddof=1) / np.sqrt(len(tb))) if len(tb) > 1 else float("nan"),
             "min_ess": float(min(r["ess"] for r in rs))}
        summary.append(s)
        print(f"{contact:9s} {kappa:5.2f} {s['n']:3d} {s['borrowed_fraction']:8.3f} {s['rel_bias']:9.3f} "
              f"{s['rel_bias_se']:6.3f} {s['paired_delta']:9.3f} {s['paired_delta_se']:6.3f} "
              f"{s['coverage']:8.2f} {s['mu_rel_bias']:10.3f} {s['tl_rel_bias']:11.3f} {s['min_ess']:6.0f}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    colors = {"uniform": "#4a7bb7", "distance": "#d1495b"}
    s0 = [s for s in summary if s["kappa"] == 0]
    for contact in DESIGN["contact"]:
        ss = s0 + [s for s in summary if s["contact"] == contact]
        k = np.array([s["kappa"] for s in ss])
        fb = np.array([s["borrowed_fraction"] for s in ss])
        y = np.array([s["rel_bias"] for s in ss]); e = np.array([s["rel_bias_se"] for s in ss])
        lab = f"{contact} contact" + (f" (scale {DESIGN['contact'][contact].get('distance_scale', 0):g} y)"
                                     if contact == "distance" else "")
        axes[0].errorbar(fb * 100, y * 100, yerr=2 * e * 100, fmt="o-", color=colors[contact],
                         capsize=3, label=lab)
        axes[1].plot(fb * 100, [s["coverage"] for s in ss], "o-", color=colors[contact], label=lab)
        for key, ls, nm in (("mu_rel_bias", "-", "loss rate μ"), ("tl_rel_bias", "--", "tree length")):
            axes[2].errorbar(fb * 100, [s[key] * 100 for s in ss], yerr=[2 * s[key + "_se"] * 100 for s in ss],
                             fmt="o" + ls, color=colors[contact], capsize=3, label=f"{nm}, {contact}")
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("borrowed entries in leaf wordlists (%)")
    axes[0].set_ylabel("root-age bias of posterior median (% of truth)")
    axes[0].set_title("Root-age bias vs borrowing (±2 MC s.e.)")
    axes[0].legend()
    axes[1].axhline(0.95, color="k", lw=0.8, ls="--")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("borrowed entries in leaf wordlists (%)")
    axes[1].set_ylabel("95% HPD coverage of true root age")
    axes[1].set_title("Coverage")
    axes[2].axhline(0, color="k", lw=0.8, ls="--")
    axes[2].set_xlabel("borrowed entries in leaf wordlists (%)")
    axes[2].set_ylabel("bias of posterior median (% of truth)")
    axes[2].set_title("Loss rate and tree length")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Milestone 2: SDollo + strict clock fitted to data with borrowing "
                 f"({DESIGN['n_reps']} replicates per cell)")
    fig.tight_layout()
    fig.savefig(FIG / "m2_bias_curve.png", dpi=150)
    print(f"\nfigure -> {FIG / 'm2_bias_curve.png'}")


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
