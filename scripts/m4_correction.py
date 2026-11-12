"""Milestone 4: does masking detected loans reduce the root-age bias?

Reuses the milestone 2 replicates of the cells in --cells (same seeds, so the
cognate data are identical to the uncorrected fits in data/derived/m2), adds
phoneme strings, and refits BEAST2 after masking

* ``oracle``:   the true borrowed entries (upper bound for any detector);
* ``detector``: entries whose level-1 pair-HMM score exceeds --threshold
  (robust z, see hmm/pairhmm.py).

Subcommands: prepare | run [--jobs N] | summarise
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
from hmm.pairhmm import entry_scores  # noqa: E402
from infer.beastrun import run_all  # noqa: E402
from infer.beastxml import load_prior, render_xml, simulate_replicate  # noqa: E402
from infer.correction import mask_entries, read_borrowings, read_wordlist  # noqa: E402
from pipeline.cldf import read_cognate_matrix  # noqa: E402

DESIGN = json.loads((ROOT / "infer/beast/m2_design.json").read_text())
PRIOR = load_prior(ROOT / "infer/beast" / DESIGN["prior"])
M2 = ROOT / "data/derived/m2"
OUT = ROOT / "data/derived/m4"
FIG = ROOT / "figures"
PHON = {"change_rate": 1 / 100}
METHODS = ("oracle", "detector")


def cell_names(a) -> list[str]:
    return a.cells


def cmd_prepare(a) -> None:
    for cell in a.cells:
        contact, kappa = cell.split("_kappa_")
        b = dict(DESIGN["contact"][contact], rate_over_loss=float(kappa), mode=DESIGN["mode"])
        stats = []
        for i in range(DESIGN["n_reps"]):
            base = OUT / cell / f"rep_{i:03d}"
            data = base / "data"
            if not (data / "truth.json").exists():
                simulate_replicate(PRIOR, DESIGN["seed0"] + i, data, borrowing=b, phonology=PHON)
            cm = read_cognate_matrix(data / "Wordlist-metadata.json", concept_level_missing=False)
            m2cm = read_cognate_matrix(M2 / cell / f"rep_{i:03d}" / "Wordlist-metadata.json",
                                       concept_level_missing=False)
            assert np.array_equal(cm.matrix, m2cm.matrix), "phonology changed the cognate data"
            truth = set(read_borrowings(data))
            wl, _ = read_wordlist(data)
            sc = entry_scores(wl)
            flagged = {e for e, v in sc.items() if np.isfinite(v) and v > a.threshold}
            rec = {"n_entries": len(sc), "n_borrowed": len(truth), "n_flagged": len(flagged),
                   "precision": len(flagged & truth) / max(len(flagged), 1),
                   "recall": len(flagged & truth) / max(len(truth), 1)}
            for method, ent in (("oracle", truth), ("detector", flagged)):
                d = base / method
                d.mkdir(parents=True, exist_ok=True)
                mcm, st = mask_entries(cm, ent)
                rec[method] = st
                if not (d / "beast.xml").exists():
                    (d / "beast.xml").write_text(render_xml(mcm, PRIOR, stem="beast"))
            (base / "correction.json").write_text(json.dumps(rec, indent=2))
            stats.append(rec)
        print(f"{cell}: borrowed {np.mean([s['n_borrowed'] / s['n_entries'] for s in stats]):.3f} of entries; "
              f"detector precision {np.mean([s['precision'] for s in stats]):.2f}, "
              f"recall {np.mean([s['recall'] for s in stats]):.2f}; masked language-concepts: "
              f"oracle {np.mean([s['oracle']['masked_fraction'] for s in stats]):.3f}, "
              f"detector {np.mean([s['detector']['masked_fraction'] for s in stats]):.3f}", flush=True)


def cmd_run(a) -> None:
    reps = [OUT / c / f"rep_{i:03d}" / m for i in range(DESIGN["n_reps"]) for c in a.cells for m in METHODS]
    run_all(reps, lambda r: 104729 + int(r.parent.name.split("_")[1]), a.jobs)


def _fit(d: Path, true_root: float) -> dict | None:
    if not (d / "done").exists():
        return None
    x = read_log(d / "beast.log", burnin_frac=PRIOR["mcmc"]["burnin_frac"])["Tree.height"]
    lo, hi = hpd(x)
    return {"median": float(np.median(x)), "hpd_lo": lo, "hpd_hi": hi, "width": hi - lo,
            "covered": bool(lo <= true_root <= hi), "ess": ess(x)}


def cmd_summarise(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for cell in a.cells:
        for i in range(DESIGN["n_reps"]):
            base = OUT / cell / f"rep_{i:03d}"
            t = json.loads((base / "data/truth.json").read_text())
            T = t["root_age"]
            fits = {"none": _fit(M2 / cell / f"rep_{i:03d}", T), "nobor": _fit(M2 / "kappa_0" / f"rep_{i:03d}", T)}
            fits.update({m: _fit(base / m, T) for m in METHODS})
            if any(v is None for v in fits.values()):
                continue
            rows.append({"cell": cell, "rep": i, "true_root": T, "fits": fits,
                         "correction": json.loads((base / "correction.json").read_text())})
    if not rows:
        sys.exit("no complete replicates")
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    labels = {"nobor": "no borrowing (reference)", "none": "uncorrected", "oracle": "oracle mask",
              "detector": "pair-HMM mask"}
    summary = []
    print(f"{'cell':22s} {'fit':26s} {'n':>3s} {'rel.bias':>9s} {'±se':>6s} {'coverage':>8s} {'HPD width':>9s}")
    for cell in a.cells:
        rs = [r for r in rows if r["cell"] == cell]
        for m in ("nobor", "none", "oracle", "detector"):
            rb = np.array([(r["fits"][m]["median"] - r["true_root"]) / r["true_root"] for r in rs])
            s = {"cell": cell, "fit": m, "n": len(rs), "rel_bias": float(rb.mean()),
                 "rel_bias_se": float(rb.std(ddof=1) / np.sqrt(len(rb))) if len(rb) > 1 else float("nan"),
                 "coverage": float(np.mean([r["fits"][m]["covered"] for r in rs])),
                 "mean_hpd_width": float(np.mean([r["fits"][m]["width"] for r in rs]))}
            summary.append(s)
            print(f"{cell:22s} {labels[m]:26s} {s['n']:3d} {s['rel_bias']:9.3f} {s['rel_bias_se']:6.3f} "
                  f"{s['coverage']:8.2f} {s['mean_hpd_width']:9.0f}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, len(a.cells), figsize=(5.5 * len(a.cells), 4.2), squeeze=False)
    for ax, cell in zip(axes[0], a.cells):
        ss = [s for s in summary if s["cell"] == cell]
        x = np.arange(len(ss))
        ax.bar(x, [s["rel_bias"] * 100 for s in ss], yerr=[2 * s["rel_bias_se"] * 100 for s in ss],
               color=["#999", "#d1495b", "#4a7bb7", "#66a182"], capsize=4)
        for xi, s in zip(x, ss):
            ax.text(xi, 0, f"cov {s['coverage']:.2f}", ha="center", va="bottom" if s["rel_bias"] < 0 else "top",
                    fontsize=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [labels[s["fit"]] for s in ss], rotation=15, fontsize=8)
        ax.set_ylabel("root-age bias (% of truth, ±2 s.e.)")
        ax.set_title(cell.replace("_kappa_", ", κ = "))
    fig.suptitle("Milestone 4: masking detected loans before dating")
    fig.tight_layout()
    fig.savefig(FIG / "m4_correction.png", dpi=150)
    print(f"figure -> {FIG / 'm4_correction.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["uniform_kappa_1", "distance_kappa_1"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--threshold", type=float, default=1.0)
    p = sub.add_parser("run"); p.add_argument("--jobs", type=int, default=7)
    sub.add_parser("summarise")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
