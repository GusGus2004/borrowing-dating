"""Milestone 5: dating Proto-Mayan on ASJP core vocabulary (no borrowing correction yet).

Variants (all with infer/beast/m5_mayan_prior.json unless noted):
    strict            strict clock, main calibrations
    relaxed           UCLN relaxed clock (needs babelfix), main calibrations
    strict_cholan     + contested Classic Ch'olti'an calibration (sensitivity)
    strict_t065       cognates at LexStat threshold 0.65 (cognate-coding sensitivity)
    strict_t085       cognates at LexStat threshold 0.85
    strict_cholantzeltalan  + candidate Cholan-Tzeltalan minimum age (>= 1700 BP)
    strict_sca055     cognates by SCA distance at 0.55 (LexStat splits identical forms on 40-item lists)

Subcommands: prepare | run [--jobs N] [--variants ...] | summarise
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
from pipeline.asjp import build  # noqa: E402
from pipeline.cldf import read_cognate_matrix  # noqa: E402

PRIOR = load_prior(ROOT / "infer/beast/m5_mayan_prior.json")
OUT = ROOT / "data/derived/m5"
FIG = ROOT / "figures"
ASJP = ROOT / "data/raw/asjp/cldf"
KAUFMAN_BP = 4200.0  # c. 2200 BC, glottochronological (comparison only)


def dataset(threshold: float, method: str = "lexstat") -> Path:
    if method == "lexstat":
        d = ROOT / ("data/derived/asjp_mayan" if threshold == 0.75 else f"data/derived/asjp_mayan_t{int(threshold * 100):03d}")
    else:
        d = ROOT / f"data/derived/asjp_mayan_{method}{int(threshold * 100):03d}"
    if not (d / "Wordlist-metadata.json").exists():
        build(ASJP, d, threshold=threshold, method=method)
    return d / "Wordlist-metadata.json"


def variants() -> dict[str, tuple[dict, float]]:
    relaxed = copy.deepcopy(PRIOR)
    relaxed["clock_sd"] = {"dist": "exponential", "mean": 0.3}
    cholan = copy.deepcopy(PRIOR)
    cholan["calibrations"] = PRIOR["calibrations"] + PRIOR["sensitivity_calibrations"]
    ct = copy.deepcopy(PRIOR)
    ct["calibrations"] = PRIOR["calibrations"] + PRIOR["candidate_calibrations"]
    return {"strict": (PRIOR, 0.75), "relaxed": (relaxed, 0.75), "strict_cholan": (cholan, 0.75),
            "strict_t065": (PRIOR, 0.65), "strict_t085": (PRIOR, 0.85), "strict_sca055": (PRIOR, ("sca", 0.55)),
            "strict_cholantzeltalan": (ct, 0.75)}


def cmd_prepare(a) -> None:
    for name, (prior, thr) in variants().items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        method, t = thr if isinstance(thr, tuple) else ("lexstat", thr)
        if (d / "beast.xml").exists() and (d / "done").exists():
            continue
        cm = read_cognate_matrix(dataset(t, method), concept_level_missing=True)
        (d / "beast.xml").write_text(render_xml(cm, prior, stem="beast"))
        (d / "prior.json").write_text(json.dumps(prior, indent=2))
        print(f"{name}: {cm.shape[0]} languages x {cm.shape[1]} cognate sets (threshold {thr})")


def cmd_run(a) -> None:
    names = a.variants or list(variants())
    run_all([OUT / n for n in names], lambda r: 31337, a.jobs)


def cmd_summarise(a) -> None:
    rows = {}
    for name in variants():
        d = OUT / name
        if not (d / "done").exists():
            continue
        log = read_log(d / "beast.log", burnin_frac=PRIOR["mcmc"]["burnin_frac"])
        r = {}
        for col in [c for c in log if c == "Tree.height" or c.startswith("mrca.age") or c in ("deathRate", "ucldStdev")]:
            x = log[col]
            lo, hi = hpd(x)
            r[col] = {"median": float(np.median(x)), "hpd_lo": lo, "hpd_hi": hi, "ess": ess(x)}
        rows[name] = r
    if not rows:
        sys.exit("no finished variants")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"{'variant':15s} {'root median (BP)':>16s} {'95% HPD':>18s} {'ESS':>6s}  {'Proto-Mayan (AD/BC)':>20s}")
    for name, r in rows.items():
        t = r["Tree.height"]
        yr = lambda bp: f"{abs(2000 - bp):.0f} {'BC' if bp > 2000 else 'AD'}"
        print(f"{name:15s} {t['median']:16.0f} {t['hpd_lo']:8.0f}-{t['hpd_hi']:<8.0f} {t['ess']:6.0f}  "
              f"{yr(t['median']):>10s} ({yr(t['hpd_hi'])} - {yr(t['hpd_lo'])})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(rows) + 1.5))
    for k, (name, r) in enumerate(rows.items()):
        t = r["Tree.height"]
        ax.plot([t["hpd_lo"], t["hpd_hi"]], [k, k], color="#4a7bb7", lw=3)
        ax.plot(t["median"], k, "o", color="#4a7bb7")
    ax.axvline(KAUFMAN_BP, color="#d1495b", ls="--", label="Kaufman, c. 2200 BC (glottochronology)")
    ax.axvspan(PRIOR["root_age"]["lower"], PRIOR["root_age"]["upper"], color="#ddd", alpha=0.4, zorder=0,
               label="root prior (uniform)")
    ax.set_yticks(range(len(rows)), list(rows))
    ax.set_xlabel("Proto-Mayan age (years before AD 2000), posterior median and 95% HPD")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Milestone 5 (draft calibrations, no borrowing correction)")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "m5_root_age.png", dpi=150)
    print(f"figure -> {FIG / 'm5_root_age.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("run"); p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--variants", nargs="*")
    sub.add_parser("summarise")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
