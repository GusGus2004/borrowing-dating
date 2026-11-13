"""Milestone 5b: dating Proto-Mayan on PMED Swadesh-100 (expert etyma) vs ASJP 40 (LexStat).

Chicomuceltec is absent from PMED, so the Huastecan calibration cannot be
used; both datasets get the same calibrations so that differences come from
the data:

    pmed_yuc       PMED, Yucatecan crown Normal(1050, 200)
    pmed_yuc_ct    PMED, + candidate Cholan-Tzeltalan >= 1700 BP
    asjp_yuc       ASJP 40 / LexStat 0.75, Yucatecan only (control)
    asjp_yuc_ct    ASJP, + Cholan-Tzeltalan (control)
    pmed_prior     PMED, pmed_yuc_ct sampled from the prior (no data)
    pmed_ct_innov  pmed_yuc_ct with every missing (language, concept) cell coded as
                   a singleton cognate set (an unrecorded innovation): the opposite
                   bound to treating it as missing, since PMED lists only reflexes
                   of reconstructed etyma
    pmed40_ct      pmed_yuc_ct restricted to the ASJP 40 concepts (list length vs
                   cognate coding)

Other priors as in infer/beast/m5_mayan_prior.json (root Uniform(1500, 9000),
Yule estimated, loss-rate LogNormal(-9, 1), strict clock).

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
from infer.beastrun import BEAST, run_all  # noqa: E402
from infer.beastxml import load_prior, render_xml  # noqa: E402
from pipeline.cldf import MISSING, CognateMatrix, read_cognate_matrix  # noqa: E402

BASE = load_prior(ROOT / "infer/beast/m5_mayan_prior.json")
OUT = ROOT / "data/derived/m5b"
FIG = ROOT / "figures"
DATA = {"pmed": ROOT / "data/derived/pmed_mayan/Wordlist-metadata.json",
        "asjp": ROOT / "data/derived/asjp_mayan/Wordlist-metadata.json"}
TAXA = {  # Yucatecan and Cholan-Tzeltalan members in each dataset
    "pmed": {"Yucatecan": ["YUK", "ITZ", "LAK", "MOP"], "CholanTzeltalan": ["CHL", "CHR", "TZE", "TZO"]},
    "asjp": {"Yucatecan": ["ITZAJ", "LACANDON", "MAYA_YUCATECA", "MOPAN"],
             "CholanTzeltalan": ["CHOL_TUMBALA", "CHONTAL_TABASCO", "CHORTI", "TZELTAL_BACHAJON", "TZOTZIL_SAN_ANDRES"]},
}
YUC = next(c for c in BASE["calibrations"] if c["name"] == "Yucatecan")
CT = BASE["candidate_calibrations"][0]


def prior_for(data: str, cals: list[str]) -> dict:
    p = copy.deepcopy(BASE)
    p["calibrations"] = []
    for name in cals:
        c = copy.deepcopy(YUC if name == "Yucatecan" else CT)
        c["taxa"] = TAXA[data][name]
        p["calibrations"].append(c)
    return p


ASJP40 = ["I", "you", "we", "one", "two", "person", "fish", "dog", "louse", "tree", "leaf", "skin", "blood",
          "bone", "horn", "ear", "eye", "nose", "tooth", "tongue", "knee", "hand", "breast", "liver", "drink",
          "see", "hear", "die", "come", "sun", "star", "water", "stone", "fire", "path", "mountain", "night",
          "full", "new", "name"]


def innovations(cm: CognateMatrix) -> CognateMatrix:
    """Code each missing (language, concept) cell as a singleton cognate set."""
    cols, sets, concepts = [cm.matrix], list(cm.cognatesets), list(cm.concepts)
    m = cm.matrix.copy()
    for c in sorted(set(cm.concepts)):
        idx = [j for j, x in enumerate(cm.concepts) if x == c]
        for i, l in enumerate(cm.languages):
            if (m[i, idx] == MISSING).all():
                m[i, idx] = 0
                col = np.zeros((len(cm.languages), 1), dtype=np.int8)
                col[i] = 1
                cols.append(col); sets.append(f"{c}-innov-{l}"); concepts.append(c)
    cols[0] = m
    return CognateMatrix(cm.languages, sets, concepts, np.hstack(cols))


def restrict(cm: CognateMatrix, keep: list[str]) -> CognateMatrix:
    idx = [j for j, c in enumerate(cm.concepts) if c in keep]
    return CognateMatrix(cm.languages, [cm.cognatesets[j] for j in idx], [cm.concepts[j] for j in idx],
                         cm.matrix[:, idx])


def matrix_for(name: str, data: str) -> CognateMatrix:
    cm = read_cognate_matrix(DATA[data], concept_level_missing=True)
    if name.endswith("_innov"):
        return innovations(cm)
    if name.startswith("pmed40"):
        return restrict(cm, ASJP40)
    return cm


def variants() -> dict[str, tuple[str, dict, bool]]:
    return {"pmed_yuc": ("pmed", prior_for("pmed", ["Yucatecan"]), False),
            "pmed_yuc_ct": ("pmed", prior_for("pmed", ["Yucatecan", "CholanTzeltalan"]), False),
            "asjp_yuc": ("asjp", prior_for("asjp", ["Yucatecan"]), False),
            "asjp_yuc_ct": ("asjp", prior_for("asjp", ["Yucatecan", "CholanTzeltalan"]), False),
            "pmed_prior": ("pmed", prior_for("pmed", ["Yucatecan", "CholanTzeltalan"]), True),
            "pmed_ct_innov": ("pmed", prior_for("pmed", ["Yucatecan", "CholanTzeltalan"]), False),
            "pmed40_ct": ("pmed", prior_for("pmed", ["Yucatecan", "CholanTzeltalan"]), False)}


def cmd_prepare(a) -> None:
    for name, (data, prior, _) in variants().items():
        if a.variants and name not in a.variants:
            continue
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        cm = matrix_for(name, data)
        (d / "beast.xml").write_text(render_xml(cm, prior, stem="beast"))
        (d / "prior.json").write_text(json.dumps(prior, indent=2))
        print(f"{name}: {cm.shape[0]} languages x {cm.shape[1]} cognate sets; "
              f"missing {np.mean(cm.matrix < 0):.1%}")


def cmd_run(a) -> None:
    import subprocess
    v = variants()
    names = a.variants or list(v)
    # prior-only run: BEAST's -sampleFromPrior, run directly (fast)
    for n in [n for n in names if v[n][2]]:
        d = OUT / n
        if not (d / "done").exists():
            with open(d / "beast.out", "w") as out:
                r = subprocess.run(BEAST + ["-sampleFromPrior", "-overwrite", "-seed", "31337", "-threads", "1",
                                            "beast.xml"], cwd=d, stdout=out, stderr=subprocess.STDOUT)
            if r.returncode == 0:
                (d / "done").touch()
            print(f"{d}: prior {'ok' if r.returncode == 0 else 'FAILED'}", flush=True)
    run_all([OUT / n for n in names if not v[n][2]], lambda r: 31337, a.jobs)


def cmd_summarise(a) -> None:
    rows = {}
    for name in variants():
        d = OUT / name
        if not (d / "done").exists():
            continue
        log = read_log(d / "beast.log", burnin_frac=BASE["mcmc"]["burnin_frac"])
        rows[name] = {c: {"median": float(np.median(log[c])), "hpd": hpd(log[c]), "ess": ess(log[c])}
                      for c in log if c == "Tree.height" or c.startswith("mrca.age") or c == "deathRate"}
    if not rows:
        sys.exit("no finished variants")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    yr = lambda bp: f"{abs(2000 - bp):.0f} {'BC' if bp > 2000 else 'AD'}"
    print(f"{'variant':13s} {'root median':>11s} {'95% HPD (BP)':>15s} {'ESS':>5s}  {'Proto-Mayan':>10s}  "
          f"{'loss rate':>9s}  calibrated nodes (median BP)")
    for name, r in rows.items():
        t = r["Tree.height"]
        cal = ", ".join(f"{k.split('taxa_')[1].rstrip(')')} {v['median']:.0f}" for k, v in r.items() if k.startswith("mrca"))
        print(f"{name:13s} {t['median']:11.0f} {t['hpd'][0]:7.0f}-{t['hpd'][1]:<7.0f} {t['ess']:5.0f}  "
              f"{yr(t['median']):>10s}  {r['deathRate']['median']:9.2e}  {cal}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(rows) + 1.6))
    for k, (name, r) in enumerate(rows.items()):
        t = r["Tree.height"]
        col = "#999999" if name.endswith("prior") else ("#2a7f62" if name.startswith("pmed") else "#4a7bb7")
        ax.plot(t["hpd"], [k, k], color=col, lw=3)
        ax.plot(t["median"], k, "o", color=col)
    ax.axvline(4200, color="#d1495b", ls="--", label="Kaufman, c. 2200 BC (glottochronology)")
    ax.set_yticks(range(len(rows)), list(rows))
    ax.set_xlabel("Proto-Mayan age (years before AD 2000): posterior median and 95% HPD")
    ax.set_title("PMED Swadesh-100 (expert etyma, green) vs ASJP 40 (LexStat, blue)")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "m5b_root_age.png", dpi=150)
    print(f"figure -> {FIG / 'm5b_root_age.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").add_argument("--variants", nargs="*")
    p = sub.add_parser("run"); p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--variants", nargs="*")
    sub.add_parser("summarise")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
