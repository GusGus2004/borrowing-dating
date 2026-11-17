"""Milestone 5d: the IE threshold remedy (milestone 7) tried on Mayan.

On IE-CoR, high-precision LexStat thresholds (0.45, 0.55) recover the expert root while the
best-F threshold does not (LexStat over-merges there). On Mayan LexStat already over-splits,
so lowering the threshold should make the root younger, not fix it. Same setup as the 5c
complete-list runs (27 ASJP languages, all 1,317 forms, asjp_yuc_ct calibrations, seed 31337);
references: 5c full_lexstat (0.75) and hybrid (Kaufman's etyma where known).

Subcommands: prepare | run [--jobs N] | summarise
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from analysis.mcmc import ess, hpd, read_log  # noqa: E402
from infer.beastrun import run_all  # noqa: E402
from infer.beastxml import render_xml  # noqa: E402
from m5c_asjp_expert import BASE, CT, MIN_LABELLED, TAXA, YUC, labelled_forms, matrix  # noqa: E402
from pipeline.asjp import lexstat_cognates  # noqa: E402
from sim_realism import bcubed  # noqa: E402

OUT = ROOT / "data/derived/m5d"
M5C = ROOT / "data/derived/m5c"
THRESHOLDS = (0.45, 0.55, 0.65)


def cmd_prepare(a) -> None:
    languages, forms, concept, gold = labelled_forms()
    n_lab = collections.Counter(f["Language_ID"] for f in forms if f["ID"] in gold)
    keep_l = [l for l in languages if n_lab[l] >= MIN_LABELLED]
    forms = [f for f in forms if f["Language_ID"] in keep_l]
    lang = {f["ID"]: f["Language_ID"] for f in forms}
    prior = copy.deepcopy(BASE)
    prior["calibrations"] = []
    for c, name in ((YUC, "Yucatecan"), (CT, "CholanTzeltalan")):
        c = copy.deepcopy(c)
        c["taxa"] = [t for t in TAXA[name] if t in keep_l]
        prior["calibrations"].append(c)
    stats = {}
    for t in THRESHOLDS + (0.75,):
        with contextlib.redirect_stderr(io.StringIO()):
            lex = lexstat_cognates(forms, t)
        lab = [f for f in lex if f in gold]
        P, R, F = bcubed({f: gold[f] for f in lab}, {f: lex[f] for f in lab})
        sets = collections.defaultdict(set)
        for f, s in lex.items():
            sets[s].add(lang[f])
        name = f"full_t{round(t * 100)}"
        stats[name] = {"threshold": t, "P": P, "R": R, "F": F, "sets": len(sets),
                       "sets_labelled": len({lex[f] for f in lab}), "gold_sets_labelled": len({gold[f] for f in lab}),
                       "singleton_share": float(np.mean([len(v) == 1 for v in sets.values()]))}
        print(f"{name}: {len(sets)} sets, singletons {stats[name]['singleton_share']:.2f}; on labelled forms "
              f"{stats[name]['sets_labelled']} vs {stats[name]['gold_sets_labelled']} gold sets, P {P:.3f} R {R:.3f} "
              f"F {F:.3f}", flush=True)
        if t == 0.75:
            continue  # the 5c full_lexstat run is this coding
        cm = matrix(keep_l, forms, concept, lex)
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "beast.xml").write_text(render_xml(cm, prior, stem="beast"))
        (d / "prior.json").write_text(json.dumps(prior, indent=2))
    (OUT / "coding_stats.json").write_text(json.dumps(stats, indent=2))


def cmd_run(a) -> None:
    run_all([OUT / f"full_t{round(t * 100)}" for t in THRESHOLDS], lambda r: 31337, a.jobs)


def cmd_summarise(a) -> None:
    stats = json.loads((OUT / "coding_stats.json").read_text())
    rows = {}
    for name, d in [(f"full_t{round(t * 100)}", OUT / f"full_t{round(t * 100)}") for t in THRESHOLDS] + \
                   [("full_t75 (5c full_lexstat)", M5C / "full_lexstat"), ("hybrid (5c)", M5C / "hybrid")]:
        if not (d / "beast.log").exists():
            continue
        log = read_log(d / "beast.log", burnin_frac=BASE["mcmc"]["burnin_frac"])
        x = log["Tree.height"]
        rows[name] = {"median": float(np.median(x)), "hpd": hpd(x), "ess": ess(x), "done": (d / "done").exists(),
                      "loss": float(np.median(log["deathRate"]))}
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    yr = lambda bp: f"{abs(2000 - bp):.0f} {'BC' if bp > 2000 else 'AD'}"
    for name, r in rows.items():
        s = stats.get(name.split(" ")[0], {})
        extra = f"; F {s['F']:.3f} P {s['P']:.3f} sets {s['sets']}" if s else ""
        print(f"{name:28s} root {r['median']:5.0f} BP ({yr(r['median'])}), HPD {r['hpd'][0]:.0f}-{r['hpd'][1]:.0f}, "
              f"ESS {r['ess']:.0f}; loss {r['loss']:.2e}{extra}{'' if r['done'] else ' (RUNNING)'}")


def cmd_figure(a) -> None:
    """Both families: root age against LexStat threshold, with the expert reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    may = json.loads((OUT / "summary.json").read_text())
    ie = json.loads((ROOT / "data/derived/m7/summary.json").read_text())
    ie_stats = json.loads((ROOT / "data/derived/m7/coding_stats.json").read_text())
    panels = [("Mayan (ASJP, 27 languages): LexStat over-splits",
               [(0.45, may["full_t45"]), (0.55, may["full_t55"]), (0.65, may["full_t65"]),
                (0.75, may["full_t75 (5c full_lexstat)"])], may["hybrid (5c)"], "Kaufman's etyma where known (hybrid)"),
              ("Indo-European (IE-CoR, 40 languages): LexStat over-merges",
               [(ie_stats[n]["threshold"], ie[n]["Tree.height"]) for n in ("lexstat_t45", "lexstat_t55", "lexstat_t65",
                                                                          "lexstat")], ie["expert"]["Tree.height"],
               "IE-CoR expert sets")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, pts, ref, lab) in zip(axs, panels):
        ax.axhspan(*ref["hpd"], color="#2a7f62", alpha=0.15, label=lab + ", 95% HPD")
        ax.axhline(ref["median"], color="#2a7f62", lw=1)
        for t, r in pts:
            ax.plot([t, t], r["hpd"], color="#4a7bb7", lw=3)
            ax.plot(t, r["median"], "o", color="#4a7bb7")
        ax.set_xlabel("LexStat threshold (lower = higher precision, more splitting)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
    axs[0].set_ylabel("root age, BP (draft calibrations)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/m5d_m7_thresholds.png", dpi=150)
    print("figure -> figures/m5d_m7_thresholds.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("run"); p.add_argument("--jobs", type=int, default=3)
    sub.add_parser("summarise")
    sub.add_parser("figure")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise, "figure": cmd_figure}[a.cmd](a)


if __name__ == "__main__":
    main()
