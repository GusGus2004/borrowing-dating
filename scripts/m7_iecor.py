"""Milestone 7: replicate the cognate-coding result (5c, 6s) on Indo-European (IE-CoR).

Data: IE-CoR v1.2.1 (lexibank/iecor, Heggarty, Anderson & Scarborough; Anderson et al. 2025),
in data/raw/iecor. Expert cognate sets and phonetic segments for the same words, so the
expert and automated codings are compared on identical forms (cleaner than 5c, where ASJP
words were labelled with Kaufman's etyma).

40 modern languages from seven branches with near-complete segmented forms (Albanian and
English have no segments in IE-CoR, so they are absent), all 170 concepts, only forms with
segments. Codings:
  expert         IE-CoR cognate sets
  lexstat        LingPy LexStat 0.75 on the same segments (as in 5/5c)
  split_lexstat  meet(expert, lexstat): expert sets plus only LexStat's splits
  merge_lexstat  join(expert, lexstat): expert sets plus only LexStat's merges
  expert_noloans expert with forms in IE-CoR loan sets removed

Calibrations: modern-only clade ages from the historical record (no tip dating yet),
see infer/beast/m7_iecor_prior.json. DRAFT: the comparison between codings is the point;
absolute dates are not to be quoted.

Subcommands: prepare | run [--jobs N] | summarise
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from analysis.mcmc import ess, hpd, read_log  # noqa: E402
from infer.beastrun import run_all  # noqa: E402
from infer.beastxml import load_prior, render_xml  # noqa: E402
from m5c_asjp_expert import join, matrix, meet  # noqa: E402
from pipeline.asjp import lexstat_cognates  # noqa: E402

RAW = ROOT / "data/raw/iecor"
OUT = ROOT / "data/derived/m7"
FIG = ROOT / "figures"
PRIOR = ROOT / "infer/beast/m7_iecor_prior.json"
LANGUAGES = [
    "Armenian: Western", "Armenian: Eastern",
    "Greek: Modern Std", "Greek: Pontic",
    "Gaelic: Irish", "Welsh: North", "Breton: Treger",
    "German", "Frisian", "Luxembourgish", "Danish", "Norwegian: Bokmål", "Swedish", "Icelandic",
    "Spanish", "Portuguese", "Catalan", "French", "Italian", "Romanian", "Sardinian: Logudoro",
    "Lithuanian", "Latvian", "Russian", "Ukrainian", "Polish", "Czech", "Bulgarian", "Serbo-Croat", "Slovene",
    "Hindi", "Bengali", "Marathi", "Nepali", "Kashmiri", "Palula", "Pashto", "Ossetic: Iron", "Bakhtiari", "Kamviri",
]
THRESHOLDS = (0.45, 0.55, 0.65)  # remedy test; 0.75 is the main "lexstat" run
TH_CHAIN = 3_000_000  # main runs reached root ESS ~1500 at 10M; 3M gives ~450
VARIANTS = ("expert", "lexstat", "split_lexstat", "merge_lexstat", "expert_noloans") + tuple(
    f"lexstat_t{round(t * 100)}" for t in THRESHOLDS)


def load():
    """-> (taxa, forms, concept, expert, loan_forms); forms have ID, Language_ID (taxon), Parameter_ID, Form, Segments."""
    langs = {r["ID"]: r for r in csv.DictReader(open(RAW / "languages.csv"))}
    name2id = {r["Name"]: i for i, r in langs.items()}
    missing = [n for n in LANGUAGES if n not in name2id]
    if missing:
        sys.exit(f"unknown languages {missing}")
    taxon = {name2id[n]: langs[name2id[n]]["ascii_name"] for n in LANGUAGES}
    cog = collections.defaultdict(list)
    for r in csv.DictReader(open(RAW / "cognates.csv")):
        cog[r["Form_ID"]].append(r["Cognateset_ID"])
    loans = {r["Cognateset_ID"] for r in csv.DictReader(open(RAW / "loans.csv"))}
    forms, expert, loan_forms = [], {}, set()
    for f in csv.DictReader(open(RAW / "forms.csv")):
        if f["Language_ID"] not in taxon or not f["Segments"].strip() or not cog.get(f["ID"]):
            continue
        sets = sorted(set(cog[f["ID"]]))
        forms.append({"ID": f["ID"], "Language_ID": taxon[f["Language_ID"]], "Parameter_ID": f["Parameter_ID"],
                      "Form": f["Form"], "Segments": f["Segments"]})
        expert[f["ID"]] = sets[0]  # a handful of forms sit in two sets; keep one, deterministically
        if any(s in loans for s in sets):
            loan_forms.add(f["ID"])
    concept = {f["ID"]: f["Parameter_ID"] for f in forms}
    return [taxon[name2id[n]] for n in LANGUAGES], forms, concept, expert, loan_forms


def shape(cog: dict, lang: dict) -> dict:
    sets = collections.defaultdict(set)
    for f, s in cog.items():
        sets[s].add(lang[f])
    n = np.array([len(v) for v in sets.values()])
    return {"sets": len(sets), "singleton_share": float((n == 1).mean()), "mean_languages": float(n.mean())}


def cmd_prepare(a) -> None:
    taxa, forms, concept, expert, loan_forms = load()
    lang = {f["ID"]: f["Language_ID"] for f in forms}
    print(f"{len(taxa)} languages, {len(forms)} segmented forms with expert sets, {len(set(concept.values()))} "
          f"concepts; {len(loan_forms)} forms in loan sets", flush=True)
    with contextlib.redirect_stderr(io.StringIO()):
        lex = lexstat_cognates(forms, 0.75)
    from sim_realism import bcubed
    P, R, F = bcubed(expert, lex)
    codings = {"expert": dict(expert), "lexstat": dict(lex), "split_lexstat": meet(expert, lex),
               "merge_lexstat": join(expert, lex),
               "expert_noloans": {f: s for f, s in expert.items() if f not in loan_forms}}
    prior = load_prior(PRIOR)
    stats = {"lexstat_vs_expert": {"P": P, "R": R, "F": F}}
    for name, cog in codings.items():
        cm = matrix(taxa, forms, concept, cog)
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "beast.xml").write_text(render_xml(cm, prior, stem="beast"))
        (d / "prior.json").write_text(json.dumps(prior, indent=2))
        stats[name] = {**shape(cog, lang), "columns": cm.shape[1], "missing": float(np.mean(cm.matrix < 0))}
        print(f"{name:15s} {cm.shape[0]} x {cm.shape[1]} sets; singleton share {stats[name]['singleton_share']:.2f}; "
              f"missing {stats[name]['missing']:.1%}", flush=True)
    print(f"LexStat vs expert B-cubed P {P:.3f} R {R:.3f} F {F:.3f}")
    (OUT / "coding_stats.json").write_text(json.dumps(stats, indent=2))


def cmd_thresholds(a) -> None:
    """Remedy test: does some LexStat threshold recover the expert date, and is it the best-scoring one?"""
    from sim_realism import bcubed
    taxa, forms, concept, expert, _ = load()
    lang = {f["ID"]: f["Language_ID"] for f in forms}
    prior = load_prior(PRIOR)
    prior["mcmc"] = dict(prior["mcmc"], chain_length=TH_CHAIN)
    stats = json.loads((OUT / "coding_stats.json").read_text())
    for t in THRESHOLDS:
        with contextlib.redirect_stderr(io.StringIO()):
            lex = lexstat_cognates(forms, t)
        P, R, F = bcubed(expert, lex)
        name = f"lexstat_t{round(t * 100)}"
        cm = matrix(taxa, forms, concept, lex)
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "beast.xml").write_text(render_xml(cm, prior, stem="beast"))
        (d / "prior.json").write_text(json.dumps(prior, indent=2))
        stats[name] = {**shape(lex, lang), "columns": cm.shape[1], "missing": float(np.mean(cm.matrix < 0)),
                       "threshold": t, "P": P, "R": R, "F": F}
        print(f"{name}: {cm.shape[1]} sets, singletons {stats[name]['singleton_share']:.2f}; P {P:.3f} R {R:.3f} "
              f"F {F:.3f}", flush=True)
    stats["lexstat"].update(threshold=0.75, **stats["lexstat_vs_expert"])
    (OUT / "coding_stats.json").write_text(json.dumps(stats, indent=2))


def cmd_run(a) -> None:
    run_all([OUT / v for v in VARIANTS if (OUT / v).exists()], lambda r: 27182, a.jobs)


def cmd_summarise(a) -> None:
    prior = load_prior(PRIOR)
    rows = {}
    for name in VARIANTS:
        d = OUT / name
        if not (d / "beast.log").exists():
            continue
        log = read_log(d / "beast.log", burnin_frac=prior["mcmc"]["burnin_frac"])
        rows[name] = {c: {"median": float(np.median(log[c])), "hpd": hpd(log[c]), "ess": ess(log[c])}
                      for c in log if c == "Tree.height" or c.startswith("mrca.age") or c == "deathRate"}
        rows[name]["done"] = (d / "done").exists()
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    stats = json.loads((OUT / "coding_stats.json").read_text())
    for name, r in rows.items():
        t = r["Tree.height"]
        print(f"{name:15s} root {t['median']:6.0f} BP, HPD {t['hpd'][0]:.0f}-{t['hpd'][1]:.0f}, ESS {t['ess']:.0f}; "
              f"loss {r['deathRate']['median']:.2e}; sets {stats[name]['sets']}, singletons "
              f"{stats[name]['singleton_share']:.2f}{'' if r['done'] else ' (RUNNING)'}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = {"expert": "IE-CoR expert sets", "expert_noloans": "expert, known loans removed",
           "merge_lexstat": "expert + LexStat's merges only", "split_lexstat": "expert + LexStat's splits only",
           "lexstat": "LexStat"}
    col = {"expert": "#2a7f62", "expert_noloans": "#2a7f62", "merge_lexstat": "#7fb89f", "split_lexstat": "#e08a8f",
           "lexstat": "#4a7bb7"}
    order = [v for v in ("expert", "expert_noloans", "merge_lexstat", "split_lexstat", "lexstat") if v in rows]
    lab.update({n: f"LexStat {stats[n]['threshold']}" for n in rows if n.startswith("lexstat_t")})
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(order) + 1.4))
    for k, n in enumerate(order):
        t = rows[n]["Tree.height"]
        y = len(order) - 1 - k
        ax.plot(t["hpd"], [y, y], color=col[n], lw=3)
        ax.plot(t["median"], y, "o", color=col[n])
    ax.set_yticks([len(order) - 1 - k for k in range(len(order))], [lab[n] for n in order], fontsize=8)
    ax.set_xlabel("Proto-Indo-European root age (years before AD 2000); draft calibrations")
    ax.set_title("IE-CoR, 40 modern languages: only the cognate coding differs")
    fig.tight_layout()
    fig.savefig(FIG / "m7_iecor.png", dpi=150)
    print(f"figure -> {FIG / 'm7_iecor.png'}")
    th = sorted((stats[n]["threshold"], n) for n in rows if n.startswith("lexstat") and "threshold" in stats.get(n, {}))
    if len(th) > 1:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        e = rows["expert"]["Tree.height"]
        ax.axhspan(*e["hpd"], color="#2a7f62", alpha=0.15, label="expert sets, 95% HPD")
        ax.axhline(e["median"], color="#2a7f62", lw=1)
        for t, n in th:
            r = rows[n]["Tree.height"]
            ax.plot([t, t], r["hpd"], color="#4a7bb7", lw=3)
            ax.plot(t, r["median"], "o", color="#4a7bb7")
        ax.set_xlabel("LexStat threshold")
        ax.set_ylabel("root age, BP (draft calibrations)")
        ax2 = ax.twinx()
        ax2.plot([t for t, _ in th], [stats[n]["F"] for _, n in th], "s--", color="#d1495b", label="B-cubed F vs expert")
        ax2.set_ylabel("B-cubed F vs expert", color="#d1495b")
        ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
        ax.set_title("IE-CoR: can a LexStat threshold recover the expert date?")
        fig.tight_layout()
        fig.savefig(FIG / "m7_thresholds.png", dpi=150)
        print(f"figure -> {FIG / 'm7_thresholds.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("run"); p.add_argument("--jobs", type=int, default=5)
    sub.add_parser("summarise")
    sub.add_parser("thresholds")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise, "thresholds": cmd_thresholds}[a.cmd](a)


if __name__ == "__main__":
    main()
