"""Milestone 5c: the ASJP Mayan word lists dated with Kaufman's etyma instead of LexStat sets.

Milestone 5b: same calibrations, PMED/etyma c. 2800 BC vs ASJP/LexStat c.
900 BC. Milestone 6: LexStat errors alone bias simulated roots by only -8%.
This separates the two ingredients on the real data: the ASJP words with
expert cognate sets.

Expert labels as in ``scripts/benchmark_cognates.py --mode unambiguous``: an
ASJP form gets Kaufman's etymon when PMED has exactly one etymon for its
(language, concept) cell. Variants (all: Yucatecan + candidate
Cholan-Tzeltalan calibrations, as asjp_yuc_ct in 5b):

    sub_expert    labelled forms only, Kaufman's etyma; other cells missing
    sub_lexstat   the same forms, LexStat 0.75 sets (computed on all forms)
                  -> sub_expert vs sub_lexstat differ only in cognate coding
    full_lexstat  every form, LexStat 0.75 sets (baseline for hybrid)
    full_partial  every form, morpheme-level partial LexStat 0.85 (best on the
                  expert benchmark)
    sub_partial   the labelled forms only, partial LexStat 0.85
    split_lexstat the labelled forms, Kaufman's etyma with only LexStat's splits
                  applied (meet of the two partitions: grouped iff grouped by both)
    merge_lexstat ... with only LexStat's merges applied (join: grouped if grouped
                  by either, transitively)
    split_partial, merge_partial   the same with partial LexStat 0.85
    hybrid        every form: labelled ones take their etymon; an unlabelled
                  form takes the etymon most common among labelled forms in
                  its LexStat set, or keeps its LexStat set if there are none

Languages with fewer than 10 labelled forms are dropped from all variants.
Subcommands: prepare | run [--jobs N] | summarise
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analysis.mcmc import ess, hpd, read_log  # noqa: E402
from benchmark_cognates import ASJP, ASJP2SW, PMED  # noqa: E402
from infer.beastrun import run_all  # noqa: E402
from infer.beastxml import load_prior, render_xml  # noqa: E402
from pipeline.asjp import lexstat_cognates, partial_cognates, select_mayan  # noqa: E402
from pipeline.cldf import MISSING, CognateMatrix  # noqa: E402
from pipeline.pmed import match  # noqa: E402

OUT = ROOT / "data/derived/m5c"
FIG = ROOT / "figures"
BASE = load_prior(ROOT / "infer/beast/m5_mayan_prior.json")
YUC = next(c for c in BASE["calibrations"] if c["name"] == "Yucatecan")
CT = BASE["candidate_calibrations"][0]
TAXA = {"Yucatecan": ["ITZAJ", "LACANDON", "MAYA_YUCATECA", "MOPAN"],
        "CholanTzeltalan": ["CHOL_TUMBALA", "CHONTAL_TABASCO", "CHORTI", "TZELTAL_BACHAJON", "TZOTZIL_SAN_ANDRES"]}
MIN_LABELLED = 10
VARIANTS = ("sub_expert", "sub_lexstat", "sub_partial", "split_lexstat", "merge_lexstat", "split_partial",
            "merge_partial", "full_lexstat", "full_partial", "hybrid")


def labelled_forms():
    langs, params, forms = select_mayan(ASJP)
    forms = [f for f in forms if f["Loan"] != "true" and f["Segments"].strip()]
    glott = {l["ID"]: l["Glottocode"] for l in langs}
    pname = {p["ID"]: p["Name"].lstrip("*") for p in params}
    plangs, pm = match(PMED)
    ets = collections.defaultdict(set)
    for f, c, e in pm:
        if e:
            ets[(plangs[f["Language_ID"]]["Glottocode"], c)].add(e)
    concept = {f["ID"]: ASJP2SW.get(pname[f["Parameter_ID"]], pname[f["Parameter_ID"]]) for f in forms}
    gold = {}
    for f in forms:
        e = ets.get((glott[f["Language_ID"]], concept[f["ID"]]))
        if e and len(e) == 1:
            gold[f["ID"]] = f"{concept[f['ID']]}-{next(iter(e))}"
    return [l["ID"] for l in langs], forms, concept, gold


def meet(a: dict, b: dict) -> dict:
    """Finest common coarsening's opposite: forms together iff together in both partitions."""
    return {f: f"{a[f]}|{b[f]}" for f in a}


def join(a: dict, b: dict) -> dict:
    """Forms together if together in either partition (transitive closure, union-find)."""
    parent = {f: f for f in a}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for part in (a, b):
        first = {}
        for f, s in part.items():
            if s in first:
                parent[find(f)] = find(first[s])
            else:
                first[s] = f
    return {f: f"j{find(f)}" for f in a}


def matrix(languages: list[str], forms: list[dict], concept: dict, cog: dict) -> CognateMatrix:
    """Concept-level missing: a language with no included form for a concept is MISSING for it."""
    lang = {f["ID"]: f["Language_ID"] for f in forms}
    sets = sorted({cog[f] for f in cog})
    col = {s: j for j, s in enumerate(sets)}
    row = {l: i for i, l in enumerate(languages)}
    set_concept = {cog[f]: concept[f] for f in cog}
    m = np.zeros((len(languages), len(sets)), dtype=np.int8)
    has = collections.defaultdict(set)
    for f, s in cog.items():
        if lang[f] in row:
            m[row[lang[f]], col[s]] = 1
            has[lang[f]].add(concept[f])
    for l, i in row.items():
        for j, s in enumerate(sets):
            if set_concept[s] not in has[l]:
                m[i, j] = MISSING
    keep = np.flatnonzero((m == 1).any(axis=0))
    return CognateMatrix(languages, [sets[j] for j in keep], [set_concept[sets[j]] for j in keep], m[:, keep])


def cmd_prepare(a) -> None:
    languages, forms, concept, gold = labelled_forms()
    n_lab = collections.Counter(f["Language_ID"] for f in forms if f["ID"] in gold)
    keep_l = [l for l in languages if n_lab[l] >= MIN_LABELLED]
    print(f"labelled {len(gold)} of {len(forms)} forms; languages kept {len(keep_l)} of {len(languages)}; "
          f"dropped {[l for l in languages if l not in keep_l]}")
    forms = [f for f in forms if f["Language_ID"] in keep_l]
    lex = lexstat_cognates(forms, 0.75)
    codings = {"sub_expert": {f: gold[f] for f in lex if f in gold},
               "sub_lexstat": {f: lex[f] for f in lex if f in gold}}
    members = collections.defaultdict(list)
    for f, s in lex.items():
        if f in gold:
            members[s].append(gold[f])
    hyb = {}
    for f, s in lex.items():
        hyb[f] = gold[f] if f in gold else (collections.Counter(members[s]).most_common(1)[0][0] if members[s] else s)
    codings["hybrid"] = hyb
    codings["full_lexstat"] = dict(lex)
    part = partial_cognates(forms, 0.85, method="lexstat")
    codings["full_partial"] = dict(part)
    codings["sub_partial"] = {f: part[f] for f in part if f in gold}
    for m in ("lexstat", "partial"):
        auto = codings[f"sub_{m}"]
        codings[f"split_{m}"] = meet(codings["sub_expert"], auto)
        codings[f"merge_{m}"] = join(codings["sub_expert"], auto)
    prior = copy.deepcopy(BASE)
    prior["calibrations"] = []
    for c, name in ((YUC, "Yucatecan"), (CT, "CholanTzeltalan")):
        c = copy.deepcopy(c)
        c["taxa"] = [t for t in TAXA[name] if t in keep_l]
        prior["calibrations"].append(c)
    for name, cog in codings.items():
        if a.only and name not in a.only:
            continue
        cm = matrix(keep_l, forms, concept, cog)
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "beast.xml").write_text(render_xml(cm, prior, stem="beast"))
        (d / "prior.json").write_text(json.dumps(prior, indent=2))
        print(f"{name}: {cm.shape[0]} languages x {cm.shape[1]} sets ({cm.shape[1] / len(set(cm.concepts)):.2f} per "
              f"concept); missing {np.mean(cm.matrix < 0):.1%}; calibration taxa "
              f"{[len(c['taxa']) for c in prior['calibrations']]}")


def cmd_run(a) -> None:
    run_all([OUT / v for v in VARIANTS], lambda r: 31337, a.jobs)


def cmd_summarise(a) -> None:
    rows = {}
    for name in VARIANTS:
        d = OUT / name
        if not (d / "done").exists():
            continue
        log = read_log(d / "beast.log", burnin_frac=BASE["mcmc"]["burnin_frac"])
        rows[name] = {c: {"median": float(np.median(log[c])), "hpd": hpd(log[c]), "ess": ess(log[c])}
                      for c in log if c == "Tree.height" or c.startswith("mrca.age") or c == "deathRate"}
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    yr = lambda bp: f"{abs(2000 - bp):.0f} {'BC' if bp > 2000 else 'AD'}"
    for name, r in rows.items():
        t = r["Tree.height"]
        cal = ", ".join(f"{k.split('taxa_')[1].rstrip(')')} {v['median']:.0f}" for k, v in r.items() if k.startswith("mrca"))
        print(f"{name:12s} root {t['median']:5.0f} BP ({yr(t['median'])}), HPD {t['hpd'][0]:.0f}-{t['hpd'][1]:.0f}, "
              f"ESS {t['ess']:.0f}; loss {r['deathRate']['median']:.2e}; {cal}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [("Same 857 labelled forms", [("sub_expert", "Kaufman's etyma"), ("merge_lexstat", "etyma + LexStat's merges only"),
                                           ("split_lexstat", "etyma + LexStat's splits only"), ("sub_lexstat", "LexStat"),
                                           ("merge_partial", "etyma + partial's merges only"),
                                           ("split_partial", "etyma + partial's splits only"), ("sub_partial", "partial LexStat")]),
              ("All 1,317 forms", [("hybrid", "etyma where known (hybrid)"), ("full_partial", "partial LexStat"),
                                   ("full_lexstat", "LexStat")])]
    col = {"sub_expert": "#2a7f62", "hybrid": "#2a7f62", "merge_lexstat": "#7fb89f", "merge_partial": "#7fb89f",
           "split_lexstat": "#e08a8f", "split_partial": "#e08a8f"}
    items = [(g, n, lab) for g, xs in groups for n, lab in xs if n in rows]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(items) + 1.6))
    for k, (g, n, lab) in enumerate(items):
        t = rows[n]["Tree.height"]
        y = len(items) - 1 - k
        c = col.get(n, "#4a7bb7")
        ax.plot(t["hpd"], [y, y], color=c, lw=3)
        ax.plot(t["median"], y, "o", color=c)
    ax.set_yticks([len(items) - 1 - k for k in range(len(items))], [lab for _, _, lab in items], fontsize=8)
    ax.axvline(9000, color="#999999", lw=0.8, ls=":")
    ax.axvline(4200, color="#d1495b", ls="--", lw=1, label="Kaufman, c. 2200 BC")
    n1 = sum(1 for g, _, _ in items if g == groups[0][0])
    ax.axhline(len(items) - n1 - 0.5, color="#cccccc", lw=0.8)
    ax.set_xlabel("Proto-Mayan age (years before AD 2000); dotted line = prior upper bound")
    ax.set_title("ASJP Mayan words: only the cognate coding differs")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "m5c_cognate_coding.png", dpi=150)
    print(f"figure -> {FIG / 'm5c_cognate_coding.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").add_argument("--only", nargs="*")
    p = sub.add_parser("run"); p.add_argument("--jobs", type=int, default=4)
    sub.add_parser("summarise")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
