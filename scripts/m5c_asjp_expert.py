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
from pipeline.asjp import lexstat_cognates, select_mayan  # noqa: E402
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
VARIANTS = ("sub_expert", "sub_lexstat", "full_lexstat", "hybrid")


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
    prior = copy.deepcopy(BASE)
    prior["calibrations"] = []
    for c, name in ((YUC, "Yucatecan"), (CT, "CholanTzeltalan")):
        c = copy.deepcopy(c)
        c["taxa"] = [t for t in TAXA[name] if t in keep_l]
        prior["calibrations"].append(c)
    for name, cog in codings.items():
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


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("run"); p.add_argument("--jobs", type=int, default=4)
    sub.add_parser("summarise")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
