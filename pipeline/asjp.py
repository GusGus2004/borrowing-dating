"""Mayan basic-vocabulary dataset from ASJP with automated cognate detection.

* Doculects: ASJP ``Family == "Mayan"``; one per Glottolog languoid (the list
  with most of the 40 ASJP core concepts; ties broken by name), dropping lists
  with fewer than ``min_core`` core concepts.
* Concepts: the 40 ASJP core items (``*`` in the parameter name).
* Forms flagged ``Loan`` in ASJP are dropped (mostly Spanish loans: borrowing
  from outside the family, which this project does not model); the language
  is then missing for that concept unless it has another form.
* Cognates: LingPy LexStat distances clustered with UPGMA at ``threshold``,
  scorer from ``runs`` permutations, fixed seed. Clusters are within concepts.
  The default 0.75 was the best of 0.45-0.75 on simulated wordlists of the
  same shape (31 languages x 40 concepts, root ~4000 y): B-cubed P 0.93,
  R 0.93, F 0.93; LingPy's usual 0.55 gave R 0.72 and visibly split obvious
  Mayan cognates (e.g. reflexes of *ka'ib' "two").

Output is a CLDF Wordlist (forms, cognates, cognate sets), so it goes through
``pipeline.cldf.read_cognate_matrix`` exactly like the simulated data.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycldf import Wordlist


def select_mayan(asjp_cldf: Path | str, min_core: int = 28) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (languages, core concepts, forms) for the chosen Mayan doculects."""
    d = Path(asjp_cldf)
    langs = {r["ID"]: r for r in csv.DictReader(open(d / "languages.csv")) if r["Family"] == "Mayan"}
    params = {r["ID"]: r for r in csv.DictReader(open(d / "parameters.csv")) if r["Name"].startswith("*")}
    forms = [r for r in csv.DictReader(open(d / "forms.csv"))
             if r["Language_ID"] in langs and r["Parameter_ID"] in params]
    core = defaultdict(set)
    for f in forms:
        core[f["Language_ID"]].add(f["Parameter_ID"])
    best: dict[str, str] = {}
    for lid, r in sorted(langs.items()):
        g = r["Glottocode"] or lid
        if len(core[lid]) < min_core:
            continue
        if g not in best or len(core[lid]) > len(core[best[g]]):
            best[g] = lid
    keep = sorted(best.values())
    return ([langs[l] for l in keep], [params[p] for p in sorted(params, key=int)],
            [f for f in forms if f["Language_ID"] in keep])


def lexstat_cognates(forms: list[dict], threshold: float = 0.75, runs: int = 1000,
                     seed: int = 1) -> dict[str, str]:
    """Form ID -> cognate set ID ("<concept>-<cluster>") by LingPy LexStat."""
    from lingpy import LexStat

    random.seed(seed)
    np.random.seed(seed)
    D = {0: ["doculect", "concept", "ipa", "tokens", "form_id"]}
    for i, f in enumerate(forms, start=1):
        D[i] = [f["Language_ID"], f["Parameter_ID"], f["Form"], f["Segments"].split(), f["ID"]]
    lex = LexStat(D, check=False)
    lex.get_scorer(runs=runs, preprocessing=False)
    lex.cluster(method="lexstat", threshold=threshold, ref="cogid", cluster_method="upgma")
    return {lex[i, "form_id"]: f"{lex[i, 'concept']}-{lex[i, 'cogid']}" for i in lex}


def build(asjp_cldf: Path | str, outdir: Path | str, threshold: float = 0.75, runs: int = 1000,
          min_core: int = 28, seed: int = 1) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    langs, params, forms = select_mayan(asjp_cldf, min_core)
    n_loans = sum(f["Loan"] == "true" for f in forms)
    forms = [f for f in forms if f["Loan"] != "true" and f["Segments"].strip()]
    cog = lexstat_cognates(forms, threshold, runs, seed)

    ds = Wordlist.in_dir(outdir)
    ds.add_component("LanguageTable")
    ds.add_component("ParameterTable")
    ds.add_component("CognateTable")
    ds.add_component("CognatesetTable")
    ds.properties["dc:title"] = "Mayan ASJP core vocabulary with LexStat cognates"
    ds.properties["dc:description"] = (
        f"Derived from lexibank/asjp v21.1: one doculect per Glottolog languoid, 40 core concepts, "
        f"ASJP-flagged loans removed, cognates by LingPy LexStat (threshold {threshold}, {runs} runs, "
        f"seed {seed}).")
    ds.write(
        LanguageTable=[{"ID": l["ID"], "Name": l["Name"], "Glottocode": l["Glottocode"] or None,
                        "Latitude": l["Latitude"] or None, "Longitude": l["Longitude"] or None}
                       for l in langs],
        ParameterTable=[{"ID": p["ID"], "Name": p["Name"].lstrip("*"),
                         "Concepticon_ID": p["Concepticon_ID"] or None} for p in params],
        FormTable=[{"ID": f["ID"], "Language_ID": f["Language_ID"], "Parameter_ID": f["Parameter_ID"],
                    "Value": f["Value"], "Form": f["Form"], "Segments": f["Segments"].split()}
                   for f in forms],
        CognateTable=[{"ID": f"{f['ID']}-cog", "Form_ID": f["ID"], "Cognateset_ID": cog[f["ID"]]}
                      for f in forms],
        CognatesetTable=[{"ID": c} for c in sorted(set(cog.values()))],
    )
    sizes = Counter(cog.values())
    return {"n_languages": len(langs), "n_concepts": len(params), "n_forms": len(forms),
            "n_asjp_loans_removed": n_loans, "n_cognatesets": len(sizes),
            "n_singleton_sets": sum(v == 1 for v in sizes.values()),
            "languages": [l["Name"] for l in langs]}
