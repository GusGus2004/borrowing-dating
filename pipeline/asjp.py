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
                     seed: int = 1, method: str = "lexstat") -> dict[str, str]:
    """Form ID -> cognate set ID ("<concept>-<cluster>") by LingPy LexStat.

    ``method="sca"`` uses LingPy's language-independent SCA distances instead
    of LexStat's pair-specific correspondence scores (more robust on short lists).
    """
    from lingpy import LexStat

    random.seed(seed)
    np.random.seed(seed)
    D = {0: ["doculect", "concept", "ipa", "tokens", "form_id"]}
    for i, f in enumerate(forms, start=1):
        D[i] = [f["Language_ID"], f["Parameter_ID"], f["Form"], f["Segments"].split(), f["ID"]]
    lex = LexStat(D, check=False)
    if method == "lexstat":
        lex.get_scorer(runs=runs, preprocessing=False)
    lex.cluster(method=method, threshold=threshold, ref="cogid", cluster_method="upgma")
    return {lex[i, "form_id"]: f"{lex[i, 'concept']}-{lex[i, 'cogid']}" for i in lex}


VOWEL_SEGS = set("aeiouɐɛɔəɪʊæɨ")


def discover_suffixes(forms: list[dict], min_concepts: int = 4, max_len: int = 3) -> list[tuple[str, ...]]:
    """Data-driven suffixes: VC(C) tails that turn one form of a concept into another.

    For every concept, if form A = form B + tail (B at least 2 segments, tail
    1-``max_len`` segments), the tail is a candidate. Tails seen in at least
    ``min_concepts`` concepts and starting with a vowel are kept (single
    consonants such as glottal stops are too often phonological).
    """
    by: dict[str, set] = defaultdict(set)
    for f in forms:
        t = tuple(f["Segments"].split())
        if "+" not in t:
            by[f["Parameter_ID"]].add(t)
    seen: dict[tuple, set] = defaultdict(set)
    for c, fs in by.items():
        for a in fs:
            for b in fs:
                if len(a) > len(b) >= 2 and a[:len(b)] == b and len(a) - len(b) <= max_len:
                    seen[a[len(b):]].add(c)
    return sorted((t for t, cs in seen.items()
                   if len(cs) >= min_concepts and len(t) >= 2 and t[0][0] in VOWEL_SEGS),
                  key=lambda t: (-len(seen[t]), t))


def segment(tokens: list[str], suffixes: list[tuple[str, ...]], min_stem: int = 2) -> list[str]:
    """Mark morpheme boundaries ('+'): ASJP word breaks, plus one known suffix per word."""
    words, cur = [], []
    for t in tokens + ["+"]:
        if t == "+":
            if cur:
                words.append(cur)
            cur = []
        else:
            cur.append(t)
    out = []
    for w in words:
        for suf in suffixes:
            if len(w) - len(suf) >= min_stem and tuple(w[-len(suf):]) == suf:
                w = w[:-len(suf)] + ["+"] + list(suf)
                break
        out += (["+"] if out else []) + w
    return out


def partial_cognates(forms: list[dict], threshold: float = 0.45, seed: int = 1,
                     suffixes: list | None = None, method: str = "sca") -> dict[str, str]:
    """Form ID -> cognate set ID via morpheme-level (partial) cognates.

    Morphemes are clustered within concepts (LingPy ``Partial``, UPGMA); each
    word then takes the partial cognate ID of its *salient* morpheme, the one
    whose partial set is largest within the concept (ties: the longest
    morpheme). Suffixes found by :func:`discover_suffixes` are segmented off
    first, so *eh* and *eh-al* can share a root.
    """
    from lingpy.compare.partial import Partial

    random.seed(seed)
    np.random.seed(seed)
    if suffixes is None:
        suffixes = discover_suffixes(forms)
    D = {0: ["doculect", "concept", "ipa", "tokens", "form_id"]}
    for i, f in enumerate(forms, start=1):
        D[i] = [f["Language_ID"], f["Parameter_ID"], f["Form"], segment(f["Segments"].split(), suffixes), f["ID"]]
    part = Partial(D, check=False)
    if method == "lexstat":
        part.get_partial_scorer(runs=1000)
    part.partial_cluster(method=method, threshold=threshold, ref="cogids", cluster_method="upgma")
    size: Counter = Counter()
    for i in part:
        for c in part[i, "cogids"]:
            size[(part[i, "concept"], c)] += 1
    out = {}
    for i in part:
        morphs = " ".join(part[i, "tokens"]).split(" + ")
        ids = part[i, "cogids"]
        # suffixes are never the root: only stems (non-final, or not a known suffix) are eligible
        sufs = {" ".join(t) for t in suffixes}
        ok = [j for j in range(len(ids)) if j == 0 or morphs[j] not in sufs] or [0]
        best = max(ok, key=lambda j: (size[(part[i, "concept"], ids[j])], len(morphs[j].split())))
        out[part[i, "form_id"]] = f"{part[i, 'concept']}-p{ids[best]}"
    return out


def build(asjp_cldf: Path | str, outdir: Path | str, threshold: float = 0.75, runs: int = 1000,
          min_core: int = 28, seed: int = 1, method: str = "lexstat") -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    langs, params, forms = select_mayan(asjp_cldf, min_core)
    n_loans = sum(f["Loan"] == "true" for f in forms)
    forms = [f for f in forms if f["Loan"] != "true" and f["Segments"].strip()]
    if method == "partial":
        cog = partial_cognates(forms, threshold, seed)
    else:
        cog = lexstat_cognates(forms, threshold, runs, seed, method)

    ds = Wordlist.in_dir(outdir)
    ds.add_component("LanguageTable")
    ds.add_component("ParameterTable")
    ds.add_component("CognateTable")
    ds.add_component("CognatesetTable")
    ds.properties["dc:title"] = "Mayan ASJP core vocabulary with LexStat cognates"
    ds.properties["dc:description"] = (
        f"Derived from lexibank/asjp v21.1: one doculect per Glottolog languoid, 40 core concepts, "
        f"ASJP-flagged loans removed, cognates by LingPy {method} (threshold {threshold}, {runs} runs, "
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
