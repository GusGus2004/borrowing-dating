"""Swadesh-100 Mayan wordlist from Kaufman & Justeson's PMED (lexibank kaufmanpmed).

PMED is organised by etymon: every form is a reflex of a reconstructed
Proto-Mayan (or subgroup) etymon and carries the gloss its source gave it,
as "Spanish//English". This module turns it into a meaning-based wordlist:

* **Glosses** are split into senses (at ``//``, ``;`` and ``,``; text in
  parentheses dropped; ``n*`` read as ñ and other ``*`` accent marks removed;
  leading "to"/articles dropped). A form belongs to a concept when one of its
  senses *equals* one of the concept's English or Spanish glosses in
  ``pmed_swadesh100.tsv``. Exact matching keeps "pecho de mujer" out of
  "woman" and "chile diente de perro" out of "tooth" and "dog".
* **Varieties**: modern varieties only (no proto-languages, epigraphic or
  colonial sources, no group labels without a Glottocode), one per Glottocode:
  the one covering most concepts (ties: most forms).
* **Cognates** are Kaufman's etyma (``cognates.csv``). A matched form with no
  etymon gets its own set.

Caveat: forms that are not reflexes of a reconstructed etymon may be absent
from PMED, so a missing concept can mean "innovation not recorded" as well
as "no data". Missing cells are coded as missing, never as absence.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from pycldf import Wordlist

CONCEPTS_TSV = Path(__file__).with_name("pmed_swadesh100.tsv")
EXCLUDE_IDS = {"EpM", "EpMCh", "EpMYu", "CHT", "ColYUK", "KAQcol"}  # epigraphic / colonial


def norm(s: str) -> str:
    s = s.lower().replace("n*", "ñ").replace("*", "")
    s = unicodedata.normalize("NFD", s.replace("ñ", "\0"))
    s = "".join(c for c in s if not unicodedata.combining(c)).replace("\0", "ñ")
    return s


def senses(description: str) -> list[str]:
    out = []
    for part in description.split("//"):
        part = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", norm(part))
        for s in re.split(r"[;,]", part):
            s = s.strip(" .\"'?!:")
            s = re.sub(r"^(to|el|la|los|las|un|una|vt|vi|adj)\s+", "", s).strip()
            if s:
                out.append(s)
    return out


def load_concepts(path: Path = CONCEPTS_TSV) -> dict[str, set[str]]:
    return {c: v[0] for c, v in load_concept_table(path).items()}


def load_concept_table(path: Path = CONCEPTS_TSV) -> dict[str, tuple[set[str], set[str]]]:
    """concept -> (gloss patterns, excluding senses)."""
    out = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        pats = {norm(p.strip()) for col in ("english", "spanish") for p in r[col].split("|") if p.strip()}
        excl = {norm(p.strip()) for p in (r.get("exclude") or "").split("|") if p.strip()}
        out[r["concept"]] = (pats, excl)
    return out


ENCLITIC = re.compile(r"(selo|sela|les|los|las|lo|la|le|se)$")


def sense_keys(s: str) -> list[str]:
    """The sense itself, plus the bare infinitive for Spanish verbs with enclitics (morderlo -> morder)."""
    out = [s]
    m = ENCLITIC.search(s)
    if m and re.search(r"[aei]r$", s[:m.start()]):
        out.append(s[:m.start()])
    return out


def match(pmed_cldf: Path | str, concepts: dict[str, set[str]] | None = None):
    """Return (languages, matches) where matches = [(form row, concept, etymon or None)]."""
    d = Path(pmed_cldf)
    table = load_concept_table()
    concepts = concepts or {c: v[0] for c, v in table.items()}
    langs = {r["ID"]: r for r in csv.DictReader(open(d / "languages.csv"))}
    cog = {r["Form_ID"]: r["Cognateset_ID"] for r in csv.DictReader(open(d / "cognates.csv"))}
    sense_to_concepts: dict[str, list[str]] = defaultdict(list)
    for c, pats in concepts.items():
        for p in pats:
            sense_to_concepts[p].append(c)
    modern = {l for l, r in langs.items() if r["Glottocode"] and r["Is_Proto"] != "true"
              and not l.startswith("p") and l not in EXCLUDE_IDS}
    matches = []
    for f in csv.DictReader(open(d / "forms.csv")):
        if f["Language_ID"] not in modern or not f["Form"].strip():
            continue
        ss = senses(f["Description"])
        hit = {c for s in ss for k in sense_keys(s) for c in sense_to_concepts.get(k, [])}
        hit = {c for c in hit if not (set(ss) & table.get(c, (set(), set()))[1])}
        for c in sorted(hit):
            matches.append((f, c, cog.get(f["ID"])))
    return langs, matches


def build(pmed_cldf: Path | str, outdir: Path | str, min_concepts: int = 30) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    concepts = load_concepts()
    langs, matches = match(pmed_cldf, concepts)
    cover = defaultdict(set)
    nforms = Counter()
    for f, c, _ in matches:
        cover[f["Language_ID"]].add(c)
        nforms[f["Language_ID"]] += 1
    best: dict[str, str] = {}
    for l in sorted(cover):
        g = langs[l]["Glottocode"]
        key = (len(cover[l]), nforms[l])
        if g not in best or key > (len(cover[best[g]]), nforms[best[g]]):
            best[g] = l
    keep = sorted(l for l in best.values() if len(cover[l]) >= min_concepts)
    dropped = sorted((langs[l]["Name"], len(cover[l])) for l in best.values() if l not in keep)
    rows = [(f, c, e) for f, c, e in matches if f["Language_ID"] in keep]

    ds = Wordlist.in_dir(outdir)
    for comp in ("LanguageTable", "ParameterTable", "CognateTable", "CognatesetTable"):
        ds.add_component(comp)
    ds.properties["dc:title"] = "Mayan Swadesh-100 wordlist from PMED (Kaufman & Justeson) with expert etyma"
    ds.properties["dc:description"] = (
        "Derived from lexibank/kaufmanpmed: glosses matched exactly to a Swadesh-100 concept list "
        "(pipeline/pmed_swadesh100.tsv), one modern variety per Glottocode, cognate sets = Kaufman's etyma.")
    forms, cogs, sets = [], [], set()
    seen = set()
    for f, c, e in rows:
        fid = f"{f['ID']}-{c}"
        if fid in seen:
            continue
        seen.add(fid)
        cs = f"{c}-{e}" if e else f"{c}-u{f['ID']}"
        segs = f["Segments"].split() if f["Segments"].strip() else list(f["Form"].replace(" ", "_"))
        forms.append({"ID": fid, "Language_ID": f["Language_ID"], "Parameter_ID": c,
                      "Value": f["Value"], "Form": f["Form"], "Segments": segs,
                      "Comment": f["Description"]})
        cogs.append({"ID": f"{fid}-cog", "Form_ID": fid, "Cognateset_ID": cs})
        sets.add(cs)
    ds.write(
        LanguageTable=[{"ID": l, "Name": langs[l]["Name"], "Glottocode": langs[l]["Glottocode"],
                        "Latitude": langs[l]["Latitude"] or None, "Longitude": langs[l]["Longitude"] or None}
                       for l in keep],
        ParameterTable=[{"ID": c, "Name": c} for c in concepts],
        FormTable=forms, CognateTable=cogs, CognatesetTable=[{"ID": s} for s in sorted(sets)],
    )
    per_concept = Counter()
    for l in keep:
        for c in cover[l]:
            per_concept[c] += 1
    return {"n_languages": len(keep), "n_forms": len(forms), "n_cognatesets": len(sets),
            "n_unlinked_forms": sum(1 for _, _, e in rows if not e),
            "dropped_languages": dropped,
            "languages": {langs[l]["Name"]: len(cover[l]) for l in keep},
            "concept_coverage": {c: per_concept.get(c, 0) for c in concepts}}
