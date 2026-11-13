"""Score automated cognate detection on the real ASJP Mayan data against Kaufman's etyma.

Gold labels: for each ASJP form (language, concept) whose language and
concept also occur in the PMED Swadesh-100 wordlist (``pipeline.pmed``), find
the PMED form of the same language and concept that is most similar in
spelling (both converted to a coarse ASJP-like code). If the normalised edit
distance is at most ``--max-ned``, the ASJP form takes that PMED form's
etymon as its true cognate set. Every method clusters the full ASJP dataset;
B-cubed precision, recall and F are computed over the labelled forms.

Output: data/derived/cognate_benchmark/results.json and a table.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.asjp import lexstat_cognates, partial_cognates, select_mayan  # noqa: E402
from pipeline.pmed import match  # noqa: E402

OUT = ROOT / "data/derived/cognate_benchmark"
PMED = ROOT / "data/raw/kaufmanpmed/cldf"
ASJP = ROOT / "data/raw/asjp/cldf"
# ASJP 40-item names -> Swadesh-100 concept names used in pipeline/pmed_swadesh100.tsv
ASJP2SW = {"I": "I", "you": "you", "we": "we", "one": "one", "two": "two", "person": "person", "fish": "fish",
           "dog": "dog", "louse": "louse", "tree": "tree", "leaf": "leaf", "skin": "skin", "blood": "blood",
           "bone": "bone", "horn": "horn", "ear": "ear", "eye": "eye", "nose": "nose", "tooth": "tooth",
           "tongue": "tongue", "knee": "knee", "hand": "hand", "breasts": "breast", "liver": "liver",
           "drink": "drink", "see": "see", "hear": "hear", "die": "die", "come": "come", "sun": "sun",
           "star": "star", "water": "water", "stone": "stone", "fire": "fire", "path": "path",
           "mountain": "mountain", "night": "night", "full": "full", "new": "new", "name": "name"}


def coarse_kaufman(s: str) -> str:
    """Kaufman's practical orthography -> coarse ASJP-like code (no ejectives, length, morpheme marks)."""
    s = s.lower().split("~")[0].split(" ")[0] if " " not in s.strip() else s.lower()
    s = re.sub(r"\([^)]*\)|\[[^\]]*\]|[#*=.:\-!\"]|erg|abs", "", s)
    for a, b in (("ch'", "C"), ("tz'", "c"), ("tx'", "C"), ("ch", "C"), ("tz", "c"), ("tx", "C"),
                 ("b'", "b"), ("'", ""), ("x", "S"), ("j", "x"), ("@", "3"), ("nh", "N"), ("y", "y")):
        s = s.replace(a, b)
    s = re.sub(r"([aeiou3])\1+", r"\1", s)
    return re.sub(r"\s+", "", s)


def coarse_asjp(s: str) -> str:
    s = re.sub(r'[\"*~$]', "", s)
    s = s.replace("E", "e").replace("3", "3").replace("8", "t").replace("4", "n")
    return re.sub(r"([aeiou3])\1+", r"\1", s.replace(" ", ""))


def ned(a: str, b: str) -> float:
    if not a or not b:
        return 1.0
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1] / max(len(a), len(b))


def bcubed(gold: dict, pred: dict) -> tuple[float, float, float]:
    keys = [k for k in gold if k in pred]
    bt, bp = collections.defaultdict(set), collections.defaultdict(set)
    for k in keys:
        bt[gold[k]].add(k); bp[pred[k]].add(k)
    P = np.mean([len(bt[gold[k]] & bp[pred[k]]) / len(bp[pred[k]]) for k in keys])
    R = np.mean([len(bt[gold[k]] & bp[pred[k]]) / len(bt[gold[k]]) for k in keys])
    return float(P), float(R), float(2 * P * R / (P + R))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ned", type=float, default=0.5)
    ap.add_argument("--mode", choices=["spelling", "unambiguous"], default="unambiguous",
                    help="spelling: link to the most similar PMED form (NED <= max-ned); unambiguous: only "
                         "cells where PMED has exactly one etymon, no spelling comparison (avoids favouring "
                         "spelling-based methods)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    langs, params, forms = select_mayan(ASJP)
    forms = [f for f in forms if f["Loan"] != "true" and f["Segments"].strip()]
    glott = {l["ID"]: l["Glottocode"] for l in langs}
    pname = {p["ID"]: p["Name"].lstrip("*") for p in params}

    plangs, pm = match(PMED)
    pmed = collections.defaultdict(list)  # (glottocode, concept) -> [(coarse form, etymon)]
    for f, c, e in pm:
        if e:
            pmed[(plangs[f["Language_ID"]]["Glottocode"], c)].append((coarse_kaufman(f["Form"]), e))

    gold, examples = {}, []
    n_overlap = 0
    for f in forms:
        key = (glott[f["Language_ID"]], ASJP2SW.get(pname[f["Parameter_ID"]]))
        if key not in pmed:
            continue
        n_overlap += 1
        if a.mode == "unambiguous":
            ets = {e for _, e in pmed[key]}
            if len(ets) == 1:
                gold[f["ID"]] = f"{key[1]}-{ets.pop()}"
            continue
        x = coarse_asjp(f["Form"])
        d, (pf, e) = min(((ned(x, pf), (pf, e)) for pf, e in pmed[key]), key=lambda t: t[0])
        if d <= a.max_ned:
            gold[f["ID"]] = f"{key[1]}-{e}"
            if len(examples) < 12:
                examples.append((f["Form"], pf, e, round(d, 2)))
    print(f"ASJP forms: {len(forms)}; with a PMED counterpart (language+concept): {n_overlap}; "
          f"labelled ({a.mode}): {len(gold)}; gold sets: {len(set(gold.values()))}")
    print("examples (ASJP, PMED coarse, etymon, NED):", examples)

    methods = {"LexStat 0.55": lambda: lexstat_cognates(forms, 0.55),
               "LexStat 0.65": lambda: lexstat_cognates(forms, 0.65),
               "LexStat 0.75 (current)": lambda: lexstat_cognates(forms, 0.75),
               "LexStat 0.85": lambda: lexstat_cognates(forms, 0.85),
               "SCA 0.45": lambda: lexstat_cognates(forms, 0.45, method="sca"),
               "SCA 0.55": lambda: lexstat_cognates(forms, 0.55, method="sca"),
               "edit distance 0.75": lambda: lexstat_cognates(forms, 0.75, method="edit-dist"),
               "partial SCA 0.45": lambda: partial_cognates(forms, 0.45),
               "partial LexStat 0.75": lambda: partial_cognates(forms, 0.75, method="lexstat"),
               "partial LexStat 0.85": lambda: partial_cognates(forms, 0.85, method="lexstat")}
    res = {}
    print(f"\n{'method':24s} {'P':>6s} {'R':>6s} {'F':>6s}")
    for name, fn in methods.items():
        P, R, F = bcubed(gold, fn())
        res[name] = {"P": P, "R": R, "F": F}
        print(f"{name:24s} {P:6.3f} {R:6.3f} {F:6.3f}", flush=True)
    (OUT / f"results_{a.mode}.json").write_text(json.dumps({"n_labelled": len(gold), "max_ned": a.max_ned,
                                                    "methods": res}, indent=2))


if __name__ == "__main__":
    main()
