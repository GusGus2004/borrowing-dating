"""Borrowing correction by masking: flagged loans become missing data.

A loan replaced the recipient's inherited word for the concept, so the
recipient's inherited state is unknown for *every* cognate set of that concept.
Masking therefore codes the language MISSING for all sets of the concept (the
usual practice when linguists exclude known loans), not just 0 for the loan.
Sets left with no present ("1") entry are dropped, since SDollo conditions on
a set being observed somewhere.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from pipeline.cldf import MISSING, CognateMatrix


def mask_entries(cm: CognateMatrix, entries: Iterable[tuple[str, str]]) -> tuple[CognateMatrix, dict]:
    """Mask (language, cognate set) entries; returns the new matrix and stats."""
    m = cm.matrix.copy()
    li = {l: i for i, l in enumerate(cm.languages)}
    ci = {c: j for j, c in enumerate(cm.cognatesets)}
    concept = np.array(cm.concepts)
    masked_lc = set()
    for lang, cs in entries:
        if lang in li and cs in ci:
            masked_lc.add((lang, cm.concepts[ci[cs]]))
    for lang, c in masked_lc:
        m[li[lang], concept == c] = MISSING
    keep = (m == 1).any(axis=0)
    out = CognateMatrix(languages=list(cm.languages),
                        cognatesets=[c for c, k in zip(cm.cognatesets, keep) if k],
                        concepts=[c for c, k in zip(cm.concepts, keep) if k],
                        matrix=m[:, keep])
    n_lc = len(cm.languages) * len(set(cm.concepts))
    return out, {"masked_language_concepts": len(masked_lc),
                 "masked_fraction": len(masked_lc) / n_lc,
                 "dropped_cognatesets": int((~keep).sum())}


def read_wordlist(dataset_dir: Path | str) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Simulated CLDF -> (wordlist[lang][cogset] = form, concept_of[cogset])."""
    d = Path(dataset_dir)
    forms = {r["ID"]: r for r in csv.DictReader(open(d / "forms.csv"))}
    wl: dict[str, dict[str, str]] = {}
    concept_of: dict[str, str] = {}
    for r in csv.DictReader(open(d / "cognates.csv")):
        f = forms[r["Form_ID"]]
        wl.setdefault(f["Language_ID"], {})[r["Cognateset_ID"]] = f["Form"]
        concept_of[r["Cognateset_ID"]] = f["Parameter_ID"]
    return wl, concept_of


def read_borrowings(dataset_dir: Path | str) -> list[tuple[str, str]]:
    p = Path(dataset_dir) / "borrowings.csv"
    return [(r["Language_ID"], r["Cognateset_ID"]) for r in csv.DictReader(open(p))]
