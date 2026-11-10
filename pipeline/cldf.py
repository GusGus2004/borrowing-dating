"""CLDF Wordlist I/O shared by simulated and real data.

* ``write_simulation`` turns an SDollo simulation into a valid CLDF Wordlist
  (languages, parameters, forms, cognates, cognatesets) plus ground-truth
  sidecar files (``truth.json``, ``tree.nwk``).
* ``read_cognate_matrix`` reads *any* CLDF Wordlist with cognate judgements —
  simulated or Lexibank — into the same ``CognateMatrix`` structure, so that
  everything downstream (BEAST2 XML, Track B) sees one format.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from pycldf import Wordlist

MISSING = -1


# ---------------------------------------------------------------------- matrix


@dataclass
class CognateMatrix:
    """Languages x cognate sets, coded 1 / 0 / MISSING (-1).

    ``concepts[j]`` is the concept (Parameter_ID) of column j. Missing data is
    assigned at the concept level: if a language has no form for a concept,
    all that concept's columns are MISSING for it.
    """

    languages: list[str]
    cognatesets: list[str]
    concepts: list[str]
    matrix: np.ndarray  # int8 (n_languages, n_cognatesets)

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape  # type: ignore[return-value]

    def to_nexus(self, path: Path | str) -> None:
        """Binary NEXUS (0/1/?) with one character per cognate set."""
        sym = {1: "1", 0: "0", MISSING: "?"}
        width = max(len(l) for l in self.languages) + 2
        lines = [
            "#NEXUS",
            "BEGIN DATA;",
            f"  DIMENSIONS NTAX={len(self.languages)} NCHAR={len(self.cognatesets)};",
            '  FORMAT DATATYPE=STANDARD SYMBOLS="01" MISSING=? GAP=-;',
            "  MATRIX",
        ]
        for lang, row in zip(self.languages, self.matrix):
            lines.append("    " + lang.ljust(width) + "".join(sym[int(v)] for v in row))
        lines += ["  ;", "END;", ""]
        Path(path).write_text("\n".join(lines))

    def to_csv(self, path: Path | str) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Language_ID"] + self.cognatesets)
            for lang, row in zip(self.languages, self.matrix):
                w.writerow([lang] + [("?" if v == MISSING else str(int(v))) for v in row])


def read_cognate_matrix(
    metadata: Path | str,
    languages: Optional[Iterable[str]] = None,
    exclude_proto: bool = True,
    concept_level_missing: bool = True,
) -> CognateMatrix:
    """Build a cognate matrix from a CLDF Wordlist.

    Parameters
    ----------
    metadata : path to the dataset's ``*-metadata.json``.
    languages : optional subset of Language_IDs to keep (order preserved).
    exclude_proto : drop languages flagged ``Is_Proto`` (Lexibank convention).
    concept_level_missing : if True, a language with no form for a concept is
        coded MISSING for every cognate set of that concept. If False (useful
        for etymological dictionaries where "concepts" are language-specific
        glosses), a cognate set is MISSING never and absence is coded 0.

    A cognate set that spans several concepts is assigned to the concept of
    the majority of its forms (ties: lexicographically smallest).
    """
    ds = Wordlist.from_metadata(metadata)

    lang_rows = list(ds.iter_rows("LanguageTable", "id"))
    lang_ids = [r["id"] for r in lang_rows]
    if exclude_proto:
        lang_ids = [r["id"] for r in lang_rows if not _truthy(r.get("Is_Proto"))]
    if languages is not None:
        wanted = list(languages)
        unknown = set(wanted) - set(lang_ids)
        if unknown:
            raise KeyError(f"unknown or excluded languages: {sorted(unknown)}")
        lang_ids = wanted
    lang_pos = {l: i for i, l in enumerate(lang_ids)}

    form_lang: dict[str, str] = {}
    form_concept: dict[str, str] = {}
    has_concept: dict[str, set[str]] = defaultdict(set)  # language -> concepts
    for r in ds.iter_rows("FormTable", "id", "languageReference", "parameterReference"):
        if r["languageReference"] not in lang_pos:
            continue
        form_lang[r["id"]] = r["languageReference"]
        form_concept[r["id"]] = r["parameterReference"]
        has_concept[r["languageReference"]].add(r["parameterReference"])

    members: dict[str, set[str]] = defaultdict(set)          # cogset -> languages
    cs_concepts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if "CognateTable" in ds:
        rows = ds.iter_rows("CognateTable", "formReference", "cognatesetReference")
        pairs = ((r["formReference"], r["cognatesetReference"]) for r in rows)
    else:  # cognate set id stored directly on forms
        rows = ds.iter_rows("FormTable", "id", "cognatesetReference")
        pairs = ((r["id"], r["cognatesetReference"]) for r in rows)
    for fid, cs in pairs:
        if cs in (None, "") or fid not in form_lang:
            continue
        cs = str(cs)
        members[cs].add(form_lang[fid])
        cs_concepts[cs][form_concept[fid]] += 1

    cogsets = sorted(members, key=_natural_key)
    concepts = [min(cs_concepts[c].items(), key=lambda kv: (-kv[1], kv[0]))[0] for c in cogsets]

    m = np.zeros((len(lang_ids), len(cogsets)), dtype=np.int8)
    for j, cs in enumerate(cogsets):
        for l in members[cs]:
            m[lang_pos[l], j] = 1
    if concept_level_missing:
        for i, l in enumerate(lang_ids):
            for j, c in enumerate(concepts):
                if c not in has_concept[l]:
                    m[i, j] = MISSING
    return CognateMatrix(languages=lang_ids, cognatesets=cogsets, concepts=concepts, matrix=m)


# ---------------------------------------------------------------------- writing


def write_simulation(result, outdir: Path | str, dataset_id: str = "sim") -> Path:
    """Write an SDollo simulation as a CLDF Wordlist plus ground truth.

    Only observable classes (present in at least one language) are written.
    Forms are placeholders (``<concept>_<class>``) until phoneme strings are
    simulated (milestone 3); the cognate structure is what matters here.

    Returns the path to the metadata JSON.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    obs = result.observed()

    ds = Wordlist.in_dir(outdir)
    ds.add_component("LanguageTable")
    ds.add_component("ParameterTable")
    ds.add_component("CognateTable")
    ds.add_component("CognatesetTable")
    ds.properties["dc:title"] = f"Simulated SDollo wordlist ({dataset_id})"
    ds.properties["dc:description"] = (
        "Forward simulation under stochastic Dollo; see truth.json for the generating "
        "tree, node ages, branch and concept rates."
    )

    languages = [{"ID": l, "Name": l} for l in obs.leaf_names]
    concept_ids = [f"c{c + 1:04d}" for c in range(obs.params.n_concepts)]
    parameters = [{"ID": cid, "Name": cid} for cid in concept_ids]

    cs_ids = [f"cs{j + 1:06d}" for j in range(obs.n_classes)]
    cognatesets = [
        {"ID": cs_ids[j], "Description": f"born on branch above {obs.class_birth_node[j]} "
                                          f"at age {obs.class_birth_age[j]:.6g}"}
        for j in range(obs.n_classes)
    ]

    forms, cognates = [], []
    for i, lang in enumerate(obs.leaf_names):
        for j in np.flatnonzero(obs.presence[i]):
            concept = concept_ids[obs.class_concept[j]]
            fid = f"{lang}-{cs_ids[j]}"
            value = f"{concept}_{cs_ids[j]}"
            forms.append({"ID": fid, "Language_ID": lang, "Parameter_ID": concept,
                          "Value": value, "Form": value})
            cognates.append({"ID": f"{fid}-cog", "Form_ID": fid, "Cognateset_ID": cs_ids[j]})

    ds.write(
        LanguageTable=languages,
        ParameterTable=parameters,
        FormTable=forms,
        CognateTable=cognates,
        CognatesetTable=cognatesets,
    )
    (outdir / "truth.json").write_text(json.dumps(obs.truth(), indent=2))
    (outdir / "tree.nwk").write_text(obs.tree.to_newick(precision=10) + "\n")
    return outdir / "Wordlist-metadata.json"


# ---------------------------------------------------------------------- helpers


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes"}


def _natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
