"""How realistic are the simulated words? Real Mayan vs simulated, on the statistics that matter for cognate coding.

Real: the ASJP Mayan forms with Kaufman's etyma as gold (milestone 5c,
857 labelled forms in 27 languages; LexStat run on all their forms).
Simulated: Mayan-shaped datasets (31 languages, 40 concepts, milestone 1
priors) under a phonology configuration given as JSON overrides of
``PhonologyParams`` (and optionally ``kappa``).

Statistics (pairs are within a concept, across languages):

* word length, forms per (language, concept) cell, gold sets per concept;
* normalised edit distance (NED) of true-cognate pairs, overall and for
  pairs spanning the root split (real: Huastecan vs Core Mayan);
* NED of non-cognate pairs;
* LexStat 0.75 against gold: sets ratio (LexStat / gold), B-cubed P/R/F,
  pairwise recall of true-cognate pairs (overall and root-spanning), and
  the fraction of non-cognate pairs merged.

Extra config keys (not PhonologyParams): ``kappa`` (borrowing rate / loss
rate), ``tree: "pmed"`` (simulate on the PMED MCC tree of milestone 5b,
29 languages, instead of drawing Yule trees), ``mu`` (fixed loss rate),
``one_per_cell`` (keep one randomly chosen form per language and concept;
biases dating, see README), ``tip_noise`` / ``tip_compound`` (per observed
form, probability of one sporadic change / of a compound element added at
elicitation), ``process`` / ``p_synonym`` /
``synonym_loss_mult`` (lexical process, see ``SDolloParams``).

Usage: python scripts/sim_realism.py [--config '{"change_rate": 0.008}'] [--reps 4] [--real]
Results are appended to data/derived/sim_realism/results.jsonl.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import csv
import io
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from infer.beastxml import draw_truth, load_prior  # noqa: E402
from pipeline.asjp import lexstat_cognates  # noqa: E402
from sim.borrowing import BorrowingParams, simulate_with_borrowing  # noqa: E402
from sim.phonology import PhonologyParams  # noqa: E402
from sim.sdollo import SDolloParams  # noqa: E402

OUT = ROOT / "data/derived/sim_realism"
BASE_PHON = dict(min_syllables=1, max_syllables=1, p_final_consonant=1.0, stable_segments="aiuhmnk",
                 stable_weight=0.05)  # milestone 5/6 settings


def ned(a, b) -> float:
    if not a or not b:
        return 1.0
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1] / max(len(a), len(b))


def bcubed(gold: dict, pred: dict) -> tuple[float, float, float]:
    bt, bp = collections.defaultdict(set), collections.defaultdict(set)
    for k in gold:
        bt[gold[k]].add(k); bp[pred[k]].add(k)
    P = float(np.mean([len(bt[gold[k]] & bp[pred[k]]) / len(bp[pred[k]]) for k in gold]))
    R = float(np.mean([len(bt[gold[k]] & bp[pred[k]]) / len(bt[gold[k]]) for k in gold]))
    return P, R, 2 * P * R / (P + R)


def stats(forms: list[dict], gold: dict, auto: dict, deep) -> dict:
    """forms: dicts with ID, Language_ID, concept, seg (list). gold/auto: ID -> set. deep(l1, l2) -> bool."""
    lab = [f for f in forms if f["ID"] in gold]
    by = collections.defaultdict(list)
    for f in lab:
        by[f["concept"]].append(f)
    cog, cog_deep, non = [], [], []
    rec, rec_deep, fm = [], [], []
    for fs in by.values():
        for a, b in itertools.combinations(fs, 2):
            if a["Language_ID"] == b["Language_ID"]:
                continue
            d = ned(a["seg"], b["seg"])
            same_auto = auto[a["ID"]] == auto[b["ID"]]
            if gold[a["ID"]] == gold[b["ID"]]:
                cog.append(d); rec.append(same_auto)
                if deep(a["Language_ID"], b["Language_ID"]):
                    cog_deep.append(d); rec_deep.append(same_auto)
            else:
                non.append(d); fm.append(same_auto)
    cells = collections.Counter((f["Language_ID"], f["concept"]) for f in lab)
    n_gold = len({gold[f["ID"]] for f in lab})
    P, R, F = bcubed({f["ID"]: gold[f["ID"]] for f in lab}, auto)
    return {"n_forms": len(lab), "word_len": float(np.mean([len(f["seg"]) for f in lab])),
            "forms_per_cell": float(np.mean(list(cells.values()))),
            "gold_sets_per_concept": n_gold / len(by),
            "ned_cognate": float(np.mean(cog)), "ned_cognate_deep": float(np.mean(cog_deep)) if cog_deep else None,
            "ned_noncognate": float(np.mean(non)) if non else None,
            "sets_ratio": len({auto[f["ID"]] for f in lab}) / n_gold,
            "P": P, "R": R, "F": F, "pair_recall": float(np.mean(rec)),
            "pair_recall_deep": float(np.mean(rec_deep)) if rec_deep else None,
            "false_merge": float(np.mean(fm)) if fm else None}


def real() -> dict:
    from m5c_asjp_expert import MIN_LABELLED, labelled_forms
    with contextlib.redirect_stderr(io.StringIO()):
        languages, forms, concept, gold = labelled_forms()
        n = collections.Counter(f["Language_ID"] for f in forms if f["ID"] in gold)
        forms = [f for f in forms if n[f["Language_ID"]] >= MIN_LABELLED]
        auto = lexstat_cognates(forms, 0.75)
    cls = {r["ID"]: r["classification_glottolog"].split(",")
           for r in csv.DictReader(open(ROOT / "data/raw/asjp/cldf/languages.csv")) if r["Family"] == "Mayan"}
    fs = [{"ID": f["ID"], "Language_ID": f["Language_ID"], "concept": concept[f["ID"]],
           "seg": [s for s in f["Segments"].split() if s != "+"]} for f in forms]
    return stats(fs, gold, auto, lambda a, b: cls[a][1] != cls[b][1])


MAYAN_V1 = {"tree": "pmed", "one_per_cell": True, "min_syllables": 1, "max_syllables": 1, "segment_skew": 1.0,
            "suffix_rate": 1e-4, "compound_share": 0.5, "mu": 6e-5, "derived_prob": 0.3, "change_rate": 1e-3,
            "sporadic_rate": 1e-4}  # first calibration against the real ASJP/Kaufman statistics
# V1 biases dating through one_per_cell (see README). V2: replacement lexicon (about one word per
# concept, 10% synonymy) plus per-form transcription noise at the tips; no subsampling.
MAYAN_V2 = {"tree": "pmed", "min_syllables": 1, "max_syllables": 1, "segment_skew": 1.0, "compound_share": 0.5,
            "derived_prob": 0.3, "change_rate": 1e-3, "process": "replacement", "p_synonym": 0.1, "mu": 6e-5,
            "suffix_rate": 1e-4, "sporadic_rate": 1e-4, "tip_noise": 0.3}
# V2 splits sets into subgroup-sized pieces (lineage-level sporadic change); real LexStat errors mostly peel single
# languages off (singleton fragments, recall equal at all depths). V3 moves divergence from the lineages to the tips.
MAYAN_V3 = {**MAYAN_V2, "sporadic_rate": 3e-5, "suffix_rate": 5e-5, "tip_noise": 0.4, "tip_compound": 0.15}


def simulate_forms(cfg: dict, seed: int):
    """-> (forms, gold, tree, mu, borrowed_fraction); forms carry ID, Language_ID, Parameter_ID, concept, Form,
    Segments, seg."""
    cfg = dict(cfg)
    kappa = cfg.pop("kappa", 0.0)
    tree_src, mu_fix, one = cfg.pop("tree", None), cfg.pop("mu", None), cfg.pop("one_per_cell", False)
    lex = {k: cfg.pop(k) for k in ("process", "p_synonym", "synonym_loss_mult") if k in cfg}
    tip_noise, tip_compound = cfg.pop("tip_noise", 0.0), cfg.pop("tip_compound", 0.0)
    prior = copy.deepcopy(load_prior(ROOT / "infer/beast/m1_strict_prior.json"))
    prior["n_tips"], prior["n_concepts"] = 31, 40
    t = draw_truth(prior, np.random.default_rng(seed))
    if tree_src == "pmed":
        from sim.tree import Tree
        t["tree"] = Tree.from_newick((ROOT / "data/derived/m5b/mcc/pmed_yuc_ct.nwk").read_text())
    mu = mu_fix if mu_fix is not None else t["loss_rate"]
    res = simulate_with_borrowing(t["tree"], SDolloParams(40, mu, mu, **lex), BorrowingParams(kappa * mu),
                                  seed=seed + 1, phonology=PhonologyParams(**{**BASE_PHON, **cfg}))
    obs = res.observed()
    keep = res.presence.any(axis=0)
    B = res.extra["borrowed"][:, keep] if "borrowed" in res.extra else np.zeros_like(obs.presence, dtype=bool)
    forms, gold = [], {}
    for r, l in enumerate(obs.leaf_names):
        for j in np.flatnonzero(obs.presence[r]):
            fid = f"{l}|{j}"
            w = obs.forms[r, j]
            forms.append({"ID": fid, "Language_ID": l, "Parameter_ID": str(obs.class_concept[j]),
                          "concept": str(obs.class_concept[j]), "Form": w, "Segments": " ".join(w), "seg": list(w),
                          "borrowed": bool(B[r, j])})
            gold[fid] = str(j)
    if tip_noise > 0 or tip_compound > 0:  # observation noise: transcription, elicited compounds
        from sim.phonology import random_protoform, sporadic_change
        nrng = np.random.default_rng(seed + 3)
        pp = PhonologyParams(min_syllables=1, max_syllables=1, p_final_consonant=1.0,
                             segment_skew=cfg.get("segment_skew", 0.0))
        for f in forms:
            w = f["Form"]
            if nrng.random() < tip_noise:
                w = sporadic_change(w, nrng)
            if nrng.random() < tip_compound:
                r = random_protoform(pp, nrng)
                w = w + r if nrng.random() < 0.5 else r + w
            f["Form"], f["Segments"], f["seg"] = w, " ".join(w), list(w)
    if one:
        rng = np.random.default_rng(seed + 2)
        cell = collections.defaultdict(list)
        for f in forms:
            cell[(f["Language_ID"], f["concept"])].append(f)
        forms = [fs[int(rng.integers(len(fs)))] for fs in cell.values()]
        gold = {f["ID"]: gold[f["ID"]] for f in forms}
    bf = float(np.mean([f["borrowed"] for f in forms])) if forms else 0.0
    return forms, gold, t["tree"], mu, bf


def sim_one(args) -> dict:
    cfg, i = args
    t = {}
    forms, gold, t["tree"], mu, _ = simulate_forms(cfg, 9100 + 10 * i)
    with contextlib.redirect_stderr(io.StringIO()):
        auto = lexstat_cognates(forms, 0.75)
    root = t["tree"].root
    side = {}
    for k, ch in enumerate(root.children):
        stack = [ch]
        while stack:
            n = stack.pop()
            if n.is_leaf:
                side[n.name] = k
            stack.extend(n.children)
    return stats(forms, gold, auto, lambda a, b: side[a] != side[b])


KEYS = ["word_len", "forms_per_cell", "gold_sets_per_concept", "ned_cognate", "ned_cognate_deep", "ned_noncognate",
        "sets_ratio", "P", "R", "F", "pair_recall", "pair_recall_deep", "false_merge"]


def show(label: str, s: dict) -> None:
    print(f"{label:34s} " + " ".join(f"{(s[k] if s[k] is not None else float('nan')):6.2f}" for k in KEYS), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="{}")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'':34s} " + " ".join(f"{k[:6]:>6s}" for k in KEYS))
    if a.real:
        s = real()
        show("REAL (ASJP vs Kaufman)", s)
        with open(OUT / "results.jsonl", "a") as fh:
            fh.write(json.dumps({"label": "real", **s}) + "\n")
        return
    cfg = json.loads(a.config)
    with ProcessPoolExecutor(min(a.reps, 6)) as pool:
        rows = list(pool.map(sim_one, [(cfg, i) for i in range(a.reps)]))
    s = {k: (float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None)
         for k in rows[0]}
    show(a.label or json.dumps(cfg)[:34], s)
    with open(OUT / "results.jsonl", "a") as fh:
        fh.write(json.dumps({"label": a.label, "config": cfg, "reps": a.reps, **s}) + "\n")


if __name__ == "__main__":
    main()
