"""Level-3 loan detector applied to the Mayan ASJP data.

Training: simulated datasets with the Mayan shape (31 languages, 40
concepts, milestone 1 root and loss-rate priors), Mayan-like phonology (CVC
roots, stable segments), kappa = 0.5, half uniform and half distance contact.
With ``--pipeline`` the simulated words go through the same LexStat cognate
detection as the real data (threshold 0.75), so the training data contain
the same kinds of coding errors (false merges, splits); an entry is labelled
borrowed if its true cognate class reached the language by borrowing.
Features are standardised within each dataset (simulated or real), so the
classifier sees relative, not absolute, feature values. Held-out AUC is
reported by 2-fold cross-validation over replicates.

Then every Mayan entry (language, LexStat cognate set) is scored. Output:
data/derived/m5_detector/{scores.csv, summary.json}.
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmm.pairhmm import pair_scores  # noqa: E402
from hmm.phylo import (FEATURES, feature_matrix, fit_logistic, patchiness_scores,  # noqa: E402
                       predict_logistic, set_centred_scores, transfer_scores, upgma_tree)
from infer.beastxml import draw_truth, load_prior  # noqa: E402
from sim.borrowing import BorrowingParams, simulate_with_borrowing  # noqa: E402
from sim.phonology import PhonologyParams  # noqa: E402
from sim.sdollo import SDolloParams  # noqa: E402

OUT = ROOT / "data/derived/m5_detector"
MAYAN = ROOT / "data/derived/asjp_mayan"
PHON = dict(min_syllables=1, max_syllables=1, p_final_consonant=1.0, stable_segments="aiuhmnk", stable_weight=0.05)
CONTACT = [dict(contact="uniform"), dict(contact="distance", distance_scale=1000.0)]
N_REPS = 20
KAPPA = 0.5


def features(wl: dict, concept_of: dict) -> tuple[list, np.ndarray]:
    form = set_centred_scores(pair_scores(wl))
    H = upgma_tree(wl, concept_of)
    pa, tr = patchiness_scores(wl, concept_of, list(H)), transfer_scores(wl, concept_of, H)
    patch = {e: {**pa[e], **tr[e]} for e in pa}
    entries = [e for e in form]
    X = feature_matrix(form, patch, entries, len(wl))
    return entries, (X - X.mean(0)) / (X.std(0) + 1e-9)


def lexstat_sets(wl_true: dict, concept_of: dict, threshold: float = 0.75) -> dict:
    """Re-code simulated cognates with LexStat: {(lang, true set): inferred set}."""
    from pipeline.asjp import lexstat_cognates
    forms, key = [], {}
    for l, sets in wl_true.items():
        for k, form in sets.items():
            fid = f"{l}|{k}"
            forms.append({"ID": fid, "Language_ID": l, "Parameter_ID": concept_of[k], "Form": form,
                          "Segments": " ".join(form)})
            key[fid] = (l, k)
    cog = lexstat_cognates(forms, threshold=threshold)
    return {key[fid]: c for fid, c in cog.items()}


def simulated(seed: int, prior: dict, contact: dict, pipeline: bool = False):
    rng = np.random.default_rng(seed)
    t = draw_truth(prior, rng)
    mu = t["loss_rate"]
    res = simulate_with_borrowing(
        t["tree"], SDolloParams(prior["n_concepts"], mu * prior["classes_per_concept"], mu),
        BorrowingParams(KAPPA * mu, **contact), seed=seed + 1, phonology=PhonologyParams(**PHON))
    keep = res.presence.any(axis=0)
    obs = res.observed()
    B = res.extra["borrowed"][:, keep]
    wl = {l: {str(j): obs.forms[i, j] for j in np.flatnonzero(obs.presence[i])} for i, l in enumerate(obs.leaf_names)}
    concept_of = {str(j): str(c) for j, c in enumerate(obs.class_concept)}
    li = {l: i for i, l in enumerate(obs.leaf_names)}
    if not pipeline:
        entries, X = features(wl, concept_of)
        return X, np.array([bool(B[li[l], int(k)]) for l, k in entries])
    inferred = lexstat_sets(wl, concept_of)
    wl2, c2, lab = {}, {}, {}
    for (l, k), c in inferred.items():
        wl2.setdefault(l, {})[c] = wl[l][k]
        c2[c] = concept_of[k]
        lab[(l, c)] = lab.get((l, c), False) or bool(B[li[l], int(k)])
    entries, X = features(wl2, c2)
    return X, np.array([lab[e] for e in entries])


def mayan():
    langs = {r["ID"]: r["Name"] for r in csv.DictReader(open(MAYAN / "languages.csv"))}
    pname = {r["ID"]: r["Name"] for r in csv.DictReader(open(MAYAN / "parameters.csv"))}
    cog = {r["Form_ID"]: r["Cognateset_ID"] for r in csv.DictReader(open(MAYAN / "cognates.csv"))}
    wl, concept_of, shown = {}, {}, {}
    for f in csv.DictReader(open(MAYAN / "forms.csv")):
        l, k = langs[f["Language_ID"]], cog[f["ID"]]
        concept_of[k] = pname[f["Parameter_ID"]]
        if k not in wl.setdefault(l, {}):  # synonyms in the same set: keep the first
            wl[l][k] = f["Segments"].split()
            shown[(l, k)] = f["Form"]
    return wl, concept_of, shown


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", action="store_true")
    a = ap.parse_args()
    global OUT
    if a.pipeline:
        OUT = OUT.with_name(OUT.name + "_pipeline")
    prior = copy.deepcopy(load_prior(ROOT / "infer/beast/m1_strict_prior.json"))
    prior["n_tips"], prior["n_concepts"] = 31, 40
    OUT.mkdir(parents=True, exist_ok=True)
    data = [simulated(7000 + 10 * i, prior, CONTACT[i % 2], a.pipeline) for i in range(N_REPS)]
    print(f"simulated {N_REPS} Mayan-shaped datasets, borrowed fraction "
          f"{np.mean([d[1].mean() for d in data]):.3f}", flush=True)

    # held-out evaluation (2-fold by replicate)
    folds = [list(range(0, N_REPS, 2)), list(range(1, N_REPS, 2))]
    pred, ys, l1 = [], [], []
    for tr, te in (folds, folds[::-1]):
        w = fit_logistic(np.vstack([data[i][0] for i in tr]), np.concatenate([data[i][1] for i in tr]).astype(float))
        for i in te:
            pred.append(predict_logistic(w, data[i][0])); ys.append(data[i][1]); l1.append(data[i][0][:, 0])
    from scripts.m3b_detector import auc
    P, Y, L1 = np.concatenate(pred), np.concatenate(ys), np.concatenate(l1)
    summary = {"heldout_auc_level3": auc(P, Y), "heldout_auc_level1": auc(L1, Y),
               "borrowed_fraction": float(Y.mean())}
    print(f"held-out AUC on Mayan-shaped simulations: level 1 {summary['heldout_auc_level1']:.3f}, "
          f"level 3 {summary['heldout_auc_level3']:.3f}", flush=True)

    w = fit_logistic(np.vstack([d[0] for d in data]), np.concatenate([d[1] for d in data]).astype(float))
    summary["weights"] = dict(zip(["intercept", *FEATURES], map(float, w)))
    wl, concept_of, shown = mayan()
    entries, X = features(wl, concept_of)
    p3 = predict_logistic(w, X)
    rows = sorted(({"language": l, "concept": concept_of[k], "cognate_set": k, "form": shown[(l, k)],
                    "p_loan": float(p), "level1_z": float(x[0]), "mrca_height_z": float(x[FEATURES.index("mrca_height")])}
                   for (l, k), p, x in zip(entries, p3, X)), key=lambda r: -r["p_loan"])
    for r in rows:
        r["rank_level3"] = rows.index(r) + 1
    by1 = sorted(rows, key=lambda r: -r["level1_z"])
    for i, r in enumerate(by1):
        r["rank_level1"] = i + 1
    with open(OUT / "scores.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    summary["n_entries"] = len(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nTop 25 Mayan entries by level 3 (of {len(rows)}):")
    for r in rows[:25]:
        print(f"  {r['rank_level3']:3d}  p={r['p_loan']:.2f}  {r['language'][:22]:22s} {r['concept']:10s} {r['form']:12s} "
              f"set {r['cognate_set']:8s} (level-1 rank {r['rank_level1']})")
    print("\nLevel-1 top 10 and where level 3 puts them:")
    for r in by1[:10]:
        print(f"  L1 {r['rank_level1']:3d} -> L3 {r['rank_level3']:4d}  {r['language'][:22]:22s} {r['concept']:10s} {r['form']}")


if __name__ == "__main__":
    main()
