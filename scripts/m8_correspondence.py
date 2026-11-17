"""Milestone 8: proof of concept for correspondence-based cognate detection.

Similarity thresholds fail in opposite directions on Mayan (over-splitting) and IE (over-merging),
and no threshold is right for both (5d, 7). The comparative method decides cognacy instead by
whether words fit REGULAR sound correspondences between two languages, which is evidence a
threshold does not use. This is a minimal version of that logic:

  1. seeds   high-precision LexStat sets (threshold 0.45; P about 0.95 in both families)
  2. learn   for every language pair, which sound classes correspond, from aligned seed cognates
             (two rounds of align-and-count; sparse pairs shrunk towards family-wide counts)
  3. score   every cross-language word pair of a concept by its best alignment under that pair's
             learned log-odds, as a z-score against word pairs of DIFFERENT concepts from the
             same two languages (a per-pair null, as in the comparative method's chance baseline)
  4. merge   seed sets of one concept, by average linkage, while the mean z between two sets
             is at least Z_MERGE

Z_MERGE = 2.0 was fixed before any result was seen and is the same for both families: the point
is a criterion that needs no tuning per family. Other values are reported only as sensitivity.

Subcommands: detect [--family mayan|ie|both] | prepare | run [--jobs N] | summarise
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import io
import itertools
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

OUT = ROOT / "data/derived/m8"
Z_MERGE = 2.0
SEED_THRESHOLD = 0.45
SHRINK = 30.0  # pseudo-counts towards the family-wide correspondence table
GAP, END_GAP = -2.0, float(os.environ.get("M8_END_GAP", "-1.0"))  # END_GAP 0 (--edge-free): a root inside a longer word aligns free of affixes
NULL_PAIRS = 200  # random cross-concept word pairs per language pair
SENSITIVITY = (1.0, 1.5, 2.0, 3.0, 4.0)


# ---------------------------------------------------------------- data
def family_data(family: str):
    """-> (taxa, forms, concept, gold, prior, gold_is_partial). Forms: ID, Language_ID, Parameter_ID, Form, Segments."""
    with contextlib.redirect_stderr(io.StringIO()):
        if family == "mayan":
            from m5c_asjp_expert import BASE, CT, MIN_LABELLED, TAXA, YUC, labelled_forms
            languages, forms, concept, gold = labelled_forms()
            n_lab = collections.Counter(f["Language_ID"] for f in forms if f["ID"] in gold)
            taxa = [l for l in languages if n_lab[l] >= MIN_LABELLED]
            forms = [f for f in forms if f["Language_ID"] in taxa]
            prior = copy.deepcopy(BASE)
            prior["calibrations"] = []
            for c, name in ((YUC, "Yucatecan"), (CT, "CholanTzeltalan")):
                c = copy.deepcopy(c)
                c["taxa"] = [t for t in TAXA[name] if t in taxa]
                prior["calibrations"].append(c)
            return taxa, forms, concept, gold, prior, True
        from infer.beastxml import load_prior
        from m7_iecor import PRIOR, load
        taxa, forms, concept, gold, _ = load()
        prior = load_prior(PRIOR)
        prior["mcmc"] = dict(prior["mcmc"], chain_length=3_000_000)
        return taxa, forms, concept, gold, prior, False


def classes(segments: str) -> tuple[str, ...]:
    from lingpy.sequence.sound_classes import tokens2class
    toks = [t for t in segments.split() if t not in ("+", "_")]
    return tuple(tokens2class(toks, "sca")) if toks else ()


# ---------------------------------------------------------------- alignment
def align(a: tuple, b: tuple, score) -> tuple[float, list]:
    """Global alignment, cheaper end gaps. -> (score, aligned pairs incl. gaps as '-')."""
    n, m = len(a), len(b)
    F = np.zeros((n + 1, m + 1))
    T = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        F[i, 0], T[i, 0] = F[i - 1, 0] + END_GAP, 1
    for j in range(1, m + 1):
        F[0, j], T[0, j] = F[0, j - 1] + END_GAP, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            gi = END_GAP if j == m else GAP
            gj = END_GAP if i == n else GAP
            d, u, l = F[i - 1, j - 1] + score(a[i - 1], b[j - 1]), F[i - 1, j] + gi, F[i, j - 1] + gj
            if d >= u and d >= l:
                F[i, j], T[i, j] = d, 0
            elif u >= l:
                F[i, j], T[i, j] = u, 1
            else:
                F[i, j], T[i, j] = l, 2
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        t = T[i, j]
        if i > 0 and j > 0 and t == 0:
            pairs.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i > 0 and (j == 0 or t == 1):
            pairs.append((a[i - 1], "-")); i -= 1
        else:
            pairs.append(("-", b[j - 1])); j -= 1
    return float(F[n, m]), pairs[::-1]


def identity_score(x, y):
    return 2.0 if x == y else -1.0


class Correspondences:
    """Per-language-pair log-odds of sound-class correspondences, shrunk towards the family-wide table."""

    def __init__(self, pair_counts, all_counts, marg):
        self.pc, self.ac, self.marg = pair_counts, all_counts, marg
        self.atot = sum(all_counts.values())
        self.mtot = {l: sum(c.values()) for l, c in marg.items()}
        self.ptot = {k: sum(c.values()) for k, c in pair_counts.items()}

    def scorer(self, la: str, lb: str):
        key = (la, lb) if la <= lb else (lb, la)
        flip = la > lb
        pc, pt = self.pc.get(key, {}), self.ptot.get(key, 0)
        ma, mb = self.marg[la], self.marg[lb]
        na, nb = self.mtot[la] + 30, self.mtot[lb] + 30

        def s(x, y):
            k = (y, x) if flip else (x, y)
            p_all = (self.ac.get(k, 0) + 0.1) / (self.atot + 0.1 * 900)
            p = (pc.get(k, 0) + SHRINK * p_all) / (pt + SHRINK)
            q = ((ma.get(x, 0) + 1) / na) * ((mb.get(y, 0) + 1) / nb)
            return math.log(p / q)
        return s


def learn(words, seeds, langs, rounds: int = 2) -> Correspondences:
    """words: ID -> (lang, concept, classes). seeds: ID -> set id."""
    marg = {l: collections.Counter() for l in langs}
    for lang, _, cl in words.values():
        marg[lang].update(cl)
    members = collections.defaultdict(list)
    for i, s in seeds.items():
        members[s].append(i)
    cog_pairs = [(a, b) for ids in members.values() for a, b in itertools.combinations(sorted(ids), 2)
                 if words[a][0] != words[b][0]]
    corr = None
    for _ in range(rounds):
        pc, ac = collections.defaultdict(collections.Counter), collections.Counter()
        for a, b in cog_pairs:
            la, lb = words[a][0], words[b][0]
            if la > lb:
                a, b, la, lb = b, a, lb, la
            sc = identity_score if corr is None else corr.scorer(la, lb)
            _, al = align(words[a][2], words[b][2], sc)
            pc[(la, lb)].update(al)
            ac.update(al)
        corr = Correspondences(pc, ac, marg)
    return corr


# ---------------------------------------------------------------- detection
def _pair_scores(args):
    corr, jobs = args
    out = []
    for key, a, b, wa, wb, la, lb in jobs:
        sc, _ = align(wa, wb, corr.scorer(la, lb))
        out.append((key, a, b, sc / max(1.0, (len(wa) + len(wb)) / 2)))
    return out


def detect(family: str, workers: int = 6) -> dict:
    from pipeline.asjp import lexstat_cognates
    taxa, forms, concept, gold, _, _ = family_data(family)
    with contextlib.redirect_stderr(io.StringIO()):
        seeds = lexstat_cognates(forms, SEED_THRESHOLD)
    words = {f["ID"]: (f["Language_ID"], concept[f["ID"]], classes(f["Segments"])) for f in forms}
    words = {i: w for i, w in words.items() if w[2]}
    seeds = {i: s for i, s in seeds.items() if i in words}
    corr = learn(words, seeds, taxa)
    by_c = collections.defaultdict(list)
    for i, (l, c, _) in words.items():
        by_c[c].append(i)
    rng = random.Random(1)
    by_l = collections.defaultdict(list)
    for i, (l, c, _) in words.items():
        by_l[l].append(i)
    jobs = []
    for c, ids in by_c.items():  # real pairs
        for a, b in itertools.combinations(ids, 2):
            la, lb = words[a][0], words[b][0]
            if la != lb:
                jobs.append((("real", c), a, b, words[a][2], words[b][2], la, lb))
    for la, lb in itertools.combinations(sorted(by_l), 2):  # null: different concepts, same two languages
        for _ in range(NULL_PAIRS):
            a, b = rng.choice(by_l[la]), rng.choice(by_l[lb])
            if words[a][1] != words[b][1]:
                jobs.append((("null", (la, lb)), a, b, words[a][2], words[b][2], la, lb))
    chunks = [jobs[k::workers * 4] for k in range(workers * 4)]
    with ProcessPoolExecutor(workers) as pool:
        res = [r for part in pool.map(_pair_scores, [(corr, ch) for ch in chunks]) for r in part]
    null = collections.defaultdict(list)
    real = []
    for key, a, b, s in res:
        if key[0] == "null":
            null[key[1]].append(s)
        else:
            real.append((a, b, s))
    mom = {k: (float(np.mean(v)), float(np.std(v)) or 1.0) for k, v in null.items()}
    z = {}
    for a, b, s in real:
        la, lb = sorted((words[a][0], words[b][0]))
        m, sd = mom[(la, lb)]
        z[(a, b)] = z[(b, a)] = (s - m) / sd
    out = {"family": family, "seed": seeds, "z": {f"{a}\t{b}": v for (a, b), v in z.items() if a < b}}
    out["codings"] = {f"corr_z{zt:g}": merge(seeds, z, words, zt) for zt in SENSITIVITY}
    return out


def merge(seeds: dict, z: dict, words: dict, z_merge: float) -> dict:
    """Average-linkage merging of seed sets within each concept while mean cross z >= z_merge."""
    out = {}
    by_c = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, s in seeds.items():
        by_c[words[i][1]][s].append(i)
    for c, groups in by_c.items():
        clusters = [list(v) for v in groups.values()]

        def link(A, B):
            v = [z[(a, b)] for a in A for b in B if (a, b) in z]
            return float(np.mean(v)) if v else -math.inf
        while len(clusters) > 1:
            best, bi, bj = -math.inf, -1, -1
            for i, j in itertools.combinations(range(len(clusters)), 2):
                v = link(clusters[i], clusters[j])
                if v > best:
                    best, bi, bj = v, i, j
            if best < z_merge:
                break
            clusters[bi] += clusters.pop(bj)
        for k, cl in enumerate(clusters):
            for i in cl:
                out[i] = f"{c}-m{k}"
    return out


def evaluate(family: str, codings: dict) -> dict:
    from sim_realism import bcubed
    taxa, forms, concept, gold, _, partial = family_data(family)
    lang = {f["ID"]: f["Language_ID"] for f in forms}
    res = {}
    for name, cog in codings.items():
        lab = [f for f in cog if f in gold]
        P, R, F = bcubed({f: gold[f] for f in lab}, {f: cog[f] for f in lab})
        sets = collections.defaultdict(set)
        for f, s in cog.items():
            sets[s].add(lang[f])
        res[name] = {"P": P, "R": R, "F": F, "sets": len(sets), "sets_labelled": len({cog[f] for f in lab}),
                     "gold_sets_labelled": len({gold[f] for f in lab}),
                     "singleton_share": float(np.mean([len(v) == 1 for v in sets.values()]))}
    return res


def cmd_detect(a) -> None:
    global END_GAP
    from pipeline.asjp import lexstat_cognates
    OUT.mkdir(parents=True, exist_ok=True)
    tag = ""
    if a.edge_free:  # via the environment so that spawned workers see it too
        os.environ["M8_END_GAP"] = "0.0"
        END_GAP, tag = 0.0, "_edgefree"
    for fam in (("mayan", "ie") if a.family == "both" else (a.family,)):
        d = detect(fam)
        taxa, forms, concept, gold, _, _ = family_data(fam)
        with contextlib.redirect_stderr(io.StringIO()):
            ref = {f"lexstat_t{round(t * 100)}": lexstat_cognates(forms, t) for t in (0.45, 0.55, 0.65, 0.75)}
        ev = evaluate(fam, {**ref, **d["codings"], "expert": {f: gold[f] for f in gold if f in d["seed"]}})
        (OUT / f"{fam}{tag}_codings.json").write_text(json.dumps(d["codings"]))
        (OUT / f"{fam}{tag}_eval.json").write_text(json.dumps(ev, indent=2))
        print(f"== {fam}{tag}")
        for k, v in ev.items():
            print(f"{k:14s} sets {v['sets']:5d} (labelled {v['sets_labelled']} vs gold {v['gold_sets_labelled']}); "
                  f"singletons {v['singleton_share']:.2f}; P {v['P']:.3f} R {v['R']:.3f} F {v['F']:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("detect"); p.add_argument("--family", choices=["mayan", "ie", "both"], default="both")
    p.add_argument("--edge-free", action="store_true")
    a = ap.parse_args()
    {"detect": cmd_detect}[a.cmd](a)


if __name__ == "__main__":
    main()
