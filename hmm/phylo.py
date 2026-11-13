"""Level 3: loan scores that use the tree as well as the forms.

Level 1 (``hmm.pairhmm.entry_scores``) asks "is this word more similar to its
cognates than this language pair's sound correspondences predict?". Two
failures motivate this module:

* **Retentions.** A word built from stable segments (Mayan *ha'* "water") is
  similar in every language pair, so its level-1 score is high everywhere.
  :func:`set_centred_scores` compares an entry with the other pairs *in the
  same cognate set*: a retention is unremarkable there, a loan is not.
* **No phylogenetic expectation.** A loan from a distant donor leaves a
  patchy distribution: the set appears in the recipient and the donor's
  relatives but not in the recipient's own relatives, so Dollo needs many
  losses to explain it. :func:`patchiness_scores` measures how many losses
  disappear when an entry, or a clade containing it, is treated as borrowed.
  This uses only the cognate matrix and a tree, so it works when the borrowed
  form is identical to the donor's (where form evidence is weakest).

:func:`fit_logistic` combines these with level 1; the weights are learned on
simulations where the truth is known.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

Clade = frozenset


# ---------------------------------------------------------------- trees as clade sets
def clades_from_tree(tree) -> list[Clade]:
    """All clades (sets of leaf names) of a ``sim.tree.Tree``, root included."""
    out = []
    for n in tree.postorder():
        out.append(Clade(l.name for l in _leaves(n)))
    return out


def _leaves(n):
    return [n] if n.is_leaf else [l for c in n.children for l in _leaves(c)]


def upgma_clades_all(entries: Mapping[str, set]) -> list[Clade]:
    """Clades of a UPGMA tree on cognate-sharing distance (root and leaves included)."""
    from hmm.latent import upgma_clades
    clades, _ = upgma_clades(entries)
    return list(set(clades) | {Clade(entries)})


class CladeIndex:
    def __init__(self, clades: Iterable[Clade]):
        self.clades = sorted(set(clades), key=len)
        self.parent: dict[Clade, Clade] = {}
        for c in self.clades:
            for d in self.clades:
                if len(d) > len(c) and c < d:
                    self.parent[c] = d
                    break
        self.children: dict[Clade, list[Clade]] = {c: [] for c in self.clades}
        for c, p in self.parent.items():
            self.children[p].append(c)

    def mrca(self, s: set) -> Clade:
        for c in self.clades:
            if s <= c:
                return c
        raise ValueError("no clade contains the set")

    def containing(self, leaf: str) -> list[Clade]:
        return [c for c in self.clades if leaf in c]


def dollo_losses(members: set, attested: set, idx: CladeIndex) -> int:
    """Minimum losses for a set present in ``members``, under Dollo on the tree.

    ``attested`` = languages with any form for the concept (others are missing
    data, free to be either). Losses = number of maximal clades below the
    set's MRCA that contain no member but at least one attested non-member.
    """
    if not members:
        return 0
    top = idx.mrca(members)
    absent = attested - members
    n, stack = 0, [top]
    while stack:
        c = stack.pop()
        if c & members:
            stack.extend(idx.children[c])
        elif c & absent:
            n += 1
    return n


def patchiness_scores(cog: Mapping[str, Mapping[str, str]], concept_of: Mapping[str, str],
                      clades: Iterable[Clade]) -> dict[tuple[str, str], dict[str, float]]:
    """Per entry (language, set): losses saved by treating it as borrowed.

    ``cog[lang][set] = form`` (only keys are used); ``concept_of[set]``.
    Returns, per entry, ``single`` (losses saved by removing just this
    language), ``clade`` (best saving over clades containing it, minus one for
    the transfer event), ``clade_frac`` (size of that clade / set size) and
    ``set_size``.
    """
    idx = CladeIndex(clades)
    members_of: dict[str, set] = {}
    attested_of: dict[str, set] = {}
    for l, sets in cog.items():
        for k in sets:
            members_of.setdefault(k, set()).add(l)
            attested_of.setdefault(concept_of[k], set()).add(l)
    out = {}
    for k, M in members_of.items():
        A = attested_of[concept_of[k]]
        base = dollo_losses(M, A, idx)
        for l in M:
            rest = M - {l}
            single = base - dollo_losses(rest, A, idx) if rest else 0.0
            best, frac = single - 1.0, 1 / len(M)
            for c in idx.containing(l):
                if c <= M and len(c) < len(M):
                    v = base - dollo_losses(M - c, A, idx) - 1.0
                    if v > best:
                        best, frac = v, len(c) / len(M)
            out[(l, k)] = {"single": float(single), "clade": float(best), "clade_frac": frac,
                           "set_size": float(len(M)), "base_losses": float(base)}
    return out


# ---------------------------------------------------------------- set-centred form evidence
def set_centred_scores(pair: Mapping[tuple[str, str], Mapping[str, float]]) -> dict[tuple[str, str], dict[str, float]]:
    """Form evidence relative to the entry's own cognate set.

    ``pair[(lang, set)][partner]`` = within-language-pair z score
    (``hmm.pairhmm.pair_scores``). For each entry: ``mean`` = its level-1
    score; ``centred`` = mean minus the mean over the set's pairs that do
    *not* involve this entry (how similar the set is anyway; retentions are
    similar everywhere), or 0 when the set has fewer than 3 members;
    ``max`` = its single most loan-like partner, ``max_centred`` likewise.
    """
    pairs_of: dict[str, dict[frozenset, float]] = {}
    for (l, k), p in pair.items():
        for m, v in p.items():
            pairs_of.setdefault(k, {})[frozenset((l, m))] = v
    out = {}
    for (l, k), p in pair.items():
        if not p:
            continue
        v = np.array(list(p.values()))
        others = [x for key, x in pairs_of[k].items() if l not in key]
        m = float(np.mean(others)) if others else float(v.mean())
        out[(l, k)] = {"mean": float(v.mean()), "centred": float(v.mean() - m),
                       "max": float(v.max()), "max_centred": float(v.max() - m),
                       "set_background": m if others else float("nan")}
    return out


# ---------------------------------------------------------------- combination
def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, n_iter: int = 50) -> np.ndarray:
    """L2-penalised logistic regression by Newton's method; returns weights (intercept first)."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    R = l2 * np.eye(len(w)); R[0, 0] = 0.0
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + R @ w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(w[0] + X @ w[1:])))


FEATURES: Sequence[str] = ("mean", "centred", "max_centred", "single", "clade", "clade_frac", "set_size",
                           "llr", "llr_single", "mrca_height", "deep_sim", "deep_sim_max")


def feature_matrix(form: Mapping, patch: Mapping, entries: Sequence[tuple[str, str]],
                   n_langs: int) -> np.ndarray:
    """``patch[e]`` holds the parsimony and likelihood tree features for entry e."""
    rows = []
    for e in entries:
        f, p = form[e], patch[e]
        rows.append([f["mean"], f["centred"], f["max_centred"], p["single"], p["clade"],
                     p["clade_frac"], p["set_size"] / n_langs, p["llr"], p["llr_single"], p["mrca_height"],
                     p.get("deep_sim", 0.0), p.get("deep_sim_max", 0.0)])
    return np.array(rows, float)


# ---------------------------------------------------------------- likelihood version
def heights_from_tree(tree, loss_rate: float) -> dict[Clade, float]:
    """Clade -> node age x loss rate (expected losses per class since the node)."""
    return {Clade(l.name for l in _leaves(n)): n.age * loss_rate for n in tree.postorder()}


def upgma_tree(cog: Mapping[str, Mapping[str, str]], concept_of: Mapping[str, str]) -> dict[Clade, float]:
    """UPGMA on lexical distance, heights in expected losses per class.

    Distance between two languages: among concepts both attest, the fraction
    with no shared set, d; under Dollo loss, 1 - d ~ exp(-2 mu t), so
    -log(1 - d) / 2 estimates mu t. Returns clade -> height (leaves 0).
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    langs = sorted(cog)
    conc = {l: {} for l in langs}
    for l in langs:
        for k in cog[l]:
            conc[l].setdefault(concept_of[k], set()).add(k)
    n = len(langs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            both = set(conc[langs[i]]) & set(conc[langs[j]])
            d = np.mean([not (conc[langs[i]][c] & conc[langs[j]][c]) for c in both]) if both else 0.99
            D[i, j] = D[j, i] = -np.log(max(1 - d, 1e-3))
    Z = linkage(squareform(D, checks=False), method="average")
    members = [Clade([l]) for l in langs]
    heights = {m: 0.0 for m in members}
    for a, b, h, _ in Z:
        c = members[int(a)] | members[int(b)]
        members.append(c)
        heights[c] = max(h / 2, heights[members[int(a)]], heights[members[int(b)]])
    return heights


def _log_present(c: Clade, members: set, attested: set, idx: CladeIndex, H: Mapping[Clade, float]) -> float:
    """log P(leaf pattern below c | set present at c) under Dollo loss (rate 1 per unit height)."""
    if len(c) == 1:
        (l,) = c
        return 0.0 if (l in members or l not in attested) else -np.inf
    out = 0.0
    for ch in idx.children[c]:
        b = max(H[c] - H[ch], 1e-6)
        keep = np.exp(-b)
        lp = _log_present(ch, members, attested, idx, H)
        absent_ok = not (ch & members)
        terms = [np.log(keep) + lp] if np.isfinite(lp) else []
        if absent_ok:
            terms.append(np.log1p(-keep))
        out += np.logaddexp.reduce(terms) if terms else -np.inf
    return out


def transfer_scores(cog: Mapping[str, Mapping[str, str]], concept_of: Mapping[str, str],
                    heights: Mapping[Clade, float]) -> dict[tuple[str, str], dict[str, float]]:
    """Per entry: log-likelihood gain from explaining a clade containing it as a transfer.

    Inherited: the set is present at MRCA(M) and lost along the tree.
    Transfer of clade C: the rest is present at MRCA(M - C), C's members at
    MRCA(C). ``llr`` = best gain over clades C (C a clade of the tree, C
    inside M, C != M) containing the entry; ``llr_single`` for C = {entry}.
    Birth terms are left out: the classifier's intercept and the set-size
    feature absorb the fixed cost of an extra origin.
    """
    idx = CladeIndex(heights)
    members_of: dict[str, set] = {}
    attested_of: dict[str, set] = {}
    for l, sets in cog.items():
        for k in sets:
            members_of.setdefault(k, set()).add(l)
            attested_of.setdefault(concept_of[k], set()).add(l)
    out = {}
    for k, M in members_of.items():
        A = attested_of[concept_of[k]]
        top = idx.mrca(M)
        base = _log_present(top, M, A, idx, heights)
        for l in M:
            best, single = -np.inf, 0.0
            for c in idx.containing(l):
                if not (c <= M) or len(c) >= len(M):
                    continue
                rest = M - c
                v = (_log_present(idx.mrca(rest), rest, A, idx, heights)
                     + _log_present(idx.mrca(c), c, A, idx, heights) - base)
                if len(c) == 1:
                    single = v
                best = max(best, v)
            out[(l, k)] = {"llr": float(best) if np.isfinite(best) else 0.0, "llr_single": float(single),
                           "mrca_height": float(heights[top])}
    return out


def deep_similarity(pair: Mapping[tuple[str, str], Mapping[str, float]],
                    heights: Mapping[Clade, float]) -> dict[tuple[str, str], dict[str, float]]:
    """Form x tree: is the entry most similar to a *distant* partner?

    For entry e and partner j: depth of MRCA(e, j) (relative to the root)
    times the pair z score. ``deep_sim`` = mean over partners, ``deep_sim_max``
    = max. A loan is unusually similar to a distant partner; an inherited word
    is similar mostly to close relatives; a false cognate merge spans distant
    languages but with low similarity.
    """
    idx = CladeIndex(heights)
    root_h = max(heights.values()) or 1.0
    cache: dict[frozenset, float] = {}
    out = {}
    for (l, k), p in pair.items():
        if not p:
            continue
        v = []
        for m, z in p.items():
            key = frozenset((l, m))
            if key not in cache:
                cache[key] = heights[idx.mrca(set(key))] / root_h
            v.append(cache[key] * z)
        out[(l, k)] = {"deep_sim": float(np.mean(v)), "deep_sim_max": float(np.max(v))}
    return out
