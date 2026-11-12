"""Level 2: latent-donor mixture over a language's lexicon.

Every entry ``e = (A, k)`` (language A has a form in cognate set k) has a
latent origin ``z_e``: ``0`` = inherited, or ``d`` = borrowed from (a lineage
leading to) language ``d``, one of the other languages in set k. Each language
has a borrowing profile ``pi_A`` over {inherited, donors}, with a Dirichlet
prior that favours "inherited" and is sparse over donors, so a language's
loans are explained by few donors. This is an admixture model, not an HMM:
entries of a wordlist have no sequential order.

Evidence for ``z_e = d`` relative to ``z_e = 0``:

1. **Form** (level 1): the pair-HMM score ``s_ed`` (``hmm.pairhmm.pair_scores``),
   modelled as Gaussian with class-specific mean and variance, ``f1`` for the
   (e, donor) pair and ``f0`` otherwise. On its own this is symmetric: the
   donor's form looks just as "loan-like" against the recipient.
2. **Direction** (lexical, tree-free): the languages lexically closer to A
   than d is (A's relatives, judged by cognate sharing) that have some form
   for the concept. If A borrowed k from d, those relatives mostly lack k
   (probability ``theta1`` each); if A inherited k, they lack it with the
   background probability ``theta0``. For the donor side the relatives
   mostly share k, so this term breaks the symmetry.

All parameters (f0, f1, theta0, theta1, profiles) are fitted by EM. Output:
posterior P(borrowed) per entry and the posterior over donors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import stats


@dataclass
class LatentFit:
    p_borrowed: dict[tuple[str, str], float]
    donor_posterior: dict[tuple[str, str], dict[str, float]]
    profiles: dict[str, dict[str, float]]
    params: dict = field(default_factory=dict)


def _similarity(entries: Mapping[str, set[str]]) -> dict[tuple[str, str], float]:
    """Cosine similarity of cognate-set membership between languages."""
    langs = sorted(entries)
    return {(a, b): len(entries[a] & entries[b]) / np.sqrt(len(entries[a]) * len(entries[b]))
            for a in langs for b in langs if a != b}


def fit_latent(pair: Mapping[tuple[str, str], Mapping[str, float]],
               concept_of: Mapping[str, str],
               alpha_inherited: float = 10.0, alpha_donor: float = 0.5,
               use_direction: bool = True, fix_null: bool = False, min_separation: float = 1.0,
               n_iter: int = 50, tol: float = 1e-6) -> LatentFit:
    """EM for the latent-donor model.

    ``pair[(A, k)][d]`` is the level-1 score of A's and d's forms in set k (every
    entry of every language must be a key, with an empty dict for singletons).
    ``concept_of[k]`` is the concept of cognate set k. With ``fix_null`` the
    inherited score density is fixed at N(0, 1) (right for within-pair robust
    z-scores) and the loan mean is kept at least ``min_separation`` above it.
    """
    entries: dict[str, set[str]] = {}
    for (a, k) in pair:
        entries.setdefault(a, set()).add(k)
    langs = sorted(entries)
    sim = _similarity(entries)
    has_concept = {a: {concept_of[k] for k in entries[a]} for a in langs}
    members: dict[str, set[str]] = {}
    for (a, k) in pair:
        members.setdefault(k, set()).add(a)

    # Precompute per (entry, candidate donor): score and direction counts
    E = sorted(pair)
    cand: list[list[str]] = []
    S: list[np.ndarray] = []
    lack: list[np.ndarray] = []
    nrel: list[np.ndarray] = []
    for (a, k) in E:
        ds = sorted(pair[(a, k)])
        cand.append(ds)
        S.append(np.array([pair[(a, k)][d] for d in ds]))
        c = concept_of[k]
        l_, n_ = [], []
        for d in ds:
            closer = [n for n in langs if n not in (a, d) and sim[(a, n)] > sim[(a, d)]
                      and c in has_concept[n]]
            n_.append(len(closer))
            l_.append(sum(n not in members[k] for n in closer))
        lack.append(np.array(l_, float)); nrel.append(np.array(n_, float))

    allS = np.concatenate([s for s in S if len(s)]) if any(len(s) for s in S) else np.zeros(1)
    if fix_null:  # robustly z-scored level-1 scores: inherited bulk is N(0, 1)
        m0, s0 = 0.0, 1.0
    else:
        m0, s0 = float(np.median(allS)), float(allS.std() or 1.0)
    m1, s1 = m0 + 2 * s0, s0
    th0, th1 = 0.3, 0.8
    prof = {a: {"_inh": 0.9} for a in langs}
    W0 = np.zeros(len(E))
    Wd: list[np.ndarray] = [np.zeros(len(c)) for c in cand]
    last = -np.inf
    for it in range(n_iter):
        # E-step
        ll_total = 0.0
        for i, (a, k) in enumerate(E):
            if not len(cand[i]):
                W0[i] = 1.0
                continue
            log_lr = stats.norm.logpdf(S[i], m1, s1) - stats.norm.logpdf(S[i], m0, s0)
            if use_direction:
                log_lr = log_lr + lack[i] * (np.log(th1) - np.log(th0)) + \
                         (nrel[i] - lack[i]) * (np.log1p(-th1) - np.log1p(-th0))
            p_inh = prof[a]["_inh"]
            p_d = np.array([prof[a].get(d, 1e-3) for d in cand[i]])
            p_d = (1 - p_inh) * p_d / p_d.sum()
            logw = np.concatenate([[np.log(p_inh)], np.log(p_d) + log_lr])
            mx = logw.max()
            z = np.exp(logw - mx)
            ll_total += mx + np.log(z.sum())
            z /= z.sum()
            W0[i] = z[0]
            Wd[i] = z[1:]
        # M-step
        w1 = np.concatenate([w for w in Wd if len(w)])
        s_all = np.concatenate([s for s in S if len(s)])
        w0s = 1 - w1
        m1 = float((w1 * s_all).sum() / max(w1.sum(), 1e-9))
        s1 = float(np.sqrt((w1 * (s_all - m1) ** 2).sum() / max(w1.sum(), 1e-9)) + 1e-3)
        if fix_null:
            m1 = max(m1, m0 + min_separation * s0)
        else:
            m0 = float((w0s * s_all).sum() / w0s.sum())
            s0 = float(np.sqrt((w0s * (s_all - m0) ** 2).sum() / w0s.sum()) + 1e-3)
            if m1 < m0:  # keep labels identified: loans are the high-score component
                m0, m1, s0, s1 = m1, m0, s1, s0
        if use_direction:
            la = np.concatenate([x for x in lack if len(x)]); nr = np.concatenate([x for x in nrel if len(x)])
            th1 = float(np.clip(((w1 * la).sum() + 1) / ((w1 * nr).sum() + 2), 0.02, 0.98))
            th0 = float(np.clip(((w0s * la).sum() + 1) / ((w0s * nr).sum() + 2), 0.02, 0.98))
        for a in langs:
            idx = [i for i, (aa, _) in enumerate(E) if aa == a]
            cnt: dict[str, float] = {"_inh": float(W0[idx].sum()) + alpha_inherited - 1}
            for i in idx:
                for d, w in zip(cand[i], Wd[i]):
                    cnt[d] = cnt.get(d, 0.0) + w
            for d in langs:
                if d != a:
                    cnt[d] = max(cnt.get(d, 0.0) + alpha_donor - 1, 1e-6)
            tot = sum(cnt.values())
            inh = cnt.pop("_inh") / tot
            dsum = sum(cnt.values())
            prof[a] = {"_inh": float(np.clip(inh, 1e-4, 1 - 1e-4)),
                       **{d: v / dsum for d, v in cnt.items()}}
        if abs(ll_total - last) < tol * max(1.0, abs(last)):
            break
        last = ll_total

    return LatentFit(
        p_borrowed={e: float(1 - W0[i]) for i, e in enumerate(E)},
        donor_posterior={e: dict(zip(cand[i], map(float, Wd[i]))) for i, e in enumerate(E)},
        profiles=prof,
        params={"m0": m0, "s0": s0, "m1": m1, "s1": s1, "theta0": th0, "theta1": th1,
                "iterations": it + 1},
    )
