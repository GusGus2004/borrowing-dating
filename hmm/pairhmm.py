"""Pair HMM over phoneme strings and a mixture model for borrowing detection.

The pair HMM is the standard three-state model (Durbin et al. 1998, ch. 4):
match ``M`` emits an aligned segment pair with probability ``p(a, b)``;
``X`` / ``Y`` emit a segment against a gap with background probability
``q(a)``. Transitions: M->X, M->Y = delta; X->X, Y->Y = eps; every state
ends with probability tau. Forward and backward are computed for a whole
batch of string pairs at once (numpy over the batch, Python over the
alignment grid; strings are short).

Borrowing detection, per language pair (A, B), over the forms the two
languages have in the same cognate set:

* a **regular** component: a pair HMM whose match matrix ``P_AB`` is learned by
  EM, so it absorbs the regular sound correspondences between A and B;
* a **loan** component: a fixed pair HMM whose match matrix is near-identity
  (a loan copies the donor's form; only changes after the loan separate them).

EM fits ``P_AB`` and the loan proportion ``pi_AB`` jointly, so likely loans do
not contaminate the learned correspondences. Each pair gets a log-likelihood
ratio ``log P(x, y | loan) - log P(x, y | regular_AB)``.

The model is symmetric in who borrowed from whom; :func:`entry_scores` breaks
the symmetry by averaging over all partners: a loan looks irregular against
every member of the donor clade, whereas a donor-clade form looks irregular
only against the recipient.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from sim.phonology import INVENTORY

# Default alphabet: the simulator's inventory. Real data (lists of IPA
# segments) get an alphabet built from the data in ``pair_scores``.
K = len(INVENTORY)
SEG = {s: i for i, s in enumerate(INVENTORY)}


def encode(forms: Sequence[Sequence[str]], seg: Mapping[str, int] = SEG) -> tuple[np.ndarray, np.ndarray]:
    """Pad forms (strings or segment lists) into an int array (N, Lmax) plus lengths."""
    n = np.array([len(f) for f in forms], dtype=int)
    out = np.zeros((len(forms), max(n.max(initial=1), 1)), dtype=int)
    for k, f in enumerate(forms):
        out[k, :len(f)] = [seg[c] for c in f]
    return out, n


@dataclass
class PairHMM:
    match: np.ndarray        # (K, K) joint p(a, b), sums to 1
    gap: np.ndarray          # (K,) background q(a), sums to 1
    delta: float = 0.05
    eps: float = 0.3
    tau: float = 0.1

    def _trans(self):
        d, e, t = self.delta, self.eps, self.tau
        return 1 - 2 * d - t, 1 - e - t  # a_MM, a_XM (= a_YM)

    def forward(self, x, nx, y, ny):
        """Returns log P(x, y) per pair and the forward arrays."""
        N, I, J = x.shape[0], x.shape[1] + 1, y.shape[1] + 1
        aMM, aXM = self._trans()
        d, e = self.delta, self.eps
        fM = np.zeros((N, I, J)); fX = np.zeros((N, I, J)); fY = np.zeros((N, I, J))
        fM[:, 0, 0] = 1.0
        qx = self.gap[x]; qy = self.gap[y]
        for i in range(I):
            for j in range(J):
                if i == 0 and j == 0:
                    continue
                if i > 0 and j > 0:
                    fM[:, i, j] = self.match[x[:, i - 1], y[:, j - 1]] * (
                        aMM * fM[:, i - 1, j - 1] + aXM * (fX[:, i - 1, j - 1] + fY[:, i - 1, j - 1]))
                if i > 0:
                    fX[:, i, j] = qx[:, i - 1] * (d * fM[:, i - 1, j] + e * fX[:, i - 1, j])
                if j > 0:
                    fY[:, i, j] = qy[:, j - 1] * (d * fM[:, i, j - 1] + e * fY[:, i, j - 1])
        r = np.arange(N)
        p = self.tau * (fM[r, nx, ny] + fX[r, nx, ny] + fY[r, nx, ny])
        return np.log(p), (fM, fX, fY)

    def loglik(self, x, nx, y, ny) -> np.ndarray:
        return self.forward(x, nx, y, ny)[0]

    def expected_match_counts(self, x, nx, y, ny, weights) -> tuple[np.ndarray, np.ndarray]:
        """Posterior-expected aligned-pair counts, weighted per string pair."""
        logp, (fM, _, _) = self.forward(x, nx, y, ny)
        N, I, J = fM.shape
        aMM, aXM = self._trans()
        d, e, t = self.delta, self.eps, self.tau
        bM = np.zeros((N, I, J)); bX = np.zeros((N, I, J)); bY = np.zeros((N, I, J))
        ii = np.arange(I)[None, :, None]; jj = np.arange(J)[None, None, :]
        inside = (ii <= nx[:, None, None]) & (jj <= ny[:, None, None])
        for i in range(I - 1, -1, -1):
            for j in range(J - 1, -1, -1):
                vM = np.zeros(N); vX = np.zeros(N); vY = np.zeros(N)
                if i < I - 1 and j < J - 1:
                    pm = self.match[x[:, i], y[:, j]] * bM[:, i + 1, j + 1]
                    vM += aMM * pm; vX += aXM * pm; vY += aXM * pm
                if i < I - 1:
                    px = self.gap[x[:, i]] * bX[:, i + 1, j]
                    vM += d * px; vX += e * px
                if j < J - 1:
                    py = self.gap[y[:, j]] * bY[:, i, j + 1]
                    vM += d * py; vY += e * py
                end = (nx == i) & (ny == j)
                vM = np.where(end, t, vM); vX = np.where(end, t, vX); vY = np.where(end, t, vY)
                ok = inside[:, i, j]
                bM[:, i, j] = vM * ok; bX[:, i, j] = vX * ok; bY[:, i, j] = vY * ok
        post = fM[:, 1:, 1:] * bM[:, 1:, 1:] / np.exp(logp)[:, None, None]  # (N, I-1, J-1)
        post *= weights[:, None, None]
        counts = np.zeros(self.match.shape)
        xi = np.broadcast_to(x[:, :, None], post.shape)
        yj = np.broadcast_to(y[:, None, :], post.shape)
        np.add.at(counts, (xi.ravel(), yj.ravel()), post.ravel())
        return counts, logp


def identity_match(gap: np.ndarray, mismatch: float) -> np.ndarray:
    """Joint p(a, b) = q(a) [(1 - m) 1{a=b} + m / (K-1) 1{a!=b}]."""
    K = len(gap)
    cond = np.full((K, K), mismatch / (K - 1))
    np.fill_diagonal(cond, 1 - mismatch)
    return gap[:, None] * cond


@dataclass
class PairFit:
    regular: PairHMM
    pi: float
    llr: np.ndarray          # per string pair: log P(loan) - log P(regular)
    resp: np.ndarray         # posterior P(loan) per string pair


def fit_pair(xa: Sequence[Sequence[str]], xb: Sequence[Sequence[str]], gap: np.ndarray, loan: PairHMM,
             n_iter: int = 8, prior_strength: float = 20.0, pi_prior=(1.0, 9.0),
             seg: Mapping[str, int] = SEG) -> PairFit:
    """EM for the regular/loan mixture on aligned cognate pairs of one language pair."""
    x, nx = encode(xa, seg); y, ny = encode(xb, seg)
    log_loan = loan.loglik(x, nx, y, ny)
    base = identity_match(gap, 0.5)
    reg = PairHMM(base.copy(), gap, loan.delta, loan.eps, loan.tau)
    pi = pi_prior[0] / sum(pi_prior)
    for _ in range(n_iter):
        log_reg = reg.loglik(x, nx, y, ny)
        a = np.log(pi) + log_loan; b = np.log1p(-pi) + log_reg
        resp = np.exp(a - np.logaddexp(a, b))
        counts, _ = reg.expected_match_counts(x, nx, y, ny, 1 - resp)
        m = counts + prior_strength * base
        reg = PairHMM(m / m.sum(), gap, loan.delta, loan.eps, loan.tau)
        pi = (resp.sum() + pi_prior[0] - 1) / (len(resp) + sum(pi_prior) - 2)
        pi = float(np.clip(pi, 1e-4, 1 - 1e-4))
    log_reg = reg.loglik(x, nx, y, ny)
    a = np.log(pi) + log_loan; b = np.log1p(-pi) + log_reg
    return PairFit(reg, pi, log_loan - log_reg, np.exp(a - np.logaddexp(a, b)))


def robust_z(v: np.ndarray) -> np.ndarray:
    """(v - median) / (1.4826 MAD); falls back to sd, then to centring only."""
    med = np.median(v)
    s = 1.4826 * np.median(np.abs(v - med))
    if not s > 0:
        s = v.std()
    return (v - med) / s if s > 0 else v - med


def pair_scores(wordlist: Mapping[str, Mapping[str, str]], loan_mismatch: float = 0.1,
                standardise: bool = True, **fit_kw) -> dict[tuple[str, str], dict[str, float]]:
    """Pair-level loan scores: ``out[(lang, cogset)][partner] = score``.

    ``wordlist[lang][cogset] = form``, a string (one character per segment)
    or a sequence of segment strings. For each language pair, every shared
    cognate set gets the LLR (loan vs regular). With ``standardise`` the LLRs
    are robustly z-scored within the language pair, which removes the overall
    divergence of the two languages (raw LLRs of distant pairs are shifted as
    a block).
    """
    langs = sorted(wordlist)
    segs = [c for l in langs for f in wordlist[l].values() for c in f]
    seg = SEG if set(segs) <= set(SEG) else {c: i for i, c in enumerate(sorted(set(segs)))}
    gap = np.bincount([seg[c] for c in segs], minlength=len(seg)) + 1.0
    gap /= gap.sum()
    loan = PairHMM(identity_match(gap, loan_mismatch), gap)
    out: dict[tuple[str, str], dict[str, float]] = {(l, k): {} for l in langs for k in wordlist[l]}
    for a, b in combinations(langs, 2):
        shared = sorted(set(wordlist[a]) & set(wordlist[b]))
        if not shared:
            continue
        fit = fit_pair([wordlist[a][k] for k in shared], [wordlist[b][k] for k in shared],
                       gap, loan, seg=seg, **fit_kw)
        z = robust_z(fit.llr) if standardise else fit.llr
        for k, v in zip(shared, z):
            out[(a, k)][b] = float(v)
            out[(b, k)][a] = float(v)
    return out


def entry_scores(wordlist: Mapping[str, Mapping[str, str]], **kw) -> dict[tuple[str, str], float]:
    """Level-1 borrowing score per (language, cognate set): mean pair score over
    the other languages with a form in the same set (NaN for singleton sets)."""
    return {e: (float(np.mean(list(p.values()))) if p else float("nan"))
            for e, p in pair_scores(wordlist, **kw).items()}
