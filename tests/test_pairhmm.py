"""Tests for the pair HMM: forward against brute-force alignment enumeration,
posterior counts against finite differences, batching, and detection power."""

import itertools

import numpy as np

from hmm.pairhmm import K, PairHMM, encode, entry_scores, identity_match


def random_hmm(seed=0):
    rng = np.random.default_rng(seed)
    gap = rng.dirichlet(np.ones(K))
    m = rng.dirichlet(np.ones(K * K)).reshape(K, K)
    return PairHMM(m, gap, delta=0.07, eps=0.35, tau=0.12)


def brute_force(h: PairHMM, x: str, y: str) -> float:
    """Sum over all alignments (state paths) by explicit enumeration."""
    from hmm.pairhmm import SEG
    aMM, aXM = 1 - 2 * h.delta - h.tau, 1 - h.eps - h.tau
    trans = {("M", "M"): aMM, ("M", "X"): h.delta, ("M", "Y"): h.delta,
             ("X", "M"): aXM, ("X", "X"): h.eps, ("X", "Y"): 0.0,
             ("Y", "M"): aXM, ("Y", "Y"): h.eps, ("Y", "X"): 0.0}
    total = 0.0

    def rec(i, j, prev, p):
        nonlocal total
        if i == len(x) and j == len(y):
            total += p * h.tau
            return
        if i < len(x) and j < len(y):
            rec(i + 1, j + 1, "M", p * trans[(prev, "M")] * h.match[SEG[x[i]], SEG[y[j]]])
        if i < len(x):
            rec(i + 1, j, "X", p * trans[(prev, "X")] * h.gap[SEG[x[i]]])
        if j < len(y):
            rec(i, j + 1, "Y", p * trans[(prev, "Y")] * h.gap[SEG[y[j]]])

    rec(0, 0, "M", 1.0)
    return total


def test_forward_matches_brute_force():
    h = random_hmm()
    for x, y in [("ta", "da"), ("kasu", "ksu"), ("pi", "pino"), ("a", "ub")]:
        X, nx = encode([x]); Y, ny = encode([y])
        assert np.isclose(np.exp(h.loglik(X, nx, Y, ny)[0]), brute_force(h, x, y), rtol=1e-10)


def test_batch_with_padding_equals_individual():
    h = random_hmm(1)
    xs = ["tapa", "ki", "sulema", "o"]; ys = ["dabo", "kix", "sul", "uo"]
    X, nx = encode(xs); Y, ny = encode(ys)
    batch = h.loglik(X, nx, Y, ny)
    single = [h.loglik(*encode([a]), *encode([b]))[0] for a, b in zip(xs, ys)]
    assert np.allclose(batch, single)


def test_expected_counts_match_finite_differences():
    # E[n(a,b)] = p(a,b) * d log P / d p(a,b) for the unnormalised match matrix
    h = random_hmm(2)
    x, y = "kasat", "gasd"
    X, nx = encode([x]); Y, ny = encode([y])
    counts, logp = h.expected_match_counts(X, nx, Y, ny, np.ones(1))
    from hmm.pairhmm import SEG
    for a, b in [("k", "g"), ("a", "a"), ("s", "s"), ("t", "d"), ("s", "d")]:
        ia, ib = SEG[a], SEG[b]
        eps = 1e-6 * h.match[ia, ib]
        m2 = h.match.copy(); m2[ia, ib] += eps
        lp2 = PairHMM(m2, h.gap, h.delta, h.eps, h.tau).loglik(X, nx, Y, ny)[0]
        fd = h.match[ia, ib] * (lp2 - logp[0]) / eps
        assert np.isclose(counts[ia, ib], fd, rtol=1e-4, atol=1e-9), (a, b)
    assert counts.sum() <= min(len(x), len(y)) + 1e-9


def test_detection_power_on_simulated_loans():
    from sim.borrowing import BorrowingParams, simulate_with_borrowing
    from sim.phonology import PhonologyParams
    from sim.sdollo import SDolloParams
    from sim.tree import simulate_yule_given_root

    rng = np.random.default_rng(0)
    tree = simulate_yule_given_root(8, 4000.0, 6e-4, rng)
    p = SDolloParams(n_concepts=200, birth_rate=2e-4, loss_rate=2e-4)
    r = simulate_with_borrowing(tree, p, BorrowingParams(1e-4), seed=3,
                                phonology=PhonologyParams(change_rate=1 / 80))
    obs = r.observed()
    keep = r.presence.any(axis=0)
    borrowed = r.extra["borrowed"][:, keep]
    wl = {l: {str(j): obs.forms[i, j] for j in np.flatnonzero(obs.presence[i])}
          for i, l in enumerate(obs.leaf_names)}
    scores = entry_scores(wl)
    s, lab = [], []
    for i, l in enumerate(obs.leaf_names):
        for j in np.flatnonzero(obs.presence[i]):
            v = scores[(l, str(j))]
            if np.isfinite(v):
                s.append(v); lab.append(borrowed[i, j])
    s, lab = np.array(s), np.array(lab)
    assert lab.sum() >= 20
    # AUC via the Mann-Whitney statistic
    ranks = s.argsort().argsort() + 1
    auc = (ranks[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum())
    # ~0.75 here; form evidence cannot say which side of a loan relation borrowed,
    # which caps entry-level AUC (see README, milestone 3)
    assert auc > 0.7, auc


def test_upgma_clades_and_lineage_model():
    from hmm.latent import fit_latent_lineage, upgma_clades
    entries = {"A": {"1", "2", "3"}, "B": {"1", "2", "4"}, "C": {"5", "6", "3"}, "D": {"5", "6", "7"}}
    clades, sister = upgma_clades(entries)
    assert frozenset({"A", "B"}) in clades and frozenset({"C", "D"}) in clades
    assert sister["A"] == frozenset({"B"}) and sister["C"] == frozenset({"D"})
    # set 3 is shared across the (A,B) | (C,D) split with a loan-like score
    pair = {(l, k): {} for l, ks in entries.items() for k in ks}
    for a, b, k, v in [("A", "B", "1", 0.1), ("A", "B", "2", -0.2), ("C", "D", "5", 0.0),
                       ("C", "D", "6", 0.3), ("A", "C", "3", 4.0)]:
        pair[(a, k)][b] = v; pair[(b, k)][a] = v
    concept_of = {k: f"c{k}" for k in "1234567"}
    f = fit_latent_lineage(pair, concept_of)
    assert all(0 <= p <= 1 for p in f.p_borrowed.values())
    assert f.p_borrowed[("A", "3")] > f.p_borrowed[("A", "1")]
    assert f.p_borrowed[("B", "4")] == 0.0  # singleton: nothing to compare


def test_segment_lists_and_custom_alphabet():
    # IPA-like multi-character segments outside the simulator inventory
    wl = {"A": {"1": ["tʃ", "a"], "2": ["k'", "i", "n"], "3": ["h", "a", "ʔ"]},
          "B": {"1": ["tʃ", "e"], "2": ["q'", "i", "x"], "3": ["h", "a", "ʔ"]},
          "C": {"1": ["ts", "a"], "2": ["k'", "i", "n"], "3": ["a", "ʔ"]}}
    s = entry_scores(wl)
    assert set(s) == {(l, k) for l in wl for k in wl[l]}
    assert all(np.isfinite(v) for v in s.values())
