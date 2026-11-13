import numpy as np

from hmm.phylo import (CladeIndex, dollo_losses, fit_logistic, predict_logistic, set_centred_scores,
                       transfer_scores, upgma_tree)

F = frozenset
H = {F("A"): 0, F("B"): 0, F("C"): 0, F("D"): 0, F("AB"): 0.1, F("CD"): 0.1, F("ABCD"): 1.0}
CONCEPT = {"k": "x", "j": "x"}


def test_dollo_losses_counts_maximal_absent_clades():
    idx = CladeIndex(H)
    assert dollo_losses({"A", "B"}, set("ABCD"), idx) == 0
    assert dollo_losses({"A", "C"}, set("ABCD"), idx) == 2      # B and D lost
    assert dollo_losses({"A", "C"}, {"A", "C", "D"}, idx) == 1  # B missing: free


def test_transfer_score_prefers_patchy_sets():
    patchy = transfer_scores({"A": {"k": 1}, "C": {"k": 1}, "B": {"j": 1}, "D": {"j": 1}}, CONCEPT, H)
    clean = transfer_scores({"A": {"k": 1}, "B": {"k": 1}, "C": {"j": 1}, "D": {"j": 1}}, CONCEPT, H)
    assert patchy[("A", "k")]["llr"] > clean[("A", "k")]["llr"] + 3


def test_set_centred_removes_set_wide_similarity():
    # set s: every pair equally similar (a retention); entry A's centred score is 0
    pair = {(l, "s"): {m: 2.0 for m in "ABC" if m != l} for l in "ABC"}
    out = set_centred_scores(pair)
    assert out[("A", "s")]["mean"] == 2.0
    assert abs(out[("A", "s")]["centred"]) < 1e-12
    # set t: only the A-B pair is similar; A is anomalous relative to B-C, C-...
    pair = {("A", "t"): {"B": 3.0, "C": 3.0}, ("B", "t"): {"A": 3.0, "C": 0.0}, ("C", "t"): {"A": 3.0, "B": 0.0}}
    assert set_centred_scores(pair)[("A", "t")]["centred"] == 3.0


def test_upgma_tree_heights_monotone():
    cog = {"A": {"k": 1, "m": 1}, "B": {"k": 1, "m": 1}, "C": {"j": 1, "n": 1}, "D": {"j": 1, "m": 1}}
    co = {"k": "x", "j": "x", "m": "y", "n": "y"}
    Hu = upgma_tree(cog, co)
    assert Hu[F("AB")] < Hu[F("ABCD")]
    assert all(v >= 0 for v in Hu.values())


def test_logistic_recovers_direction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.5 + 2 * X[:, 0])))).astype(float)
    w = fit_logistic(X, y, l2=0.0)
    assert abs(w[1] - 2) < 0.3 and abs(w[2]) < 0.3
    assert predict_logistic(w, np.array([[3.0, 0.0]]))[0] > 0.9
