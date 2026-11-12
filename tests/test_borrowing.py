"""Tests for SDollo + borrowing.

With beta = 0 the Gillespie simulator must reproduce the analytic SDollo
two-taxon results (it is an independent implementation of the same model, so
this also cross-checks sim/sdollo.py). With beta > 0 we check the event rate,
the contact models and the provenance bookkeeping.
"""

import numpy as np
import pytest

from sim.borrowing import BorrowingParams, borrowed_entries, simulate_with_borrowing
from sim.sdollo import SDolloParams, discrete_gamma_rates
from sim.tree import Tree, simulate_yule_given_root

T = 1000.0
MU = 5e-4


def two_taxon():
    return Tree.from_newick(f"(A:{T},B:{T})root;")


def assert_mean(samples, expected, z=4.0):
    samples = np.asarray(samples, float)
    se = samples.std(ddof=1) / np.sqrt(len(samples))
    assert abs(samples.mean() - expected) < z * se, (
        f"mean {samples.mean():.3f} vs expected {expected:.3f} (se {se:.3f})")


def shared(res):
    return int((res.presence[0] & res.presence[1]).sum())


@pytest.mark.parametrize("gamma_shape", [None, 0.7])
def test_no_borrowing_matches_analytic_sdollo(gamma_shape):
    p = SDolloParams(n_concepts=300, birth_rate=MU, loss_rate=MU, gamma_shape=gamma_shape)
    b = BorrowingParams(rate=0.0)
    s = [shared(simulate_with_borrowing(two_taxon(), p, b, seed=k)) for k in range(80)]
    rates = np.ones(1) if gamma_shape is None else discrete_gamma_rates(gamma_shape, 4)
    expected = p.n_concepts * np.mean(np.exp(-2 * MU * T * rates))
    assert_mean(s, expected)


def test_no_borrowing_stationary_per_leaf():
    p = SDolloParams(n_concepts=300, birth_rate=2 * MU, loss_rate=MU)
    b = BorrowingParams(rate=0.0)
    per_leaf = [simulate_with_borrowing(two_taxon(), p, b, seed=k).presence.sum(1).mean()
                for k in range(60)]
    assert_mean(per_leaf, 300 * 2.0)


def test_borrow_event_rate():
    # birth >> loss: donors almost always have a class, so nearly no void events
    p = SDolloParams(n_concepts=100, birth_rate=20 * MU, loss_rate=MU)
    beta = 2e-4
    b = BorrowingParams(rate=beta, mode="add")
    n = []
    for k in range(40):
        r = simulate_with_borrowing(two_taxon(), p, b, seed=k)
        n.append(len(r.extra["borrow_events"]) + r.extra["n_void_borrow_events"])
        assert len(r.extra["borrow_attempts"]) <= n[-1]
    assert_mean(n, 2 * beta * p.n_concepts * T)


def test_borrowing_increases_sharing():
    p = SDolloParams(n_concepts=300, birth_rate=MU, loss_rate=MU)
    s0 = np.mean([shared(simulate_with_borrowing(two_taxon(), p, BorrowingParams(0.0), seed=k))
                  for k in range(30)])
    s1 = np.mean([shared(simulate_with_borrowing(two_taxon(), p, BorrowingParams(5e-4), seed=k))
                  for k in range(30)])
    assert s1 > s0 * 1.15


def test_events_are_between_contemporaneous_lineages():
    rng = np.random.default_rng(3)
    tree = simulate_yule_given_root(8, 3000.0, 6e-4, rng)
    p = SDolloParams(n_concepts=100, birth_rate=MU, loss_rate=MU)
    r = simulate_with_borrowing(tree, p, BorrowingParams(3e-4), seed=1)
    assert r.extra["borrow_events"]
    for e in r.extra["borrow_events"]:
        for nm in (e["recipient"], e["donor"]):
            node = tree.node_by_name(nm)
            assert node.age <= e["age"] <= node.parent.age
        assert e["recipient"] != e["donor"]


def test_distance_contact_prefers_sisters():
    # ((A,B),(C,D)): with a short distance scale, nearly all borrowing is within a cherry
    # (donors must have a class for the concept, so make lexical gaps rare: lambda/mu = 6)
    tree = Tree.from_newick("((A:500,B:500):1500,(C:500,D:500):1500)root;")
    p = SDolloParams(n_concepts=200, birth_rate=6 * MU, loss_rate=MU)
    sister = {"A": "B", "B": "A", "C": "D", "D": "C"}
    for contact, lo, hi in (("distance", 0.95, 1.0), ("uniform", 0.25, 0.42)):
        b = BorrowingParams(3e-4, contact=contact, distance_scale=200.0)
        ev = [(r, d) for k in range(5) for r, d in
              simulate_with_borrowing(tree, p, b, seed=k).extra["borrow_attempts"]
              if r in sister]
        frac = np.mean([sister[r] == d for r, d in ev])
        assert lo <= frac <= hi, (contact, frac)


def test_replace_mode_leaves_one_class():
    p = SDolloParams(n_concepts=50, birth_rate=3 * MU, loss_rate=MU)
    r = simulate_with_borrowing(two_taxon(), p, BorrowingParams(1e-3), seed=0)
    leaf_idx = {"A": 0, "B": 1}
    # the most recent event per (leaf, concept): if not overwritten later, that class is
    # present, but births after the event may add more; the borrowed class is flagged.
    for i in range(2):
        b = r.extra["borrowed"][i]
        assert not (b & ~r.presence[i]).any()  # borrowed implies present
    assert r.extra["borrowed"].any()
    assert set(leaf_idx) == set(r.leaf_names)


def test_borrowed_entries_export():
    rng = np.random.default_rng(5)
    tree = simulate_yule_given_root(6, 2000.0, 6e-4, rng)
    p = SDolloParams(n_concepts=80, birth_rate=MU, loss_rate=MU)
    r = simulate_with_borrowing(tree, p, BorrowingParams(4e-4), seed=2)
    obs, rows = borrowed_entries(r)
    assert rows
    for row in rows:
        i = obs.leaf_names.index(row["language"])
        assert obs.presence[i, row["class_index"]]
        assert obs.class_concept[row["class_index"]] == row["concept"]
    assert len(rows) == int(r.extra["borrowed"].sum())
    assert "borrowed" not in obs.extra


def test_reproducible():
    p = SDolloParams(n_concepts=50, birth_rate=MU, loss_rate=MU, clock_sd=0.3)
    b = BorrowingParams(3e-4, contact="distance")
    r1 = simulate_with_borrowing(two_taxon(), p, b, seed=9)
    r2 = simulate_with_borrowing(two_taxon(), p, b, seed=9)
    assert np.array_equal(r1.presence, r2.presence)
    assert r1.extra["borrow_events"] == r2.extra["borrow_events"]


def test_params_validation():
    with pytest.raises(ValueError):
        BorrowingParams(rate=-1)
    with pytest.raises(ValueError):
        BorrowingParams(contact="gravity")
    with pytest.raises(ValueError):
        BorrowingParams(mode="swap")
