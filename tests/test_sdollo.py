"""Tests for the SDollo simulator, including analytic checks on a two-taxon tree.

On a two-taxon tree with root age T (both tips at age 0), a class is shared
by both tips iff it was present at the root and survived both branches;
classes born on a branch can only reach one tip. With the root at
stationarity (Poisson(lambda/mu) classes per concept), the expected number of
shared classes is

    C * (lambda/mu) * E[ exp(-mu*T*r_c*(rho_1 + rho_2)) ]

which reduces to C*(lambda/mu)*exp(-2 mu T) under a strict clock with no
concept-rate variation, to the discrete-gamma average across categories, to
(1 + 2 mu T / alpha)^(-alpha) for a continuous gamma, and to
E[exp(-mu T rho)]^2 under an uncorrelated relaxed clock.
"""

import numpy as np
import pytest
from scipy import integrate, stats

from sim.sdollo import MISSING, SDolloParams, discrete_gamma_rates, simulate
from sim.tree import Tree, simulate_yule

T = 1000.0
LAM = MU = 5e-4  # mu*T = 0.5 -> exp(-2 mu T) = e^-1


def two_taxon():
    return Tree.from_newick(f"(A:{T},B:{T})root;")


def shared_and_per_leaf(res):
    obs = res.observed()
    shared = int((obs.presence[0] & obs.presence[1]).sum())
    per_leaf = obs.presence.sum(axis=1)
    return shared, per_leaf


def run_reps(params, n_reps, seed0=0):
    tree = two_taxon()
    shared, leaf = [], []
    for k in range(n_reps):
        s, pl = shared_and_per_leaf(simulate(tree, params, seed=seed0 + k))
        shared.append(s)
        leaf.append(pl.mean())
    return np.array(shared, float), np.array(leaf, float)


def assert_mean(samples, expected, z=4.0):
    se = samples.std(ddof=1) / np.sqrt(len(samples))
    assert abs(samples.mean() - expected) < z * se, (
        f"mean {samples.mean():.3f} vs expected {expected:.3f} (se {se:.3f})")


# ---------------------------------------------------------------------- analytic


def test_shared_classes_strict_clock_no_heterogeneity():
    p = SDolloParams(n_concepts=1000, birth_rate=LAM, loss_rate=MU)
    shared, leaf = run_reps(p, 60)
    assert_mean(shared, 1000 * (LAM / MU) * np.exp(-2 * MU * T))
    assert_mean(leaf, 1000 * LAM / MU)
    # strict clock, no heterogeneity -> shared count is exactly Poisson
    lam_hat = shared.mean()
    assert shared.var(ddof=1) == pytest.approx(lam_hat, rel=0.5)


def test_shared_classes_unequal_rates():
    # birth != death: stationary mean lambda/mu = 3
    p = SDolloParams(n_concepts=500, birth_rate=1.5e-3, loss_rate=5e-4)
    shared, leaf = run_reps(p, 40, seed0=100)
    assert_mean(shared, 500 * 3 * np.exp(-2 * 5e-4 * T))
    assert_mean(leaf, 500 * 3)


def test_shared_classes_discrete_gamma():
    alpha, k = 0.5, 4
    p = SDolloParams(n_concepts=1000, birth_rate=LAM, loss_rate=MU,
                     gamma_shape=alpha, gamma_categories=k)
    shared, leaf = run_reps(p, 60, seed0=200)
    rates = discrete_gamma_rates(alpha, k)
    expected = 1000 * (LAM / MU) * np.mean(np.exp(-2 * MU * T * rates))
    assert_mean(shared, expected)
    assert_mean(leaf, 1000 * LAM / MU)


def test_shared_classes_continuous_gamma():
    alpha = 0.8
    p = SDolloParams(n_concepts=1000, birth_rate=LAM, loss_rate=MU,
                     gamma_shape=alpha, gamma_categories=0)
    shared, _ = run_reps(p, 60, seed0=300)
    expected = 1000 * (LAM / MU) * (1 + 2 * MU * T / alpha) ** (-alpha)
    assert_mean(shared, expected)


def test_shared_classes_relaxed_clock():
    sd = 0.6
    p = SDolloParams(n_concepts=200, birth_rate=LAM, loss_rate=MU, clock_sd=sd)
    shared, _ = run_reps(p, 600, seed0=400)
    ln = stats.lognorm(s=sd, scale=np.exp(-0.5 * sd**2))
    laplace, _ = integrate.quad(lambda r: np.exp(-MU * T * r) * ln.pdf(r), 0, np.inf)
    expected = 200 * (LAM / MU) * laplace**2
    assert_mean(shared, expected)


def test_stationarity_on_yule_tree():
    # Every leaf, whatever its path from the root, has Poisson(lambda/mu) classes per concept.
    rng = np.random.default_rng(5)
    tree = simulate_yule(8, root_age=3000.0, rng=rng)
    p = SDolloParams(n_concepts=300, birth_rate=1e-3, loss_rate=5e-4,
                     gamma_shape=1.0, clock_sd=0.3)
    per_leaf = np.concatenate([
        simulate(tree, p, seed=500 + k).presence.sum(axis=1) for k in range(30)
    ]).astype(float)
    assert_mean(per_leaf, 300 * 2.0)


# ---------------------------------------------------------------------- structural


def test_dollo_property():
    tree = simulate_yule(12, 2000.0, np.random.default_rng(3))
    res = simulate(tree, SDolloParams(n_concepts=100, birth_rate=1e-3, loss_rate=1e-3,
                                      gamma_shape=0.7, clock_sd=0.4), seed=11)
    leaf_idx = {n: i for i, n in enumerate(res.leaf_names)}
    for j in range(res.n_classes):
        birth = tree.node_by_name(res.class_birth_node[j])
        desc = {l.name for l in _subtree_leaves(birth)}
        present = {res.leaf_names[i] for i in np.flatnonzero(res.presence[:, j])}
        assert present <= desc
        # birth time lies on the branch above the birth node (or at the root)
        if birth.is_root:
            assert res.class_birth_age[j] == pytest.approx(birth.age)
        else:
            assert birth.age - 1e-9 <= res.class_birth_age[j] <= birth.parent.age + 1e-9
    assert set(leaf_idx) == set(tree.leaf_names)


def _subtree_leaves(node):
    if node.is_leaf:
        return [node]
    return [l for c in node.children for l in _subtree_leaves(c)]


def test_observed_drops_only_all_absent_classes():
    tree = simulate_yule(5, 5000.0, np.random.default_rng(0))
    res = simulate(tree, SDolloParams(n_concepts=50, birth_rate=1e-3, loss_rate=1e-3), seed=1)
    obs = res.observed()
    assert obs.presence.any(axis=0).all()
    assert obs.n_classes + obs.extra["n_unobserved_dropped"] == res.n_classes


def test_binary_matrix_missing_coding():
    tree = simulate_yule(6, 4000.0, np.random.default_rng(2))
    # low lambda/mu -> many empty concept slots
    res = simulate(tree, SDolloParams(n_concepts=80, birth_rate=2e-4, loss_rate=1e-3), seed=3).observed()
    m = res.binary_matrix()
    counts = res.concept_counts()
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            c = res.class_concept[j]
            if counts[i, c] == 0:
                assert m[i, j] == MISSING
            else:
                assert m[i, j] == int(res.presence[i, j])
    assert (m == MISSING).any()


def test_reproducible_from_seed():
    tree = simulate_yule(6, 1000.0, np.random.default_rng(9))
    p = SDolloParams(n_concepts=40, gamma_shape=0.5, clock_sd=0.5, birth_rate=1e-3, loss_rate=1e-3)
    a, b = simulate(tree, p, seed=42), simulate(tree, p, seed=42)
    assert np.array_equal(a.presence, b.presence)
    assert a.branch_rates == b.branch_rates
    c = simulate(tree, p, seed=43)
    assert not (a.presence.shape == c.presence.shape and np.array_equal(a.presence, c.presence))


# ---------------------------------------------------------------------- rates


def test_discrete_gamma_matches_yang_1994():
    # Yang (1994) Table: alpha = 0.5, k = 4 mean-rate categories
    r = discrete_gamma_rates(0.5, 4)
    assert r == pytest.approx([0.0334, 0.2519, 0.8203, 2.8944], abs=1e-3)
    assert r.mean() == pytest.approx(1.0)


def test_branch_rates_mean_one():
    tree = simulate_yule(200, 1.0, np.random.default_rng(0))
    res = simulate(tree, SDolloParams(n_concepts=1, clock_sd=0.5, birth_rate=1, loss_rate=1), seed=0)
    rates = np.array(list(res.branch_rates.values()))
    assert rates.mean() == pytest.approx(1.0, abs=4 * rates.std() / np.sqrt(len(rates)))


def test_params_validation():
    with pytest.raises(ValueError):
        SDolloParams(birth_rate=0)
    with pytest.raises(ValueError):
        SDolloParams(clock_sd=-1)
    with pytest.raises(ValueError):
        SDolloParams(gamma_shape=0)
