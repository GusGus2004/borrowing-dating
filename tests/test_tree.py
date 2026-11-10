import numpy as np
import pytest

from sim.tree import Tree, simulate_yule


def test_newick_roundtrip_preserves_ages():
    nwk = "((A:1,B:1)AB:2,(C:2.5,D:2.5)CD:0.5)root;"
    t = Tree.from_newick(nwk)
    assert t.root_age == pytest.approx(3.0)
    assert t.node_by_name("AB").age == pytest.approx(1.0)
    assert t.node_by_name("CD").age == pytest.approx(2.5)
    assert all(l.age == 0 for l in t.leaves())
    t2 = Tree.from_newick(t.to_newick())
    for a, b in zip(t.preorder(), t2.preorder()):
        assert a.name == b.name
        assert a.age == pytest.approx(b.age)


def test_newick_autonames_internal_nodes_and_quotes():
    t = Tree.from_newick("(('Q an':1,B:1):1,C:2);")
    names = {n.name for n in t.preorder()}
    assert "Q an" in names
    assert len(names) == 5
    assert "'Q an'" in t.to_newick()
    assert Tree.from_newick(t.to_newick()).leaf_names == ["Q an", "B", "C"]


def test_newick_with_tip_ages_and_comments():
    t = Tree.from_newick("(A[&rate=1]:1,B:2);", tip_ages={"A": 1.0, "B": 0.0})
    assert t.root_age == pytest.approx(2.0)
    assert t.node_by_name("A").age == pytest.approx(1.0)
    with pytest.raises(ValueError):
        Tree.from_newick("(A:1,B:2);", tip_ages={"A": 0.0, "B": 0.0})


def test_tree_rejects_bad_input():
    with pytest.raises(ValueError):
        Tree.from_newick("(A:1,B:1)")
    with pytest.raises(ValueError):
        Tree.from_newick("(A:1,:1);")


def test_mrca_and_length():
    t = Tree.from_newick("((A:1,B:1)AB:2,C:3)R;")
    assert t.mrca(["A", "B"]).name == "AB"
    assert t.mrca(["A", "C"]).name == "R"
    assert t.tree_length() == pytest.approx(1 + 1 + 2 + 3)


@pytest.mark.parametrize("n", [2, 5, 20])
def test_yule_shape(n):
    rng = np.random.default_rng(1)
    t = simulate_yule(n, root_age=4000.0, rng=rng)
    assert len(t.leaves()) == n
    assert len(t.internal_nodes()) == n - 1
    assert t.root_age == pytest.approx(4000.0)
    assert all(l.age == 0 for l in t.leaves())
    assert all(b.branch_length > 0 for b in t.branches())
    t.validate()


def test_yule_reproducible():
    a = simulate_yule(10, 1.0, np.random.default_rng(7)).to_newick()
    b = simulate_yule(10, 1.0, np.random.default_rng(7)).to_newick()
    assert a == b
