"""Tests for phoneme-string simulation and its coupling to the borrowing simulator."""

import numpy as np
import pytest

from sim.borrowing import BorrowingParams, simulate_with_borrowing
from sim.phonology import (INVENTORY, OUTCOMES, LineagePhonology, PhonologyParams, SoundChange,
                           random_protoform, sporadic_change)
from sim.sdollo import SDolloParams
from sim.tree import Tree, simulate_yule_given_root

MU = 3e-4


def test_rules_and_contexts():
    assert SoundChange("t", "d", "intervocalic").apply("tata") == "tada"
    assert SoundChange("t", "s", "initial").apply("tata") == "sata"
    assert SoundChange("k", "", "final").apply("kak") == "ka"
    assert SoundChange("k", "c", "before_front").apply("kika") == "cika"
    assert SoundChange("a", "o", "any").apply("tata") == "toto"
    assert SoundChange("h", "", "any").apply("h") == "h"  # never empties a word


def test_outcomes_stay_in_inventory():
    assert set(OUTCOMES) == set(INVENTORY)
    for x, ys in OUTCOMES.items():
        assert all(len(y) <= 1 and (y == "" or y in INVENTORY) for y in ys)


def test_protoform_shape():
    rng = np.random.default_rng(0)
    p = PhonologyParams()
    for _ in range(200):
        f = random_protoform(p, rng)
        assert 4 <= len(f) <= 7 and set(f) <= set(INVENTORY)
        assert sporadic_change(f, rng) != "" or len(f) == 0


def test_schedule_rate():
    rng = np.random.default_rng(1)
    p = PhonologyParams(change_rate=1 / 100, sporadic_rate=0.0)
    n = [len(LineagePhonology(1000.0, 0.0, p, 10, rng).events) for _ in range(400)]
    assert abs(np.mean(n) - 10) < 4 * np.sqrt(10 / 400)


def test_cognate_data_unchanged_by_phonology():
    rng = np.random.default_rng(2)
    tree = simulate_yule_given_root(6, 2500.0, 6e-4, rng)
    p = SDolloParams(n_concepts=80, birth_rate=MU, loss_rate=MU)
    b = BorrowingParams(3e-4)
    r0 = simulate_with_borrowing(tree, p, b, seed=4)
    r1 = simulate_with_borrowing(tree, p, b, seed=4, phonology=PhonologyParams())
    assert np.array_equal(r0.presence, r1.presence)
    assert r0.extra["borrow_events"] == r1.extra["borrow_events"]
    assert r0.forms is None
    assert ((r1.forms != None) == r1.presence).all()  # noqa: E711


def test_regular_change_hits_all_words_of_a_lineage():
    # one lineage, no births/losses to speak of: every rule on A applies to all its words
    tree = Tree.from_newick("(A:1000,B:1000)root;")
    p = SDolloParams(n_concepts=60, birth_rate=1e-9, loss_rate=1e-9)
    ph = PhonologyParams(change_rate=1 / 100, sporadic_rate=0.0)
    r = simulate_with_borrowing(tree, p, BorrowingParams(0.0), seed=3, phonology=ph)
    assert r.extra["sound_changes"]["A"]
    # with no sporadic change, two words identical at the root stay identical in A
    # and forms in A and B differ only by regular changes: same-class words differ
    diff = np.mean([r.forms[0, j] != r.forms[1, j] for j in range(r.presence.shape[1])
                    if r.presence[:, j].all()])
    assert diff > 0.5


def test_borrowed_form_copies_donor_and_misses_own_changes():
    # A borrows from B late; A's earlier sound changes must not appear in the loan
    # (enough cognate turnover that A and B often have different classes to borrow)
    tree = Tree.from_newick("(A:2000,B:2000)root;")
    p = SDolloParams(n_concepts=150, birth_rate=4e-4, loss_rate=4e-4)
    ph = PhonologyParams(change_rate=1 / 150, sporadic_rate=0.0)
    r = simulate_with_borrowing(tree, p, BorrowingParams(1e-4), seed=11, phonology=ph)
    ev = r.extra["borrow_events"]
    assert ev
    # loans with no later sound change on either side are exact copies of the donor's form
    for e in ev:
        later_rec = [a for a, _ in r.extra["sound_changes"][e["recipient"]] if a < e["age"]]
        later_don = [a for a, _ in r.extra["sound_changes"][e["donor"]] if a < e["age"]]
        i_rec = r.leaf_names.index(e["recipient"]); i_don = r.leaf_names.index(e["donor"])
        if not later_rec and not later_don and r.presence[i_rec, e["class"]] and r.presence[i_don, e["class"]]:
            assert r.forms[i_rec, e["class"]] == r.forms[i_don, e["class"]]
    # and loans are more similar to the donor's form than inherited cognates are
    b = r.extra["borrowed"]
    shared = r.presence.all(axis=0)
    same_loan = [r.forms[0, j] == r.forms[1, j] for j in np.flatnonzero(shared & b.any(axis=0))]
    same_inh = [r.forms[0, j] == r.forms[1, j] for j in np.flatnonzero(shared & ~b.any(axis=0))]
    assert np.mean(same_loan) > np.mean(same_inh) + 0.3


def test_cldf_export_uses_forms(tmp_path):
    from pipeline.cldf import write_simulation
    rng = np.random.default_rng(5)
    tree = simulate_yule_given_root(5, 1500.0, 6e-4, rng)
    p = SDolloParams(n_concepts=30, birth_rate=MU, loss_rate=MU)
    r = simulate_with_borrowing(tree, p, BorrowingParams(2e-4), seed=1, phonology=PhonologyParams())
    r.extra = {k: v for k, v in r.extra.items() if k not in ("borrowed", "borrow_attempts")}
    write_simulation(r, tmp_path)
    import csv
    vals = [row["Form"] for row in csv.DictReader(open(tmp_path / "forms.csv"))]
    assert vals and all(set(v) <= set(INVENTORY) for v in vals)


def test_params_validation():
    with pytest.raises(ValueError):
        PhonologyParams(change_rate=-1)
    with pytest.raises(ValueError):
        PhonologyParams(min_syllables=3, max_syllables=2)


def test_stable_segments_rarely_change():
    from sim.phonology import draw_sound_change
    rng = np.random.default_rng(0)
    forms = ["hama", "kana", "pata", "tapa"]
    p = PhonologyParams(stable_segments="hmnk" + "a", stable_weight=0.0)
    targets = {draw_sound_change(forms, rng, p).target for _ in range(200)}
    assert targets <= {"p", "t"}
    # default (no stable segments) is unchanged behaviour: any segment can be targeted
    targets = {draw_sound_change(forms, rng).target for _ in range(400)}
    assert {"h", "a"} <= targets
