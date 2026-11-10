import json
from pathlib import Path

import numpy as np
import pytest
from pycldf import Wordlist

from pipeline.cldf import MISSING, read_cognate_matrix, write_simulation
from sim.sdollo import SDolloParams, simulate
from sim.tree import Tree, simulate_yule

ROOT = Path(__file__).resolve().parents[1]
KAUFMAN = ROOT / "data" / "raw" / "kaufmanpmed" / "cldf" / "cldf-metadata.json"


@pytest.fixture
def sim_result():
    tree = simulate_yule(7, 3000.0, np.random.default_rng(4))
    p = SDolloParams(n_concepts=60, birth_rate=4e-4, loss_rate=8e-4, gamma_shape=0.8, clock_sd=0.3)
    return simulate(tree, p, seed=21)


def test_written_dataset_is_valid_cldf(sim_result, tmp_path):
    md = write_simulation(sim_result, tmp_path)
    ds = Wordlist.from_metadata(md)
    assert ds.validate()


def test_roundtrip_matrix_matches_simulation(sim_result, tmp_path):
    md = write_simulation(sim_result, tmp_path)
    cm = read_cognate_matrix(md)
    obs = sim_result.observed()
    assert cm.languages == obs.leaf_names
    assert cm.shape == obs.presence.shape
    assert np.array_equal(cm.matrix, obs.binary_matrix())
    assert (cm.matrix == MISSING).any()


def test_truth_sidecar(sim_result, tmp_path):
    write_simulation(sim_result, tmp_path)
    truth = json.loads((tmp_path / "truth.json").read_text())
    assert truth["seed"] == 21
    assert truth["root_age"] == pytest.approx(3000.0)
    tree = Tree.from_newick((tmp_path / "tree.nwk").read_text())
    assert tree.root_age == pytest.approx(3000.0)
    assert set(truth["branch_rates"]) == {n.name for n in tree.branches()}


def test_nexus_export(sim_result, tmp_path):
    md = write_simulation(sim_result, tmp_path / "ds")
    cm = read_cognate_matrix(md)
    cm.to_nexus(tmp_path / "m.nex")
    text = (tmp_path / "m.nex").read_text()
    assert f"NTAX={len(cm.languages)} NCHAR={len(cm.cognatesets)}" in text
    assert "?" in text


@pytest.mark.data
@pytest.mark.skipif(not KAUFMAN.exists(), reason="run `make data` first")
def test_kaufman_mayan_loads():
    cm = read_cognate_matrix(KAUFMAN, concept_level_missing=False)
    assert len(cm.languages) > 20
    assert len(cm.cognatesets) > 1000
    assert set(np.unique(cm.matrix)) <= {0, 1}
    assert (cm.matrix.sum(axis=0) >= 1).all()
