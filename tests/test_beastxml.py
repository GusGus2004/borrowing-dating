import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from infer.beastxml import draw_truth, is_relaxed, load_prior, simulate_replicate

BEAST = Path(__file__).resolve().parents[1] / "infer" / "beast"


@pytest.mark.parametrize("clock", ["strict", "relaxed"])
def test_replicate_xml_is_well_formed(clock, tmp_path):
    prior = load_prior(BEAST / f"m1_{clock}_prior.json")
    rec = simulate_replicate(prior, 7, tmp_path)
    root = ET.parse(tmp_path / "beast.xml").getroot()
    seqs = root.findall("./data/sequence")
    assert len(seqs) == prior["n_tips"]
    assert {len(s.get("value")) for s in seqs} == {rec["n_cognatesets"]}
    assert all(set(s.get("value")) <= {"0", "1"} for s in seqs)
    text = (tmp_path / "beast.xml").read_text()
    assert ("UCRelaxedClockModel" in text) == (clock == "relaxed")
    assert ("babelfix.AnyTipObservationProcessRC" in text) == (clock == "relaxed")
    assert 'conditionalOnRoot="true"' in text


def test_draw_truth_matches_spec():
    prior = load_prior(BEAST / "m1_relaxed_prior.json")
    rng = np.random.default_rng(0)
    draws = [draw_truth(prior, rng) for _ in range(4000)]
    roots = np.array([d["root_age"] for d in draws])
    assert roots.mean() == pytest.approx(prior["root_age"]["mean"], abs=4 * prior["root_age"]["sd"] / 63)
    mus = np.log([d["loss_rate"] for d in draws])
    assert mus.mean() == pytest.approx(prior["loss_rate"]["meanlog"], abs=0.03)
    assert all(d["tree"].root_age == d["root_age"] for d in draws[:50])
    assert is_relaxed(prior) and not is_relaxed(load_prior(BEAST / "m1_strict_prior.json"))


def test_replicate_reproducible(tmp_path):
    prior = load_prior(BEAST / "m1_strict_prior.json")
    simulate_replicate(prior, 3, tmp_path / "a")
    simulate_replicate(prior, 3, tmp_path / "b")
    assert (tmp_path / "a/beast.xml").read_text() == (tmp_path / "b/beast.xml").read_text()
