"""Draw simulation replicates from a prior spec and render matching BEAST2 XML.

The prior spec (e.g. ``infer/beast/m1_prior.json``) is the single source of
truth: ``draw_truth`` samples every generating quantity from it, and
``render_xml`` writes the same distributions into the XML as priors. If the
two ever diverge, the well-calibrated coverage test will detect it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pipeline.cldf import MISSING, CognateMatrix, read_cognate_matrix, write_simulation
from sim.sdollo import SDolloParams, simulate
from sim.tree import Tree, simulate_yule_given_root

BEAST_DIR = Path(__file__).resolve().parent / "beast"


def load_prior(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def _draw(spec: dict, rng: np.random.Generator) -> float:
    d = spec["dist"]
    if d == "normal":
        while True:  # truncate to positive values (negligible mass for sensible specs)
            x = rng.normal(spec["mean"], spec["sd"])
            if x > 0:
                return float(x)
    if d == "lognormal":
        return float(rng.lognormal(spec["meanlog"], spec["sdlog"]))
    if d == "exponential":
        return float(rng.exponential(spec["mean"]))
    raise ValueError(f"unknown distribution {d!r}")


def is_relaxed(prior: dict) -> bool:
    """Relaxed clock iff the spec gives a prior on the clock sd; otherwise strict (rate 1)."""
    return "clock_sd" in prior


def prior_mean(spec: dict) -> float:
    d = spec["dist"]
    if d in ("normal", "exponential"):
        return float(spec["mean"])
    if d == "lognormal":
        return float(np.exp(spec["meanlog"] + spec["sdlog"] ** 2 / 2))
    raise ValueError(d)


def draw_truth(prior: dict, rng: np.random.Generator) -> dict:
    """Draw tree and model parameters from the prior."""
    root_age = _draw(prior["root_age"], rng)
    tree = simulate_yule_given_root(prior["n_tips"], root_age, prior["yule_birth_rate"], rng)
    return {
        "tree": tree,
        "root_age": root_age,
        "loss_rate": _draw(prior["loss_rate"], rng),
        "clock_sd": _draw(prior["clock_sd"], rng) if is_relaxed(prior) else 0.0,
    }


def simulate_replicate(prior: dict, seed: int, outdir: Path | str) -> dict:
    """Draw from the prior, simulate cognate data, write CLDF + truth + XML.

    Returns the truth record. Everything is reproducible from ``seed``.
    """
    outdir = Path(outdir)
    tree_ss, char_ss = np.random.SeedSequence(seed).spawn(2)
    truth = draw_truth(prior, np.random.default_rng(tree_ss))
    mu = truth["loss_rate"]
    params = SDolloParams(
        n_concepts=prior["n_concepts"],
        birth_rate=mu * prior["classes_per_concept"],
        loss_rate=mu,
        gamma_shape=None,
        clock_sd=truth["clock_sd"],
    )
    res = simulate(truth["tree"], params, rng=np.random.default_rng(char_ss))
    res.seed = seed
    res.extra["tree_length"] = truth["tree"].tree_length()
    md = write_simulation(res, outdir)
    cm = read_cognate_matrix(md, concept_level_missing=False)
    xml = render_xml(cm, prior, stem="beast")
    (outdir / "beast.xml").write_text(xml)
    rec = json.loads((outdir / "truth.json").read_text())
    rec["n_cognatesets"] = len(cm.cognatesets)
    (outdir / "truth.json").write_text(json.dumps(rec, indent=2))
    return rec


def render_xml(cm: CognateMatrix, prior: dict, stem: str,
               template: str = "sdollo.xml.j2", mcmc: dict | None = None) -> str:
    if (cm.matrix == MISSING).any():
        raise ValueError("this template expects no missing data")
    env = Environment(loader=FileSystemLoader(BEAST_DIR), undefined=StrictUndefined,
                      keep_trailing_newline=True)
    seqs = [(lang, "".join(str(int(v)) for v in row)) for lang, row in zip(cm.languages, cm.matrix)]
    init = {
        "root_age": prior_mean(prior["root_age"]),
        "loss_rate": prior_mean(prior["loss_rate"]),
        "clock_sd": prior_mean(prior["clock_sd"]) if is_relaxed(prior) else 0.0,
    }
    return env.get_template(template).render(
        sequences=seqs, prior=prior, mcmc=mcmc or prior["mcmc"], init=init, stem=stem,
        relaxed=is_relaxed(prior))


def true_tree(rep_dir: Path | str) -> Tree:
    return Tree.from_newick((Path(rep_dir) / "tree.nwk").read_text())
