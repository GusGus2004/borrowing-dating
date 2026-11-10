# Borrowing-aware Bayesian dating of language families

Quantifying and correcting the effect of lexical borrowing on Bayesian
divergence-time estimates, with a simulation benchmark and a Mayan case study.
See [`brief.md`](brief.md) for the full design.

## Status

**Phase 1: null-model simulator and data pipeline (no borrowing yet).**

| Component | State |
|---|---|
| `sim/tree.py`: time-calibrated trees, Newick I/O, Yule simulation | done, tested |
| `sim/sdollo.py`: stochastic Dollo cognate birth/loss, discrete/continuous gamma across concepts, uncorrelated lognormal relaxed clock | done, tested (analytic checks) |
| `pipeline/cldf.py`: CLDF Wordlist export of simulations + ground truth; CLDF → cognate matrix (CSV/NEXUS) for simulated *and* real data | done, tested (pycldf validation, round trip) |
| Mayan data (Lexibank `kaufmanpmed`) → cognate matrix | done |
| BEAST2 pinned install + SDollo smoke test (`make beast-smoke`) | done: Babel's SDollo example runs to completion |
| BEAST2 recovery of simulated ages / HPD coverage (milestone 1) | next |
| Borrowing, phonology, pair HMM, Track B | not started (by design) |

## Quick start

```bash
make env      # Python venv from env/requirements.lock.txt
make data     # fetch raw CLDF datasets at pinned commits into data/raw/
make matrices # Mayan cognate matrix -> data/derived/kaufmanpmed/
make sim      # example simulated dataset -> data/derived/sim/example/
make test     # pytest
make beast    # install BEAST2 + pinned packages into env/ (project-local)
make beast-smoke  # short SDollo run to check BEAST2 works (~4 min)
```

`make` with no target runs env, data, matrices, sim and test.

Simulate a custom dataset:

```bash
.venv/bin/python scripts/simulate.py --out data/derived/sim/run1 --seed 7 \
    --n-tips 12 --root-age 4000 --n-concepts 200 --birth 2e-4 --loss 2e-4 \
    --gamma-shape 0.5 --clock-sd 0.3
```

Each simulated dataset directory contains a valid CLDF Wordlist
(`Wordlist-metadata.json`, `forms.csv`, `cognates.csv`, …), `tree.nwk` (the true
tree) and `truth.json` (seed, parameters, node ages, per-branch clock rates,
per-concept rates).

## Model (as simulated)

Per concept *c* and branch *b*, cognate classes are born at rate
λ·r<sub>c</sub>·ρ<sub>b</sub> and each is lost at rate μ·r<sub>c</sub>·ρ<sub>b</sub>;
a class is born once and never re-born (Dollo). Here r<sub>c</sub> ~ Gamma(α, 1/α)
(optionally discretised into *k* categories, Yang 1994 mean method) and
ρ<sub>b</sub> ~ LogNormal(−σ²/2, σ²) (mean 1). The root is at stationarity:
Poisson(λ/μ) classes per concept. Simulation is exact in continuous time.

A language with no class for a concept is coded missing (`?`) for every class
of that concept, because absence is then a lexical gap rather than evidence.

### Validation

`tests/test_sdollo.py` checks the simulator against closed-form expectations on a
two-taxon tree of age *T*: the expected number of shared classes is
C·(λ/μ)·E[exp(−μT·r<sub>c</sub>(ρ<sub>1</sub>+ρ<sub>2</sub>))], tested under a strict
clock, unequal λ≠μ, discrete gamma, continuous gamma and a relaxed clock (the
lognormal Laplace transform is integrated numerically). Stationarity is checked
on a Yule tree. Structural tests check the Dollo property (every class sits only
in descendants of its birth branch), missing-data coding and seed
reproducibility. As a power check, a 5% error in the loss rate moves the
shared-class statistic by about 15 standard errors.

## BEAST2

`env/install_beast2.sh` installs the BEAST 2.7.7 macOS/Linux bundle (launcher +
JRE; no system Java needed) into `env/beast2/`, and the CBAN packages into
`env/beast2-packages/`. These versions are pinned exactly: BEAST.base 2.7.8, BEAST.app 2.7.8,
BEASTLabs 2.0.3, Babel 0.4.2. Babel requires BEAST.base 2.7.8, so the effective
core is 2.7.8 running on the 2.7.7 launcher. Always pass
`-packagedir env/beast2-packages` (the Makefile does).

Timing note: the shortened SDollo example (24 taxa, 2×10⁵ states) takes about 225 s
on an Apple-silicon laptop. That is the cost that motivates Track B.

## Data

| Dataset | Source | Pinned commit | Notes |
|---|---|---|---|
| `kaufmanpmed` | [lexibank/kaufmanpmed](https://github.com/lexibank/kaufmanpmed) — Kaufman & Justeson (2003) *A Preliminary Mayan Etymological Dictionary*, CC-BY-4.0 | `a774237` | 104 non-proto varieties × 3,030 cognate sets |

**Caveat on the Mayan data.** Lexibank has no Swadesh-style Mayan wordlist with
expert cognate coding; `kaufmanpmed` is the only Mayan cognate dataset there.
It is an *etymological dictionary*: its "concepts" are 5,932 language-specific
glosses (often Spanish), not a shared comparative concept list, so a missing
cognate set means "not recorded", not "lost". It is built here with
`--no-concept-missing` (absence = 0) only to exercise the pipeline. It is not
suitable for dating as it stands. Before milestone 5 we need to either
(a) map a basic-vocabulary subset onto Concepticon and treat absence there as
informative, or (b) find a dedicated Mayan basic-vocabulary dataset (for example
from the ASJP/Wichmann line or published Mayan phylogenetic studies). The
dataset's `Loan` column will be useful later as partial ground truth for
borrowing.

Raw data under `data/raw/` is fetched by `scripts/fetch_data.sh`, made
read-only, and never edited. Everything under `data/derived/` is regenerated
by `make`.

## Layout

```
env/          pinned Python lockfile; BEAST2 installer
infer/beast/  BEAST2 XML (smoke/ = shortened Babel SDollo example)
data/raw/     fetched CLDF datasets (git-ignored, pinned commits)
data/derived/ cognate matrices, simulated datasets (git-ignored, rebuilt by make)
sim/          forward simulator: tree.py, sdollo.py (borrowing.py, phonology.py to come)
pipeline/     CLDF I/O shared by real and simulated data
scripts/      CLI entry points used by the Makefile
infer/cpp/    Track B (later)
hmm/          pair HMM (milestone 3)
analysis/     bias curves, ROC, figures
tests/
figures/
```
