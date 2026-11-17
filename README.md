# Borrowing-aware Bayesian dating of language families

Quantifying and correcting the effect of lexical borrowing on Bayesian
divergence-time estimates, with a simulation benchmark and a Mayan case study.
The full project design is kept privately.

## Status

| Milestone | State |
|---|---|
| 0. Environment, CLDF pipeline, Mayan data, BEAST2 | done |
| 1. Simulator (no borrowing) + BEAST2 calibration | **strict clock: done** (100 replicates, below). Relaxed clock: `babelfix` written and passes the likelihood debug check; coverage study not yet run (`make m1-relaxed`) |
| 2. Bias curve vs borrowing rate, two contact models | **done** (below), plus 2b: under a realistic calibration design (vague root, recent calibrations) the root bias is 30–50% |
| 3. Phoneme strings + pair HMM, ROC | **done** (below); 3b/3c: level-3 detector (forms + tree) raises AUC from 0.61–0.67 to 0.77–0.80 with correct cognates; needs 100–200 concepts with automated cognates |
| 4. Latent-donor correction | **done** (below): masking the true loans removes the bias; masking detector flags barely helps. Latent-donor (level 2) model does not beat level 1 |
| 5. Mayan application | **runs done** (below). ASJP 40 with LexStat cognates gives an implausibly young root; 5b: PMED with Kaufman's etyma gives c. 2800 BC under the same calibrations, and 5c: on the same ASJP words, expert vs LexStat cognate coding alone moves the root by 1,800–4,000 years. LexStat over-splits real Mayan data far more than simulated data (6). Draft calibrations in `infer/beast/m5_mayan_prior.json` are **awaiting review**. Do not quote these dates |
| 6. Cognate-coding vs borrowing bias (simulation) | **done** (below): LexStat errors alone −8% ± 5%, borrowing −29% |

## Results so far

### Milestone 1: strict clock is well calibrated

`make m1-strict`: 100 replicates with trees, root ages and loss rates drawn
from `infer/beast/m1_strict_prior.json` (10 languages, 200 concepts), each
fitted with the same prior in BEAST2.

| Parameter | 95% HPD coverage (±2 s.e.) | min ESS | rank-uniformity KS p |
|---|---|---|---|
| root age | 0.94 ± 0.05 | 1313 | 0.19 |
| tree length | 0.91 ± 0.06 | 1311 | 0.20 |
| loss rate μ | 0.90 ± 0.06 | 1362 | 0.16 (χ² p = 0.03, not significant after correcting for 3 tests) |

BEAST2's `-sampleFromPrior` marginals match the simulator's (KS p 0.25–0.65).
Note that the root-age HPD is 94% as wide as its N(4000, 500) calibration
prior: with the rate unknown, the data constrain relative ages, loss rate and
tree length much more than the absolute root. The milestone 2 summary
therefore also reports bias in μ and tree length.

![](figures/m1_coverage_strict.png)

### Milestone 2: bias from unmodelled borrowing

`make m2`: 30 replicates per cell, with truth drawn from the milestone 1
strict prior. Data are simulated with replacement borrowing at rate κ·μ
(κ = 0.25–2) under uniform or distance-weighted contact (e-folding at 1000 y
of patristic distance). Every fit uses the no-borrowing SDollo model.
Replicate *i* shares its tree and parameters across cells, so the **paired
Δ** (shift against the same replicate's κ = 0 fit) is the precise measure.

| Contact | Borrowed entries | Root age, paired Δ (± s.e.) | Root 95% HPD coverage | Loss rate μ bias | Tree length bias |
|---|---|---|---|---|---|
| — | 0% | 0 | 1.00 | −1% | +2% |
| uniform | 6% | −2.4% ± 0.5 | 1.00 | −11% | +11% |
| uniform | 11% | −3.6% ± 0.6 | 1.00 | −14% | +16% |
| uniform | 19% | −5.5% ± 0.7 | 0.97 | −17% | +23% |
| uniform | 30% | −7.4% ± 0.8 | 0.93 | −20% | +30% |
| distance | 6% | −1.1% ± 0.4 | 1.00 | −9% | +5% |
| distance | 10% | −1.7% ± 0.4 | 1.00 | −12% | +7% |
| distance | 17% | −2.8% ± 0.4 | 1.00 | −17% | +9% |
| distance | 26% | −3.6% ± 0.5 | 1.00 | −22% | +12% |

* **Direction.** Unmodelled borrowing makes the root *younger* and the loss
  rate *lower*, and both effects grow steadily with the borrowing rate.
  Tree length goes *up* while the root goes down, so internal nodes move
  towards the root: the tree looks more star-like.
* **Contact model.** Borrowing between near neighbours does about half the
  damage to the root and tree length. The prediction that close and distant
  borrowing bias dates in *opposite* directions is not borne out at these
  settings: both push the same way.
* **Size.** The root-age bias is small here because the root has a tight
  calibration prior (N(4000, 500); milestone 1 showed the data barely narrow
  it). The distortion shows up instead in μ (up to −22%) and tree length (up
  to +30%). **Next experiment:** calibrate internal nodes and leave the root
  vague, as in real analyses (milestone 5), to see how much of the μ bias
  turns into root-age bias.

![](figures/m2_bias_curve.png)

### Milestone 2b: the bias is large under a realistic calibration design

Milestone 2 inferred with the generating root prior N(4000, 500), which holds
the root in place. Real analyses give the root a vague prior and calibrate a
few recent clades. `scripts/m2b_calibration.py` refits the same milestone 2
datasets with root ~ Uniform(1500, 9000), an estimated Yule rate, and up to
two calibrations on the oldest non-nested clades aged 8–40% of the root
(Normal(true age, 20%); Mayan Yucatecan/root ≈ 0.25). 30 replicates per cell.

| Cell | borrowed | root rel. bias | ± se | coverage | μ rel. bias | M2 root bias (same data) |
|---|---|---|---|---|---|---|
| no borrowing | 0% | +0.1% | 1.9 | 1.00 | +1% | +1.9% |
| uniform, κ = 1 | 19% | **−40.8%** | 2.2 | 0.13 | +34% | −3.6% |
| uniform, κ = 2 | 30% | **−49.3%** | 1.7 | 0.00 | +46% | −5.4% |
| distance, κ = 2 | 26% | **−28.9%** | 1.8 | 0.37 | +7% | −1.7% |

Without borrowing the design is unbiased and calibrated. With borrowing the
root comes out 30–50% too young and the 95% HPD almost never contains the
truth. The mechanism: the recent calibrations fix the rate of change near the
tips, while borrowing makes deep lineages share more cognates than descent
alone would leave, so the deep tree is compressed. The loss rate is now
*over*estimated (milestone 2, with the root held, underestimated it). The
milestone 2 numbers understate the problem; this design is the relevant one.

![](figures/m2b_calibration.png)

### Milestone 3: form-level detection (pair HMM)

`scripts/m3_roc.py`, 20 replicates per setting, about 10% of entries borrowed
(κ = 0.5). The detector sees only leaf forms and cognate sets. Each language
pair gets a pair-HMM mixture of "regular correspondences" (learned) vs "loan"
(near-identity). Pair log-likelihood ratios are robustly z-scored within each
language pair, which raised pair-level AUC from 0.63 to 0.74 by removing the
pair's overall divergence. An entry's score is the mean over its cognate-set
partners.

* AUC 0.70–0.72 once regular sound change is at least 1 per 200 y per
  lineage, and 0.65 at 1 per 400 y.
* Power depends on the **separation** between recipient and donor (MRCA age
  minus loan age): chance level below 500 y, about 0.60–0.68 at 500–1500 y,
  and about 0.72 beyond. Loans between close relatives, or made soon after a
  split, are invisible in the forms.
* **Main limitation: form evidence is symmetric.** A loan makes recipient and
  donor-clade forms look alike, and nothing in the pair says who borrowed. At
  z > 1 the detector has precision about 0.42 and recall about 0.32 (κ = 1).

![](figures/m3_roc.png)

### Milestone 3b/3c: level 3, forms plus tree

Level 1 mistakes retentions for loans and ignores the tree. Level 3
(`hmm/phylo.py`) adds, per entry (language, cognate set):

* **set-centred form scores**: similarity relative to the other pairs in the
  same set, so a word that is similar everywhere (a retention) is not flagged;
* **Dollo patchiness**: losses saved, in parsimony and likelihood versions
  (branch lengths in expected losses per class), by explaining the entry or a
  clade containing it as a transfer; and the depth of the set's MRCA;
* **form × depth** (`deep_sim_max`): how similar the entry is to its most
  distant partner (a loan is similar to a distant language; a false cognate
  merge spans distant languages with dissimilar forms);
* a logistic combination trained on simulations, always evaluated on
  held-out replicates (`scripts/m3b_detector.py`).

Pooled held-out AUC, 10 languages × 200 concepts, true cognate sets, UPGMA
tree, 20 replicates (`scripts/m3b_detector.py`):

| Contact, phonology | level 1 | level 3 | precision at base rate |
|---|---|---|---|
| uniform, default | 0.67 | **0.80** | 0.29 → 0.43 |
| uniform, Mayan-like (CVC, stable segments) | 0.62 | **0.79** | 0.21 → 0.39 |
| distance, default | 0.67 | **0.78** | 0.26 → 0.34 |
| distance, Mayan-like | 0.61 | **0.77** | 0.17 → 0.31 |

The true tree gains only about 0.01 over the UPGMA tree. The strongest single
feature is a deep MRCA with few members (an old but patchy set).

**On real Mayan data** (`scripts/m5_detector.py`), level 3 no longer ranks the
retentions *uk'* "louse" highly (ranks 364–901 of 1155, against 3–6 under
level 1), but its top hits are LexStat coding errors (e.g. Akatek *wonan* and
Chicomuceltec *ch'en* "mountain" merged in one set). A false merge looks like
an old, patchy set. Training on simulations recoded by the same LexStat
pipeline brought held-out AUC down to 0.56: on short Mayan-like words LexStat
is only slightly less accurate (B-cubed F 0.83 vs 0.89–0.93 for longer
words), but its errors give inherited sets deep, patchy patterns too, and the
MRCA-depth feature falls from 0.63 to 0.54.

**List length decides it** (`scripts/m3c_list_length.py`; 31 languages,
Mayan-like phonology, 6 replicates, held-out AUC):

| Concepts | cognates | level 1 | level 3 |
|---|---|---|---|
| 40 | true | 0.59 | 0.67 |
| 40 | LexStat | 0.58 | **0.55** |
| 100 | true | 0.61 | 0.73 |
| 100 | LexStat | 0.61 | **0.64** |
| 200 | true | 0.63 | 0.76 |
| 200 | LexStat | 0.63 | **0.69** |

With 40 concepts and automated cognates there is not enough information for
level 3 to beat level 1. At 100–200 concepts it does, by 0.03–0.06 with
automated cognates and 0.12–0.13 with correct ones. A borrowing-corrected
Mayan date therefore needs longer lists (100–200 concepts) and better cognate
coding (expert judgements, or morpheme-level partial cognates, since many
ASJP errors come from affixes such as *-al*, *-il*), not only a better
detector. The ASJP 40-item Mayan data cannot support it.

![](figures/m3b_detector.png)

### Morpheme-level (partial) cognates: robust in simulation, over-merges on real data

Many ASJP Mayan coding errors involve affixes (*eh* / *eh-al* "tooth", *im* /
*im-il* "breast"). `pipeline.asjp` now has data-driven suffix discovery
(vowel-initial tails that turn one form of a concept into another in at
least 4 concepts: *-il*, *-al*, *-el*, *-ax*), segmentation, and partial
cognates (LingPy `Partial`, UPGMA); each word takes the cognate ID of its
most widespread non-suffix morpheme. The simulator can now add Mayan-style
suffixes (`PhonologyParams.suffix_rate`), which keep cognacy.

B-cubed F against true cognates (`scripts/m5_cognate_eval.py`; 31 languages
× 40 concepts, Mayan-like phonology, 4 replicates; 3%, 8% and 14% of words
suffixed, against about 10% observed in the ASJP Mayan lists):

| Method | no suffixes | 3e-5 | 7e-5 |
|---|---|---|---|
| LexStat 0.75, whole word (current) | 0.868 | 0.857 | 0.838 |
| partial SCA 0.45 | 0.851 | 0.848 | 0.850 |
| partial LexStat 0.75 | 0.868 | 0.864 | 0.852 |
| partial LexStat 0.85 | **0.877** | **0.869** | **0.854** |

In simulation, partial LexStat is never worse and is more robust to
suffixation (+0.01 to +0.02). On the real Mayan data the same setting
over-merges: "eye" puts *sat*, *wich*, *b'aq' wach* and *hut* in one set, and
"stone" merges *ch'en*, *tun* and *k'e'en*. Real lists contain compounds and
phrases (*b'aq' wach*), whose shared generic morphemes link different roots,
and the simulator has none. Identical *wits* "mountain" forms are still split
in some languages (2% of identical pairs). Judged by eye this looked like a reason to stay on whole-word LexStat 0.75;
the expert benchmark below shows that was wrong on balance.

The lesson is the same as for the LexStat threshold: settings tuned on
simulation do not transfer to the real lists, and there is no gold standard
to tune against. Expert cognate judgements (for example Kaufman's Mayan
etymological dictionary) on a longer concept list are needed both as data
and as a benchmark.

### A richer Mayan dataset and an expert benchmark: PMED Swadesh-100

The lexibank `kaufmanpmed` dataset is Kaufman & Justeson's *Preliminary
Mayan Etymological Dictionary* (2003): 40,424 forms from 83 modern varieties
(34 languages), with 37,812 forms assigned to Kaufman's etyma. It has no
Concepticon links, but every form carries its source gloss ("Spanish//English").
`pipeline/pmed.py` maps it to the Swadesh 100 list (`pipeline/pmed_swadesh100.tsv`,
English and Spanish glosses with notes and exclusions, open to review):

* a form belongs to a concept only if one of its gloss senses *equals* a
  listed gloss, so "pecho de mujer" is not "woman" and "chile diente de perro"
  is neither "tooth" nor "dog"; Spanish enclitics are reduced (*morderlo* →
  *morder*); "monte" (bush) and "tomar" (take) are excluded, and so are
  "lengua" forms also glossed as "language";
* one modern variety per Glottocode (most concepts covered); epigraphic and
  colonial sources dropped (colonial Ch'olti' could later be a dated tip);
* cognate sets = Kaufman's etyma.

Result (`scripts/build_pmed_mayan.py` → `data/derived/pmed_mayan`): 29
languages, 2,782 forms, 511 etyma, 43–85 concepts per language (mean 20.5
languages per concept; 62 concepts in ≥ 20 languages). Verbs are sparse
(*bite* 8, *kill* 7, *sit* 1 languages): PMED records few verb roots.
Chicomuceltec and Chontal are effectively absent (2 and 3 concepts).
Caveat: PMED lists reflexes of reconstructed etyma, so a word that is not a
reflex of any etymon may be missing; missing cells are coded as missing.

**Benchmark of automated cognates on real data** (`scripts/benchmark_cognates.py`):
ASJP forms whose language and concept also occur in PMED are labelled with
Kaufman's etymon when PMED has exactly one etymon for that cell (857 forms,
103 gold sets; no spelling comparison, so spelling-based methods are not
favoured). B-cubed against the expert labels:

| Method | P | R | F |
|---|---|---|---|
| LexStat 0.55 | 0.937 | 0.539 | 0.685 |
| LexStat 0.75 (current) | 0.875 | 0.720 | 0.790 |
| LexStat 0.85 | 0.842 | 0.789 | 0.814 |
| SCA 0.55 | 0.844 | 0.759 | 0.799 |
| edit distance 0.75 | 0.873 | 0.775 | 0.821 |
| partial LexStat 0.75 | 0.859 | 0.727 | 0.788 |
| partial LexStat 0.85 | 0.826 | 0.816 | **0.821** |

Every method splits more than it merges (recall < precision), so the
current threshold is too strict. Partial LexStat at 0.85, which simulation
also ranked first, is best on real data (+0.03 F over the current setting).
My earlier impression, from a few concepts, that it over-merges on real data
was wrong on balance: it does merge some compounds, but recovers more true
cognates than it loses. A second labelling that links forms by spelling
similarity gives the same ordering for LexStat and partial LexStat but
favours edit distance (F 0.882), as expected from its circularity.

### Milestone 4: level 2 (negative so far)

Two latent-donor models are in `hmm/latent.py`, both fitted by EM. Mean AUC
on 10 test datasets (κ = 1, 1 sound change / 100 y):

| Model | uniform | distance |
|---|---|---|
| level 1 (mean pair score) | **0.66** | **0.70** |
| donor = a language, per-language Dirichlet profiles (`fit_latent`) | 0.57 | 0.61 |
| donor = a lineage (clade of a UPGMA tree from cognate sharing; `fit_latent_lineage`) | 0.63 | 0.68 |
| lineage + sister-clade "direction" cue | 0.54 | 0.62 |

Treating the donor as a lineage removes most of the loss from the
single-language version, since a loan from an ancestral lineage makes the
whole donor clade look alike. It still does not beat the simple mean.

The direction cue rested on a wrong assumption. The idea was that a
recipient's close relatives should lack a borrowed set. In simulation they
lack it as often as for inherited sets (uniform: 0.35 vs 0.37), or *less*
often (distance contact: 0.14 vs 0.30). Most loans happen on ancestral
branches, so the sister clade inherits the loan too, or the sister clade is
the donor. Form-level evidence therefore remains the binding constraint, and
the correction below uses level 1.

The correction study (`scripts/m4_correction.py`) refits the κ = 1 milestone 2
replicates after masking (language, concept) cells as missing, using either
the true loans (oracle: an upper bound on what any detector can gain) or the
level-1 detector.

Root-age relative bias (mean ± s.e., 30 replicates per cell, κ = 1):

| Contact | no borrowing | uncorrected | oracle mask | pair-HMM mask (z > 1) | coverage |
|---|---|---|---|---|---|
| uniform | +1.9% ± 1.8 | −3.6% ± 1.9 | +1.0% ± 1.8 | −3.2% ± 1.9 | 0.97–1.00 |
| distance | +1.9% ± 1.8 | −0.8% ± 1.8 | +1.6% ± 1.8 | −0.7% ± 1.8 | 1.00 |

Masking the true loans brings the root back to the no-borrowing level, at
almost no cost in HPD width (about 1840–1860 y in every fit). Masking the
detector's flags recovers almost none of the bias: with recall about 0.32 and
precision about 0.42 it removes few loans and many inherited entries. The
correction step works; detection is the bottleneck.

![](figures/m4_correction.png)

### Milestone 5: Mayan case study (draft calibrations, do not quote)

31 Mayan languages × 40 ASJP core concepts, LexStat cognates (threshold 0.75
unless stated), calibrations as drafted in `infer/beast/m5_mayan_prior.json`
(Yucatecan crown Normal(1050, 200) BP; Huastecan crown Uniform(480, 1100) BP;
root Uniform(1500, 9000) BP). 20M states per run; every ESS > 2000. Ages are
years before AD 2000.

| Variant | Proto-Mayan, median | 95% HPD | loss rate (median) |
|---|---|---|---|
| strict clock | 435 BC | 1304 BC – AD 307 | 4.0e-4 |
| relaxed clock (UCLN) | 430 BC | 1303 BC – AD 326 | 4.1e-4 |
| strict + Cholan calibration | 1486 BC | 2260 BC – 728 BC | 2.8e-4 |
| strict, cognate threshold 0.65 | 196 BC | 905 BC – AD 465 | 5.2e-4 |
| strict, cognate threshold 0.85 | 454 BC | 1283 BC – AD 292 | 3.5e-4 |
| strict, SCA cognates (0.55) | 292 BC | 1041 BC – AD 448 | 3.9e-4 |
| strict + candidate Cholan–Tzeltalan ≥ 1700 BP | 912 BC | 1605 BC – 301 BC | 3.4e-4 |

These are too young. The HPD reaches AD 300, when Classic inscriptions already
show well-separated Mayan languages, and every published Proto-Mayan estimate
is older (Kaufman's glottochronological c. 2200 BC among them). What the runs
show:

* **Not the priors.** Sampling from the prior alone gives a root median of
  4423 BP (HPD 1501–8449); the young root comes from the data.
* **Not the cognate method.** LexStat on 40-item lists does make errors
  (identical forms such as *wits* "mountain" end up in 7 sets; 115 of 255 sets
  are singletons), but SCA clustering, which never splits identical forms
  (190 sets, 58 singletons), gives the same root (292 BC). Thresholds 0.65,
  0.75 and 0.85 also agree.
* **Not rate variation.** The relaxed clock's rate SD is small (median 0.12,
  HPD 0–0.26) and the root is unchanged.
* **The data pull young against every older anchor.** With the contested
  Cholan calibration the root moves to 1486 BC and with the candidate
  Cholan–Tzeltalan minimum (Classic inscriptions are Ch'olan from AD 292) to
  912 BC, but in both cases the older node sits on its lower bound (Cholan
  median 1834 BP vs bound 1750; Cholan–Tzeltalan 1846 vs 1700).
* **This is the pattern borrowing produces.** Milestone 2b shows that with
  recent calibrations and a vague root, borrowing makes roots 30–50% too
  young and inflates the loss rate, which is what the Mayan fits show
  (loss rate about twice its prior mean, older anchors pinned at their
  bounds). The Mayan Highlands and Lowlands are known contact zones. This is
  consistent with, not proof of, borrowing as the cause. **See milestone 5b:**
  with Kaufman's expert etyma the root is c. 2800 BC under the same
  calibrations; 5c shows the difference is cognate coding.

Next: agree the calibrations (including the candidate Cholan–Tzeltalan
minimum) with the supervisor, and develop a detector that works on real data
so a borrowing-corrected Mayan fit becomes possible.

![](figures/m5_root_age.png)

### Milestone 5b: expert cognates move Proto-Mayan back by about 2,000 years (draft calibrations, do not quote)

`scripts/m5b_pmed.py`: the PMED Swadesh-100 wordlist (29 languages, Kaufman's
etyma) against the ASJP 40 / LexStat 0.75 data of milestone 5, **with the same
calibrations**: Yucatecan crown Normal(1050, 200) BP, and in the `_ct` runs
the candidate Cholan–Tzeltalan minimum Uniform(1700, 5000) BP. The Huastecan
calibration is dropped because PMED lacks Chicomuceltec. Root Uniform(1500,
9000), Yule, strict clock, 20M states, every ESS > 2500.

| Run | Data | Proto-Mayan, median | 95% HPD | loss rate | Cholan–Tzeltalan (bound 1700 BP) |
|---|---|---|---|---|---|
| asjp_yuc | ASJP 40, LexStat | 351 BC | 1219 BC – AD 494 | 4.2e-4 | |
| asjp_yuc_ct | ASJP 40, LexStat | 923 BC | 1713 BC – 295 BC | 3.4e-4 | 1855 BP |
| pmed_yuc | PMED 100, etyma | **2619 BC** | 4721 BC – 820 BC | 1.8e-4 | |
| pmed_yuc_ct | PMED 100, etyma | **2792 BC** | 4398 BC – 1606 BC | 1.8e-4 | 2020 BP |
| pmed40_ct | PMED, ASJP's 40 concepts only | 2820 BC | 4797 BC – 1356 BC | 1.2e-4 | 1986 BP |
| pmed_ct_innov | PMED 100, missing cells = unshared innovations | 1769 BC | 2541 BC – 1041 BC | 3.4e-4 | 1794 BP |
| pmed_prior | no data | 3462 BC | 6977 BC – AD 612 | | |

* **The dataset, not the calibrations, makes the root young.** With
  identical calibrations PMED puts Proto-Mayan near Kaufman's glottochronological
  c. 2200 BC; ASJP puts it in the first millennium BC. PMED's interval is far
  narrower than the prior's and its median younger, so the date comes from the
  data. With PMED the Cholan–Tzeltalan node is no longer pinned at its bound.
* **Not list length.** PMED restricted to ASJP's 40 concepts gives the same
  root (2820 BC). The difference is in the cognate coding.
* **Robust to PMED's ascertainment.** PMED lists reflexes of reconstructed
  etyma, so a language's own innovations may be absent (28% of cells are
  missing, mostly verbs). Coding every missing cell as an unshared
  innovation, the opposite extreme, raises the loss rate to ASJP's level but
  still gives 1769 BC, older than every ASJP run; it also pulls the Yucatecan
  node to 1401 BP against its prior of 1050 ± 200, a sign that this coding
  overstates change. The truth plausibly lies between 1769 and 2792 BC.
* **Revises milestone 5** (but milestone 6 qualifies this). A first reading: the
  young Mayan root there is mainly a cognate-coding artefact: LexStat on 40-item ASJP lists splits true cognate
  sets (benchmark above: recall 0.72), and every spurious split looks like a
  replacement, inflating the loss rate (4.2e-4 vs 1.8e-4). Borrowing remains
  a possible further bias, but the milestone 5 evidence for it is weak.
  It also shows the cost of automated cognates is large on real data, which
  matters for any borrowing correction built on them.
  Milestone 6 finds that LexStat-style errors alone shift simulated roots by
  only −8%, but milestone 5c confirms the effect on the real words: the
  simulated words are too easy for LexStat.

![](figures/m5b_root_age.png)

### Milestone 6: automated cognates alone barely bias dates in simulation

`scripts/m6_cognate_coding.py`: 20 Mayan-shaped simulations (31 languages ×
40 concepts, milestone 1 priors, Mayan-like CVC words), without borrowing and
with uniform borrowing at κ = 1. Each dataset is fitted twice: with the true
cognate classes, and with cognate sets re-inferred from the simulated words
by LexStat 0.75 (exactly as for the real ASJP data). Milestone 2b
calibration design (root Uniform(1500, 9000), Yule estimated, two internal
calibrations at 8–40% of the root, Normal(true, 20%)); 5M states, ESS > 1400.

| Cognates | Borrowing | LexStat B-cubed F | Root bias (± se) | Paired Δ vs true cognates, no borrowing | Coverage | Loss-rate bias |
|---|---|---|---|---|---|---|
| true | none | 1 | +2.8% ± 2.6 | 0 | 1.00 | +2% |
| LexStat | none | 0.85 | −4.9% ± 4.1 | −7.8% ± 4.9 | 0.90 | +72% |
| true | κ = 1 (21% borrowed) | 1 | −29.1% ± 3.1 | −31.9% ± 4.8 | 0.40 | +82% |
| LexStat | κ = 1 | 0.84 | −27.6% ± 4.0 | −30.5% ± 5.6 | 0.30 | +133% |

* **Coding errors inflate the loss rate (+72%) but hardly move the root**
  (−8% ± 5%, paired; 14 of 20 replicates younger, range −54% to +31%). The
  calibrations hold the time scale; the extra apparent change is absorbed
  elsewhere in the tree. The effect does not track LexStat's accuracy or
  whether it over- or under-splits.
* **Borrowing dominates**: −29%, and adding LexStat errors on top changes
  nothing (paired difference +1.5%).
* **So simulation does not reproduce the real-data gap** of milestone 5b
  (ASJP/LexStat about 2,000 years, or roughly 40–55%, younger than PMED).
  Caution from the interim runs: after 3–7 replicates the LexStat-only bias
  looked like −20 to −28%; it shrank to −5% at 20 replicates.
* Remaining explanations for the real gap, not yet tested: (1) LexStat is
  worse on real data (F 0.79, mostly splitting) than on these simulated
  words (F 0.85, balanced errors), so the simulated phonology may be too
  easy; (2) expert etyma may identify and separate some intra-Mayan loans
  that LexStat cannot, so the PMED fit would carry less borrowing bias,
  which would make the 5b gap partly a borrowing effect after all; (3)
  PMED's ascertainment (bounded in 5b, but not removed).

![](figures/m6_cognate_coding.png)

### Simulation realism (in progress): review, first calibration, and a failed design

`scripts/sim_realism.py` compares simulated words with the real ASJP Mayan
words (gold: Kaufman's etyma) on the statistics that drive cognate coding:

| | Real | M5/M6 simulation | MAYAN_V1 |
|---|---|---|---|
| gold sets per concept | 2.6 (retention-enriched subset) | 15.7 | 3.3 |
| forms per language and concept | 1.08 | 1.60 | 1.00 |
| NED of true cognates (all / across the root split) | 0.49 / 0.64 | 0.31 / 0.52 | 0.51 / 0.63 |
| LexStat sets / gold sets | **1.70** | 0.87 | 1.38 |
| LexStat B-cubed F | 0.80 | 0.83 | 0.79 |
| pairwise recall (all / across root) | 0.74 / 0.73 | 0.69 / 0.43 | 0.60 / 0.46 |
| non-cognate pairs merged | 0.29 | 0.04 | 0.09 |

The main problem was the lexicon, not the phonology: far too much turnover
and polymorphism, so cognate sets were tiny and there was little to split.
New `PhonologyParams` options (all off by default; earlier results reproduce
exactly; tests in `tests/test_phonology.py`): `segment_skew` (Zipf-like
segment frequencies in new roots), `derived_prob` (new classes built on old
words by compounding or suffixation), `compound_share` (compounding that
keeps cognacy, alongside `suffix_rate`). `sim_realism.py` can also simulate
on the PMED MCC tree (`tree: "pmed"`), fix the loss rate, and keep one form
per language and concept (`one_per_cell`). MAYAN_V1 combines these.

**Milestone 6r failed as designed** (`scripts/m6r_realistic.py`, 10
replicates on the PMED tree, real calibration nodes at their true ages,
real-analysis priors):

| Cognates | Borrowing | Root bias (± se) | Coverage | Loss-rate bias |
|---|---|---|---|---|
| true | none | **−41.9% ± 5.1** | 0.10 | +304% |
| LexStat | none | −5.9% ± 7.5 | 0.90 | +346% |
| true | κ = 1 (9% borrowed) | −48.4% ± 1.9 | 0.00 | +336% |
| LexStat | κ = 1 | −23.2% ± 4.0 | 0.70 | +372% |

The true-cognate baseline is badly biased, so nothing here can be
interpreted. Cause: `one_per_cell`. Under stochastic Dollo with birth =
loss, a language has several classes for a concept about a quarter of the
time; keeping one at random makes the discarded ones look lost on terminal
branches, inflating the loss rate about fourfold. Without `one_per_cell`
LexStat no longer over-splits (sets ratio 1.04), so most of MAYAN_V1's
apparent realism came from this misspecification. Fix (next): a lexical
process in which each language has about one word per concept and words are
*replaced* (with occasional synonymy), then re-check that true cognates give
unbiased dates before repeating 6r.

**Milestone 6r, design V2: a valid baseline, but the LexStat bias has the
wrong sign.** MAYAN_V2 replaces the lexical process (`process:
"replacement"`: one word per concept, a new word replaces the old one, 10%
of replacements keep the old word as a short-lived synonym) and adds
per-form transcription noise at the tips (`tip_noise: 0.3`); no
subsampling. Its statistics: LexStat sets / gold 1.46, F 0.76, NED
0.56 / 0.65, non-cognates merged 0.15, recall 0.59 / 0.46 (real 0.74 /
0.73). `python scripts/m6r_realistic.py --design v2 prepare|run|summarise`;
10 replicates, 5M states, `data/derived/m6r2`, `figures/m6r2_realistic.png`.

| Cognates | Borrowing | Root bias (± se) | Δ vs true, no borrowing (paired) | Coverage | Loss-rate bias |
|---|---|---|---|---|---|
| true | none | **−0.8% ± 5.2** | | 1.00 | +7% |
| LexStat (F 0.77, sets × 1.35) | none | +15.8% ± 7.2 | +16.7% ± 6.6 | 0.90 | +159% |
| true | κ = 1 (4.6% borrowed) | −14.0% ± 4.8 | −13.1% ± 7.6 | 0.90 | +62% |
| LexStat | κ = 1 | +6.1% ± 7.2 | +6.9% ± 9.8 | 0.80 | +187% |

The baseline is now unbiased, so the replacement process is compatible with
the dating model. Borrowing makes the root younger, as in milestones 2 and 6.
LexStat over-splitting in V2 makes the root *older* (+17%), whereas on the
real words the same over-splitting makes it ~4,000 years *younger* (5c).
The two effects partly cancel.

Why the signs differ (`scripts/m6r_split_shape.py`, real ASJP/Kaufman vs
three V2 replicates): the *shape* of the splits differs. On real data
LexStat mostly peels single languages off a set: singleton sets go from 22%
(gold) to 47% (LexStat), and 38% of the fragments of large sets are
singletons. In V2 the gold data already have ~42% singletons (largely
replacements on terminal branches; the real 22% is understated, since
Kaufman labels only forms with an etymology, about two thirds of them),
LexStat leaves that share unchanged (~40%), and only ~22% of fragments are
singletons; V2 cuts sets into subgroup-sized pieces instead. Real recall is
the same across the root split as overall (0.73 vs 0.74), so real errors are
idiosyncratic to single languages, while V2's come from divergence along
lineages and so fall on deep pairs (0.46 vs 0.59). Singleton fragments look like many recent
innovations, which pulls the root down; this matches M5b, where coding
missing PMED cells as singletons moved Proto-Mayan from 2792 BC to 1769 BC.
(The guess that subgroup-sized fragments stretch the tree was wrong: random
subgroup splits make it younger, see 6s. V2's sign is unexplained.)

So V2 does not yet reproduce the error that matters on real data, and its
+17% should not be read as the real-world LexStat effect.

**Design V3: errors at the tips reverse the LexStat bias.** Tuning (3 replicates per
setting; statistics from `sim_realism.stats` plus the split shape) showed
that the tip noise, not the regular sound change, causes most of V2's
divergence; without it, deep recall is close to real (0.67 to 0.77). No
setting matches every statistic: less lineage change raises recall but
removes the splits, and any tip-level error also costs deep recall. MAYAN_V3
moves divergence from lineages to tips (lineage sporadic change 1e-4 to
3e-5, suffixation 5e-5, tip noise 0.4, elicited tip compounds 0.15):

| | Real | V2 | V3 |
|---|---|---|---|
| LexStat sets / gold | 1.70 | 1.45 | 1.58 |
| B-cubed F | 0.80 | 0.76 | 0.77 |
| recall (all / across root) | 0.74 / 0.73 | 0.59 / 0.46 | 0.59 / 0.50 |
| LexStat singleton sets | 0.47 | 0.41 | 0.47 |
| singleton share of fragments of sets with 4+ languages | 0.38 | 0.23 | 0.33 |
| share of languages in the largest fragment | 0.81 | 0.77 | 0.79 |

`python scripts/m6r_realistic.py --design v3 prepare|run|summarise`; 10
replicates, 5M states, `data/derived/m6r3`, `figures/m6r3_realistic.png`.
The true-cognate cells are identical to V2: the same seeds give the same
cognate matrices, and V3 changes only the words.

| Cognates | Borrowing | Root bias (± se) | Δ vs true, no borrowing (paired) | Coverage | Loss-rate bias |
|---|---|---|---|---|---|
| true | none | −0.8% ± 5.2 | | 1.00 | +7% |
| LexStat (F 0.78, sets × 1.52) | none | **−12.8% ± 3.9** | **−12.0% ± 4.1** | 0.90 | +254% |
| true | κ = 1 (4.6% borrowed) | −14.0% ± 4.8 | −13.1% ± 7.6 | 0.90 | +62% |
| LexStat | κ = 1 | −21.2% ± 2.9 | −20.4% ± 5.0 | 0.70 | +297% |

With the same true histories, moving the coding errors from lineages (V2)
to single languages at the tips (V3) turns the LexStat bias from +17% to
−12%. The direction of the bias depends on the *shape* of the errors, not
just their number, and errors shaped like the real ones make the root
younger, as on real data. Borrowing and LexStat errors add up (−21%). The
simulated effect is still much smaller than the real one (5c: thousands of
years, i.e. roughly −40% or more relative to the expert-coded root), so tip
errors are a sufficient direction-level explanation but not a quantitative
one.

### Milestone 6s: single-language splits are what drive the young root

A direct test with no word simulation or LexStat
(`scripts/m6s_split_shape_dating.py prepare|run|summarise`,
`data/derived/m6s`, `figures/m6s_split_shape.png`). It uses the true cognate
sets of the same 10 histories (V2/V3, no borrowing). Each replicate gets
exactly as many extra sets as LexStat produced on it in V3 (165 to 250 sets
on average), added in one of two ways. *tip*: a random form whose set spans
2+ languages becomes a singleton. *clade*: the members of a 3+-language set
that lie under a random internal node of the true tree (2+ languages, not
all) move to a new set. Same priors, calibrations and seeds as 6r.

| Cognates | Sets | Root bias (± se) | Δ vs true (paired) | Coverage | Loss-rate bias |
|---|---|---|---|---|---|
| true | 165 | −0.8% ± 5.2 | | 1.00 | +7% |
| split off single languages | 250 | **−30.7% ± 4.3** | −29.8% ± 3.6 | 0.30 | +92% |
| split off subgroups | 250 | −13.1% ± 3.8 | −12.3% ± 3.5 | 0.90 | +92% |
| LexStat on V3 words | 250 | −12.8% ± 3.9 | −12.0% ± 4.1 | 0.90 | +254% |

With the number of errors fixed, splitting off single languages biases the
root young about 2.5 times as much as splitting off subgroups, and pushes
coverage to 0.30. This supports the explanation for real Mayan data, where
LexStat's splits are mostly single-language (5c, 6r V2). It also corrects
the 6r V2 account. Random subgroup splits make the root *younger*, not
older, so they do not explain V2's +17%. V2 and V3 LexStat also have the
same precision and recall (P 0.89, R 0.68 vs 0.69), so the explanation is
not extra merging either. V2's errors fall disproportionately across the
root split (deep recall 0.46 vs 0.59 overall), which random clade splits do
not reproduce. Splits concentrated at the root are an untested candidate.
The V2 sign is left open.

### Milestone 7: Indo-European (IE-CoR) replication (draft calibrations, do not quote)

`scripts/m7_iecor.py prepare|run|summarise`; prior `infer/beast/m7_iecor_prior.json`;
`data/derived/m7`; `figures/m7_iecor.png`. IE-CoR v1.2.1 (lexibank/iecor), with
40 modern languages from seven branches (Albanian and English have no segments
in IE-CoR) and 170 concepts. Only the 6,838 forms with phonetic segments are
used, so every coding is of *identical words*. Calibrations are broad
modern-only clade ages (Romance, North Germanic, Slavic, Baltic,
Indo-Iranian, Celtic). Root prior is Uniform(3500, 12000). These are DRAFT.

On IE, LexStat 0.75 **over-merges**: 1,829 sets against 2,509 expert sets,
B-cubed P 0.71, R 0.81, F 0.75. This is the opposite of Mayan, where it
over-split.

Results at 10M states (root ESS 1,491 to 1,801; loans removed 1,732):

| Coding | Sets | Singleton share | Root, median BP (95% HPD) | Loss rate |
|---|---|---|---|---|
| IE-CoR expert | 2,509 | 0.56 | **7528** (6985 to 8162) | 1.8e-4 |
| expert + only LexStat's splits | 3,111 | 0.62 | 7671 (7104 to 8224) | 2.4e-4 |
| expert + only LexStat's merges | 1,243 | 0.30 | **5947** (5039 to 6826) | 2.2e-4 |
| LexStat | 1,829 | 0.31 | **5364** (4867 to 5898) | 3.3e-4 |
| expert, forms in IE-CoR loan sets removed | 2,093 | 0.52 | 8276 (7435 to 9045) | 1.5e-4 |

- Automated coding again makes the root about 2,160 years younger on identical
  words, so the headline effect replicates in a second family. The expert root
  sits between the Steppe and Anatolian hypotheses, near Heggarty et al.'s
  (2023) c. 8,100 BP. That is reassuring, but the calibrations are draft.
- The error type that does the damage differs. On IE, LexStat's merges move
  the root young by about 1,580 years and its splits do almost nothing. On
  Mayan, splits did the damage.
- This qualifies 5c. There, the merge-only and expert runs both sat at the
  root prior's upper bound, so "merges are harmless" was not actually tested.
- A common mechanism fits both. With only shallow calibrations, the root age
  is roughly the ratio of deep to shallow change. Errors that add change near
  the tips make the root young: single-language splits (6s, Mayan). So do
  errors that remove change between distant branches: spurious merges across
  branches (IE), which act like borrowing. Subgroup-sized splits fall in
  between (6s).
- Removing the expert-identified loans makes the root *older*, by about
  750 years, with overlapping intervals. This is the direction borrowing predicts (milestones 2 and 6),
  and the first real-data check of it. Removing whole loan sets also drops
  the donor's inherited forms when the donor is in the sample, so this is a
  rough test.

**Remedy test: the LexStat threshold** (`python scripts/m7_iecor.py thresholds`,
then `run` and `summarise`; 3M states, root ESS 331 to 521;
`figures/m7_thresholds.png`):

| Threshold | Sets | Singleton share | P | R | F | Root, median BP (95% HPD) |
|---|---|---|---|---|---|---|
| 0.45 | 4,128 | 0.67 | 0.97 | 0.59 | 0.731 | 7332 (6158 to 8660) |
| 0.55 | 3,441 | 0.57 | 0.93 | 0.65 | 0.766 | 7193 (6348 to 8084) |
| 0.65 | 2,690 | 0.45 | 0.85 | 0.72 | **0.779** | **5816** (5317 to 6381) |
| 0.75 | 1,829 | 0.31 | 0.71 | 0.81 | 0.753 | 5364 (4867 to 5898) |
| expert | 2,509 | 0.56 | | | | 7528 (6985 to 8162) |

- **The best-scoring coding still gives a badly biased date.** Threshold 0.65
  has the highest F against the experts and the closest set count, yet its
  root is about 1,700 years too young. Only the high-precision thresholds
  (0.45 and 0.55, P ≥ 0.93) recover the expert date, even though they
  over-split heavily and have lower F.
- So cognate-detection scores are a poor guide to dating accuracy. Rama et
  al. (2018) made the same point for tree topology. On IE what matters is
  avoiding false merges. Over-splitting is nearly harmless, as in the
  split-only run.
- There is no universal fix. On Mayan, LexStat already over-splits and its
  splits do the damage (5c), and thresholds 0.65 to 0.85 did not help (5).
  The safe direction depends on which error a family is prone to. "Tune for
  precision" works on IE but would make Mayan worse.

### Milestone 5c: on real data, cognate coding alone moves the root by millennia

`scripts/m5c_asjp_expert.py`: the ASJP Mayan words recoded with Kaufman's
etyma where PMED gives an unambiguous label (857 of 1,317 forms; 27
languages with ≥ 10 labelled forms; Chicomuceltec, Chontal, Mocho' and one
K'iche' variety dropped). Calibrations as `asjp_yuc_ct` in 5b; 20M states,
ESS > 3000.

| Run | Forms | Cognate sets | Sets per concept | Proto-Mayan, median | 95% HPD (BP) | loss rate |
|---|---|---|---|---|---|---|
| sub_lexstat | 857 labelled | LexStat 0.75 | 4.4 | 1276 BC | 2356–4255 | 1.9e-4 |
| sub_expert | the same 857 | Kaufman's etyma | 2.6 | 5246 BC | 5262–8998 | 6.2e-5 |
| full_lexstat | all 1,317 | LexStat 0.75 | 5.8 | 1175 BC | 2501–3953 | 3.0e-4 |
| hybrid | all 1,317 | etyma where labelled, else the majority etymon of the LexStat set, else the LexStat set | 4.0 | 2976 BC | 3453–6659 | 1.3e-4 |
| sub_partial | the same 857 | partial LexStat 0.85 | 3.4 | 899 BC | 2032–3832 | 1.5e-4 |
| full_partial | all 1,317 | partial LexStat 0.85 | 4.1 | 1095 BC | 2424–3839 | 2.4e-4 |
| split_lexstat | the same 857 | Kaufman's etyma + only LexStat's splits (meet) | 5.4 | 1606 BC | 2773–4606 | 2.1e-4 |
| merge_lexstat | the same 857 | Kaufman's etyma + only LexStat's merges (join) | 1.7 | 5822 BC | 5757–8999 | 2.2e-5 |
| split_partial | the same 857 | etyma + only partial LexStat's splits | 4.7 | 1834 BC | 2883–5035 | 1.6e-4 |
| merge_partial | the same 857 | etyma + only partial LexStat's merges | 1.4 | 5707 BC | 5422–8997 | 1.2e-5 |

* **Identical words, identical calibrations, only the cognate coding
  differs: the root moves by about 4,000 years** (sub_lexstat vs
  sub_expert), and by 1,800 years on the complete lists (full_lexstat vs
  hybrid). So the 5b gap is a cognate-coding effect on real data.
* **Neither subset date is meaningful on its own.** The labelled forms are
  reflexes of Kaufman's etyma, which enriches them for retentions; with
  expert coding the root runs into the prior's upper bound (9000 BP). The
  complete-list hybrid (2976 BC) is close to the PMED fits of 5b (2792 BC).
* **On real data LexStat over-splits heavily**: 175 sets where Kaufman has
  103 for the same 857 forms (+70%). In the milestone 6 simulations its set
  counts matched the truth, which is why simulated coding errors moved the
  root by only −8%. The simulated words are too easy, and making them
  realistic (morphology, sound change, synonymy) is the next step for the
  simulation side.
* **The best automated method does not help.** Partial (morpheme-level)
  LexStat, best on the expert benchmark, gets the number of sets right on
  the full lists (165 vs 160 for hybrid) but dates the root to 1095 BC, like
  LexStat. So it is *which* forms are grouped, not how many sets there are,
  that moves the date (as in milestone 6, where LexStat errors inflated the
  loss rate without changing set counts).
* **Splits make the root young; merges do not.** Applying only LexStat's
  real splits to Kaufman's etyma takes the root from 5246 BC (at the prior
  bound) to 1606 BC, close to LexStat itself (1276 BC); applying only its
  merges leaves it at the bound (5822 BC). Partial cognates behave the same
  way (1834 BC vs 5707 BC). Partial cognates matched the expert set count
  by adding merges, which do not affect the date, while keeping the splits,
  which do. Each split true cognate set looks like a replacement, raising
  the loss rate; the shallow calibrations then compress the tree.
* **So the target for better automated coding is recall** (not splitting
  true cognates), not precision. Raising the threshold is not enough:
  thresholds 0.65–0.85 gave the same root in milestone 5. The damaging
  splits are between true cognates that look too different for any
  similarity threshold, i.e. ones that need regular sound correspondences
  to be recognised.
* **The errors are not simply deep cognates being missed**
  (`scripts/m5c_error_depth.py`). Pairwise against Kaufman, LexStat finds
  0.70–0.73 of true-cognate pairs between major branches and 0.83 within
  low-level subgroups (partial: 0.80–0.83 vs 0.89), a modest depth trend.
  Both methods also merge 26–49% of pairs that Kaufman assigns to
  *different* etyma, at every depth. Which kind of error drives the young
  root is still open.
* **What counts as a cognate is part of the answer.** Kaufman's etyma are
  roots: derivatives and compounds of one root share an etymon, whereas
  LexStat compares whole words. Under the stochastic Dollo model a new
  derivative of an old root is a replacement in one coding and not in the
  other, so "correct" coding for dating depends on how lexical replacement
  is defined. Morpheme-level (partial) cognates sit between the two.


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
from the ASJP/Wichmann line or published Mayan phylogenetic studies). 

**Milestone 5 data: ASJP + automated cognates** (`make matrices`,
`pipeline/asjp.py`). From `lexibank/asjp` v21.1 (pinned): the 109 Mayan
doculects reduce to 31 languages (one per Glottolog languoid, ≥ 28 of the 40
core concepts). Every major branch is present: Huastecan, Yucatecan,
Cholan–Tzeltalan, Q'anjobalan, Mamean, K'iche'an, Mocho' and Chicomuceltec.
The 15 forms ASJP flags as loans (Spanish) are removed. Cognates come from
LingPy LexStat at threshold 0.75: 255 sets, 2.5% missing cells.

* **Threshold calibration.** On simulated wordlists of the same shape
  (31 × 40, root ≈ 4000 y), B-cubed F was 0.73 / 0.84 / 0.90 / **0.93** at
  thresholds 0.45 / 0.55 / 0.65 / **0.75** (P 0.93, R 0.93 at 0.75). At
  LingPy's usual 0.55, obvious Mayan cognates were split; for example,
  reflexes of *\*ka'ib'* "two" fell into 4 sets. At 0.75 they form one.
* **Known errors at 0.75**, from spot checks: *k'ak'al* (from "fire") is
  merged into "sun"; *siwan* is merged with *tun* "stone"; *tunich* is split
  from *tun*. Cognate-coding noise of this size should be propagated (e.g.
  re-run at 0.65/0.75/0.85) and simulated before dates are interpreted.
* **Level-1 detector on real data.** The highest-scoring entries are short,
  pan-Mayan retentions (*uk'* "louse", *im* "breast", *ha'* "water"), not
  loans. Identical forms in distant languages look loan-like to the model.
  The simulator has no segment-specific stability, so it never produced this
  failure. Follow-up tests (8 simulated replicates each, κ = 0.5):

  | Simulated phonology | AUC |
  |---|---|
  | default (2–3 syllable roots) | 0.80 |
  | + stable segments (`stable_segments`, weight 0.05) | 0.74 |
  | Mayan-like CVC roots | 0.72 |
  | CVC + stable segments | **0.67** |

  So the milestone 3 AUCs are optimistic for a family like Mayan.
  Concept-wise normalisation (subtracting each concept's median score)
  *lowered* AUC (0.80 → 0.65), and normalising pair LLRs by word length had no
  effect; both were dropped. Separating shared retentions from loans will
  probably need more than form similarity, such as a model of segment-specific
  stability.

**Milestone 5 calibrations (draft, to be reviewed before any date is
reported)** are in `infer/beast/m5_mayan_prior.json`. Every point carries
its source and status. Glottochronological dates, including Kaufman's
c. 2200 BC for Proto-Mayan, are deliberately excluded: using them as
calibrations would make the test circular.

| Node | Prior (years before AD 2000) | Basis |
|---|---|---|
| root (Proto-Mayan) | Uniform(1500, 9000) | spans all published estimates (c. 6600 BC to c. 209 BC) |
| Yucatecan crown (monophyletic) | Normal(1050, 200) | Proto-Yucatecan c. AD 950 (Hofling 2018) |
| Huastecan crown (monophyletic) | Uniform(480, 1100) | Chicomuceltec–Huastec split Postclassic (Kaufman 1980; Kaufman & Justeson 2008; Robertson & Houston 2015) |
| *sensitivity only:* Cholan crown | Uniform(1750, 4000) | Classic Ch'olti'an in inscriptions from c. AD 250 (Houston, Robertson & Stuart 2000). **Contested**: Mora-Marín argues Cholan diversification is Postclassic |

The loss-rate prior, LogNormal(−9, 1) with mean about 2e-4 per year, also
carries absolute-time information and needs the same review. Variants:
strict and relaxed clocks; + Cholan calibration; cognate thresholds 0.65 and
0.85.

**Kaufmanpmed checked and rejected for dating.** `kaufmanpmed` cannot be made into a
basic-vocabulary dataset. None of its 5,932 glosses is linked to Concepticon,
and its `Loan` column is empty. Matching the Swadesh-100 concepts to etymon
glosses finds 22 concepts, only 14 of which are attested in at least 20 of
the 104 modern varieties. Searches of the Lexibank and CLDF-datasets
organisations and the literature found no cognate-coded Mayan Swadesh
dataset. The realistic options:

1. **ASJP** (`lexibank/asjp`, 40-item basic lists for nearly every Mayan
   variety) with **automated cognate detection** (LexStat), as in
   Rama & Wichmann. Fully reproducible. Cognate errors become an extra noise
   source, which the simulator could model.
2. Request the cognate-coded Swadesh data behind published Mayan
   phylogenies (e.g. Atkinson 2006) from the authors.
3. Expert cognate coding, which is out of scope.

Raw data under `data/raw/` is fetched by `scripts/fetch_data.sh`, made
read-only, and never edited. Everything under `data/derived/` is regenerated
by `make`.

## Layout

```
env/          pinned Python lockfile; BEAST2 installer
infer/        BEAST2 XML generation, runner, loan masking; beast/ = templates, prior specs, babelfix
data/raw/     fetched CLDF datasets (git-ignored, pinned commits)
data/derived/ cognate matrices, simulated datasets (git-ignored, rebuilt by make)
sim/          forward simulator: tree.py, sdollo.py, borrowing.py, phonology.py
pipeline/     CLDF I/O shared by real and simulated data
scripts/      CLI entry points used by the Makefile
infer/cpp/    Track B (later)
hmm/          pair HMM (level 1) and latent-donor mixture (level 2)
analysis/     bias curves, ROC, figures
tests/
figures/
```
