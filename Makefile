# Everything reproducible from `make`. Seeds are fixed here and recorded in truth.json.
PY      := .venv/bin/python
KAUFMAN := data/raw/kaufmanpmed/cldf/cldf-metadata.json
ASJP    := data/raw/asjp/cldf/cldf-metadata.json

.PHONY: all env data matrices sim test beast beast-smoke babelfix beast-debug-relaxed m1-strict m1-relaxed m2 clean

all: env data matrices sim test

env: .venv/.done
.venv/.done: env/requirements.lock.txt
	python3 -m venv .venv
	.venv/bin/pip install -q -r env/requirements.lock.txt
	touch $@

BEAST   := $(CURDIR)/env/beast2/bin/beast -packagedir $(CURDIR)/env/beast2-packages

beast:
	./env/install_beast2.sh

# Milestone 0 check: Babel's SDollo example, shortened to 2e5 states (~4 min).
beast-smoke:
	cd infer/beast/smoke && $(BEAST) -overwrite -seed 1 testSDollo.xml > beast.out 2>&1 \
	    && tail -3 beast.out

data: $(KAUFMAN) $(ASJP)
$(KAUFMAN) $(ASJP):
	./scripts/fetch_data.sh

matrices: data/derived/kaufmanpmed/matrix.nex data/derived/asjp_mayan/matrix.nex

# Mayan ASJP core vocabulary, LexStat cognates (milestone 5 data)
data/derived/asjp_mayan/matrix.nex: $(ASJP) pipeline/asjp.py pipeline/cldf.py scripts/build_asjp_mayan.py | env
	$(PY) scripts/build_asjp_mayan.py
data/derived/kaufmanpmed/matrix.nex: $(KAUFMAN) pipeline/cldf.py scripts/build_matrix.py | env
	$(PY) scripts/build_matrix.py $(KAUFMAN) data/derived/kaufmanpmed --no-concept-missing

sim: data/derived/sim/example/Wordlist-metadata.json
data/derived/sim/example/Wordlist-metadata.json: sim/*.py pipeline/cldf.py scripts/simulate.py | env
	$(PY) scripts/simulate.py --out data/derived/sim/example --seed 1 \
	    --n-tips 10 --root-age 4000 --n-concepts 200 --birth 2e-4 --loss 2e-4 \
	    --gamma-shape 0.5 --clock-sd 0.3
	$(PY) scripts/build_matrix.py $@ data/derived/sim/example/matrix

test: | env
	$(PY) -m pytest -q

babelfix:
	./infer/beast/babelfix/build.sh

# Likelihood-consistency check (BEAST recomputes from scratch every step) on a relaxed-clock replicate.
beast-debug-relaxed: | env
	$(PY) scripts/m1_coverage.py --clock relaxed prepare --n 1
	cd data/derived/m1_relaxed/rep_000 && sed 's/chainLength="[0-9]*"/chainLength="20000"/; s/fileName="beast\./fileName="debug./' beast.xml > debug.xml \
	    && $(CURDIR)/env/beast2/jre/bin/java -Dbeast.debug=true -Xss256m -cp $(CURDIR)/env/beast2/lib/launcher.jar \
	       beast.pkgmgmt.launcher.BeastLauncher -packagedir $(CURDIR)/env/beast2-packages -overwrite -seed 1 debug.xml

# Milestone 1 coverage studies (~25 min strict on 6 cores).
m1-strict: | env
	$(PY) scripts/m1_coverage.py --clock strict prior-check
	$(PY) scripts/m1_coverage.py --clock strict prepare --n 100
	$(PY) scripts/m1_coverage.py --clock strict run --jobs 6
	$(PY) scripts/m1_coverage.py --clock strict summarise

m1-relaxed: babelfix beast-debug-relaxed
	$(PY) scripts/m1_coverage.py --clock relaxed prior-check
	$(PY) scripts/m1_coverage.py --clock relaxed prepare --n 100
	$(PY) scripts/m1_coverage.py --clock relaxed run --jobs 6
	$(PY) scripts/m1_coverage.py --clock relaxed summarise

# Milestone 2 bias curve (~1 h on 7 cores; needs no babelfix: strict clock).
m2: | env
	$(PY) scripts/m2_bias.py prepare
	$(PY) scripts/m2_bias.py run --jobs 6
	$(PY) scripts/m2_bias.py summarise

clean:
	rm -rf data/derived/* && touch data/derived/.gitkeep
