# Everything reproducible from `make`. Seeds are fixed here and recorded in truth.json.
PY      := .venv/bin/python
KAUFMAN := data/raw/kaufmanpmed/cldf/cldf-metadata.json

.PHONY: all env data matrices sim test beast clean

all: env data matrices sim test

env: .venv/.done
.venv/.done: env/requirements.lock.txt
	python3 -m venv .venv
	.venv/bin/pip install -q -r env/requirements.lock.txt
	touch $@

beast:
	./env/install_beast2.sh

data: $(KAUFMAN)
$(KAUFMAN):
	./scripts/fetch_data.sh

matrices: data/derived/kaufmanpmed/matrix.nex
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

clean:
	rm -rf data/derived/* && touch data/derived/.gitkeep
