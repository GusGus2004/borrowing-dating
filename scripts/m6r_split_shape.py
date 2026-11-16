"""How LexStat splits gold cognate sets: real ASJP/Kaufman words vs MAYAN_V2 simulations (see README, 6r V2)."""
import sys, collections, contextlib, io, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
import numpy as np
from pipeline.asjp import lexstat_cognates
from sim_realism import MAYAN_V2, simulate_forms

def shape(ids, gold, auto, lang):
    g = collections.defaultdict(set); a = collections.defaultdict(set)
    for i in ids: g[gold[i]].add(i); a[auto[i]].add(i)
    sz = np.array([len({lang[i] for i in s}) for s in a.values()])
    gsz = np.array([len({lang[i] for i in s}) for s in g.values()])
    # for gold sets with >=4 languages: share of languages in the largest fragment, #fragments, singleton fragments
    big, nfrag, single = [], [], []
    for s in g.values():
        L = {lang[i] for i in s}
        if len(L) < 4: continue
        frag = collections.defaultdict(set)
        for i in s: frag[auto[i]].add(lang[i])
        sizes = sorted((len(v) for v in frag.values()), reverse=True)
        big.append(sizes[0]/len(L)); nfrag.append(len(sizes)); single.append(sum(x == 1 for x in sizes)/len(sizes))
    return dict(gold_sets=len(g), auto_sets=len(a), gold_mean_langs=gsz.mean(), auto_mean_langs=sz.mean(),
                auto_singleton=(sz == 1).mean(), gold_singleton=(gsz == 1).mean(),
                largest_frag_share=np.mean(big), frags_per_big_set=np.mean(nfrag), singleton_frag_share=np.mean(single))

from m5c_asjp_expert import MIN_LABELLED, labelled_forms
with contextlib.redirect_stderr(io.StringIO()):
    languages, forms, concept, gold = labelled_forms()
    n = collections.Counter(f["Language_ID"] for f in forms if f["ID"] in gold)
    forms = [f for f in forms if n[f["Language_ID"]] >= MIN_LABELLED]
    auto = lexstat_cognates(forms, 0.75)
ids = [f["ID"] for f in forms if f["ID"] in gold]
lang = {f["ID"]: f["Language_ID"] for f in forms}
print("real ", {k: round(float(v), 3) for k, v in shape(ids, gold, auto, lang).items()})
for seed in (7700, 7710, 7720):
    forms, gold, tree, mu, bf = simulate_forms({**MAYAN_V2, "kappa": 0.0}, seed)
    with contextlib.redirect_stderr(io.StringIO()):
        auto = lexstat_cognates(forms, 0.75)
    lang = {f["ID"]: f["Language_ID"] for f in forms}
    print("sim", seed, {k: round(float(v), 3) for k, v in shape([f["ID"] for f in forms], gold, auto, lang).items()})
