"""Milestone 5c diagnostic: pairwise recall and false-merge rate of automated cognates vs Kaufman by split depth (Glottolog classification levels shared)."""
import sys, csv, collections, itertools
sys.path[:0] = ['.', 'scripts']
import io, contextlib
from m5c_asjp_expert import labelled_forms, MIN_LABELLED
from pipeline.asjp import lexstat_cognates, partial_cognates
with contextlib.redirect_stderr(io.StringIO()):
    languages, forms, concept, gold = labelled_forms()
    n = collections.Counter(f["Language_ID"] for f in forms if f["ID"] in gold)
    forms = [f for f in forms if n[f["Language_ID"]] >= MIN_LABELLED]
    lex = lexstat_cognates(forms, 0.75); part = partial_cognates(forms, 0.85, method="lexstat")
cls = {r["ID"]: r["classification_glottolog"].split(",") for r in csv.DictReader(open("data/raw/asjp/cldf/languages.csv")) if r["Family"] == "Mayan"}
lang = {f["ID"]: f["Language_ID"] for f in forms}
def shared(a, b):
    k = 0
    for x, y in zip(cls[a], cls[b]):
        if x != y: break
        k += 1
    return k
lab = [f["ID"] for f in forms if f["ID"] in gold]
by = collections.defaultdict(list)
for f in lab: by[concept[f]].append(f)
stat = {m: collections.defaultdict(lambda: [0, 0, 0, 0]) for m in ("LexStat", "partial")}
for c, fs in by.items():
    for a, b in itertools.combinations(fs, 2):
        if lang[a] == lang[b]: continue
        d = min(shared(lang[a], lang[b]), 4)
        for m, cog in (("LexStat", lex), ("partial", part)):
            s = stat[m][d]
            if gold[a] == gold[b]:
                s[0] += 1; s[1] += cog[a] == cog[b]
            else:
                s[2] += 1; s[3] += cog[a] == cog[b]
names = {1: "1 (only 'Mayan': deepest split)", 2: "2", 3: "3", 4: "4+ (same low-level subgroup)"}
for m in stat:
    print(m)
    for d in sorted(stat[m]):
        t, h, f, fm = stat[m][d]
        print(f"  shared classification levels {names.get(d, d):34s} true-cognate pairs {t:5d}  found {h/t:.2f} | non-cognate pairs {f:5d}  wrongly merged {fm/max(f,1):.3f}")
