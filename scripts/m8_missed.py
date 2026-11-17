import sys, collections, json, numpy as np, csv
sys.path[:0]=["/Users/georgelythgoe/Desktop/cambridge/phd","/Users/georgelythgoe/Desktop/cambridge/phd/scripts"]
from m8_correspondence import detect, family_data, classes
def main():
    for fam in ("mayan","ie"):
        taxa, forms, concept, gold, _, _ = family_data(fam)
        d = detect(fam)
        cod = d["codings"]["corr_z2"]
        seg = {f["ID"]: classes(f["Segments"]) for f in forms}
        form = {f["ID"]: f["Form"] for f in forms}
        lang = {f["ID"]: f["Language_ID"] for f in forms}
        rows = []
        for k, z in d["z"].items():
            a, b = k.split("\t")
            if a in gold and b in gold and gold[a] == gold[b]:
                la, lb = len(seg[a]), len(seg[b])
                rows.append((z, cod.get(a) == cod.get(b), abs(la-lb), max(la,lb)/max(1,min(la,lb)), a, b))
        miss = [r for r in rows if not r[1]]; hit = [r for r in rows if r[1]]
        print(f"== {fam}: true cognate pairs {len(rows)}, missed {len(miss)} ({len(miss)/len(rows):.0%})")
        for name, R in (("hit", hit), ("missed", miss)):
            print(f"  {name:6s} mean z {np.mean([r[0] for r in R]):5.2f}; length diff >=2: {np.mean([r[2]>=2 for r in R]):.0%}; len ratio >=1.5: {np.mean([r[3]>=1.5 for r in R]):.0%}; z<0: {np.mean([r[0]<0 for r in R]):.0%}")
        rng = np.random.default_rng(0)
        print("  sample of missed pairs (z, forms):")
        for i in rng.choice(len(miss), 12, replace=False):
            z, _, _, _, a, b = miss[i]
            print(f"    z {z:5.2f}  {lang[a]}:{form[a]}  ~  {lang[b]}:{form[b]}")


if __name__ == '__main__':
    main()
