"""Build the Mayan ASJP core-vocabulary dataset (CLDF + cognate matrix)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.asjp import build  # noqa: E402
from pipeline.cldf import read_cognate_matrix  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--asjp", default=str(ROOT / "data/raw/asjp/cldf"))
ap.add_argument("--out", default=str(ROOT / "data/derived/asjp_mayan"))
ap.add_argument("--threshold", type=float, default=0.75)
a = ap.parse_args()
info = build(a.asjp, a.out, threshold=a.threshold)
cm = read_cognate_matrix(Path(a.out) / "Wordlist-metadata.json", concept_level_missing=True)
cm.to_nexus(Path(a.out) / "matrix.nex")
cm.to_csv(Path(a.out) / "matrix.csv")
info["matrix_shape"] = list(cm.shape)
info["missing_fraction"] = float((cm.matrix == -1).mean())
(Path(a.out) / "build.json").write_text(json.dumps(info, indent=2))
print(json.dumps({k: v for k, v in info.items() if k != "languages"}, indent=2))
