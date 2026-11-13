"""Build data/derived/pmed_mayan: Swadesh-100 Mayan wordlist with Kaufman's etyma (see pipeline/pmed.py)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.pmed import build  # noqa: E402

if __name__ == "__main__":
    out = ROOT / "data/derived/pmed_mayan"
    r = build(ROOT / "data/raw/kaufmanpmed/cldf", out)
    (out / "build.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in r.items() if k != "concept_coverage"}, indent=1, ensure_ascii=False))
