"""Build a cognate matrix (CSV + NEXUS + summary) from any CLDF Wordlist.

Usage:
    python scripts/build_matrix.py <metadata.json> <outdir> [--no-concept-missing]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.cldf import MISSING, read_cognate_matrix  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata")
    ap.add_argument("outdir")
    ap.add_argument("--no-concept-missing", action="store_true",
                    help="code absence as 0 rather than concept-level missing "
                         "(use for etymological dictionaries)")
    args = ap.parse_args()

    cm = read_cognate_matrix(args.metadata, concept_level_missing=not args.no_concept_missing)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cm.to_csv(out / "matrix.csv")
    cm.to_nexus(out / "matrix.nex")
    m = cm.matrix
    summary = {
        "source": str(args.metadata),
        "n_languages": len(cm.languages),
        "n_cognatesets": len(cm.cognatesets),
        "n_concepts": len(set(cm.concepts)),
        "fraction_present": float((m == 1).mean()),
        "fraction_missing": float((m == MISSING).mean()),
        "singletons": int(((m == 1).sum(axis=0) == 1).sum()),
        "cognatesets_per_language": dict(zip(cm.languages, (m == 1).sum(axis=1).tolist())),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{summary['n_languages']} languages x {summary['n_cognatesets']} cognate sets "
          f"-> {out}")


if __name__ == "__main__":
    main()
