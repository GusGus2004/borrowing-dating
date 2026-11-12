import numpy as np

from infer.correction import mask_entries
from pipeline.cldf import MISSING, CognateMatrix


def test_mask_whole_concept_and_drop_empty_sets():
    cm = CognateMatrix(languages=["A", "B", "C"], cognatesets=["s1", "s2", "s3"],
                       concepts=["c1", "c1", "c2"],
                       matrix=np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=np.int8))
    out, st = mask_entries(cm, [("A", "s1"), ("Z", "s1")])
    # A's row for concept c1 is missing; s1 has no remaining 1 and is dropped
    assert out.cognatesets == ["s2", "s3"]
    assert out.matrix[0, 0] == MISSING and out.matrix[0, 1] == 1
    assert st["masked_language_concepts"] == 1 and st["dropped_cognatesets"] == 1
    assert np.isclose(st["masked_fraction"], 1 / 6)
