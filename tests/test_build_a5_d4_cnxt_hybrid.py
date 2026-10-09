import json
from scripts.build_a5_d4_cnxt_hybrid import union, TARGET_WEIGHTS

def test_union_stable_unique():
    assert union(["a","b"],["b","c"])==["a","b","c"]

def test_target_weights_contract():
    assert len(TARGET_WEIGHTS)==12
    assert all(0 <= x <= .5 for x in TARGET_WEIGHTS)
    assert TARGET_WEIGHTS[6] == .05
    assert TARGET_WEIGHTS[8] == .05
