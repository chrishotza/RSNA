import numpy as np

from src.a6_feature_bank import (
    MAX_ANCHORS,A6SeriesDescriptor,dense_union_centers,
    expected_bank_density,select_policy_token_mask,series_manifest
)

def test_dense_union_is_deterministic_unique():
    a=dense_union_centers(71)
    b=dense_union_centers(71)
    assert a==b
    assert a==sorted(set(a))
    assert len(a)>=MAX_ANCHORS
    assert min(a)>=0 and max(a)<71

def test_union_is_denser_than_a3_six_anchors():
    assert len(dense_union_centers(40))>6

def test_manifest_contract():
    s=[f"s{i}" for i in range(30)]
    m=series_manifest(s)
    assert len(m)==len(dense_union_centers(30))
    assert all(len(x["sop_triplet"])==3 for x in m)
    assert all(0<=x["relative_depth"]<=1 for x in m)

def test_policy_depth_masks():
    tt=np.array([12,12,12,12])
    depth=np.array([0.01,.10,.50,.99])
    mask=select_policy_token_mask(tt,depth,"PF OA")
    assert mask.tolist()==[False,True,True,False]

def test_series_descriptor_type():
    d=A6SeriesDescriptor("u","s","axial",True,True)
    assert d.canonical_plane=="AXIAL"
    assert d.type_id==15

def test_density_receipt():
    r=expected_bank_density(100)
    assert r["n_centers"]>6
    assert 0<r["coverage_fraction"]<=1
