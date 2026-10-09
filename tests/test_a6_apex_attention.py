import numpy as np
import torch

from src.a6_apex_attention import (
    LABELS,SLOT_NAMES,TARGET_SLOT_ODDS,slot_index_for_depth,
    centers_for_full_coverage,build_target_window_log_prior
)

def test_slot_contract():
    s=slot_index_for_depth(96)
    assert len(s)==96
    assert [int((s==i).sum()) for i in range(5)]==[26,22,18,12,18]

def test_full_coverage_centers():
    mask=np.ones(96,dtype=np.uint8)
    c=centers_for_full_coverage(mask,96,94)
    assert len(c)==94
    assert c[0]==1 and c[-1]==94

def test_pf_prior_prefers_axial():
    c=list(range(1,95))
    p=build_target_window_log_prior(c,D=96,strength=1.0)
    q=LABELS.index("PF OA")
    s=slot_index_for_depth(96)[np.asarray(c)]
    axial=p[q,s==4].mean()
    sagittal=p[q,np.isin(s,[0,1])].mean()
    assert axial>sagittal

def test_synovitis_prefers_fluid_slots():
    c=list(range(1,95))
    p=build_target_window_log_prior(c,D=96,strength=1.0)
    q=LABELS.index("Synovitis")
    s=slot_index_for_depth(96)[np.asarray(c)]
    fs=p[q,np.isin(s,[0,2])].mean()
    nonfs=p[q,np.isin(s,[1,3])].mean()
    assert fs>nonfs

def test_zero_strength_is_neutral():
    p=build_target_window_log_prior(list(range(1,95)),D=96,strength=0.0)
    assert np.allclose(p,0)
