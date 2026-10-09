import numpy as np
import pandas as pd

from src.a6_training import (
    LABELS,A6_VARIANTS,prepare_clean_a6_targets,target_loss_weights
)

def _frames():
    rows=[]
    for i in range(6):
        r={"StudyInstanceUID":f"u{i}"}
        for j,l in enumerate(LABELS):
            r[l]=np.nan
        rows.append(r)
    # two gold studies
    for idx in [1,4]:
        for j,l in enumerate(LABELS):
            rows[idx][l]=float((idx+j)%2)
    train=pd.DataFrame(rows)

    pseudo=[]
    for i in range(6):
        r={"StudyInstanceUID":f"u{i}"}
        for j,l in enumerate(LABELS):
            r[l]=0.9 if (i+j)%2 else 0.1
        pseudo.append(r)
    return train,pd.DataFrame(pseudo)

def test_gold_is_hard_excluded():
    train,pseudo=_frames()
    out=prepare_clean_a6_targets(train,pseudo)
    assert out["n_gold"]==2
    assert out["n_train"]==4
    assert set(out["gold_uids"])=={"u1","u4"}
    assert not (set(out["gold_uids"]) & set(out["train_uids"]))
    assert out["train_targets"].shape==(4,12)

def test_gold_values_never_replace_weak_targets():
    train,pseudo=_frames()
    out=prepare_clean_a6_targets(train,pseudo)
    # all surviving targets must be the pseudo-label values 0.1/0.9
    assert set(np.unique(out["train_targets"])).issubset({0.1,0.9})

def test_variant_focus_weights():
    pf=target_loss_weights(A6_VARIANTS["A6-PF"])
    syn=target_loss_weights(A6_VARIANTS["A6-SYN"])
    lat=target_loss_weights(A6_VARIANTS["A6-LAT"])
    assert pf[LABELS.index("PF OA")]==2.0
    assert syn[LABELS.index("Synovitis")]==2.0
    assert lat[LABELS.index("Lateral OA")]==1.75
    assert lat[LABELS.index("Lateral Meniscus")]==1.75
    assert pf[LABELS.index("ACL")]==1.0
