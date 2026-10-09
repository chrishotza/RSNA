import numpy as np
import pytest
import torch

from src.a6_runner import (
    A6TrainConfig,a6_point_loss,a6_pairwise_auc_loss,assert_training_records_are_gold_free,
    focus_metric,selection_score,torch_attention_prior,assemble_oof
)
from src.a6_training import A6_VARIANTS,LABELS

def test_gold_leakage_guard():
    records=[{"uid":"a"},{"uid":"b"}]
    assert_training_records_are_gold_free(records,["z"])
    with pytest.raises(RuntimeError):
        assert_training_records_are_gold_free(records,["b"])

def test_target_focus_changes_point_loss():
    z=torch.zeros(4,12)
    y=torch.zeros(4,12)
    y[:,LABELS.index("PF OA")]=1.0
    pf=a6_point_loss(z,y,A6_VARIANTS["A6-PF"])
    syn=a6_point_loss(z,y,A6_VARIANTS["A6-SYN"])
    assert torch.isfinite(pf) and torch.isfinite(syn)
    assert float(pf) != float(syn)

def test_pairwise_loss_finite():
    torch.manual_seed(3)
    z=torch.randn(16,12)
    y=torch.randint(0,2,(16,12)).float()
    loss=a6_pairwise_auc_loss(z,y,A6_VARIANTS["A6-LAT"])
    assert torch.isfinite(loss)

def test_attention_prior_contract():
    tt=torch.tensor([[12,4,15,8],[8,12,4,15]])
    p=torch_attention_prior(tt,A6_VARIANTS["A6-PF"])
    assert p.shape==(2,12,4)
    assert torch.isfinite(p).all()

def test_focus_metric_prefers_focus_signal():
    n=20
    y=np.zeros((n,12),np.float32)
    pred=np.zeros((n,12),np.float32)
    idx=LABELS.index("PF OA")
    y[n//2:,idx]=1
    pred[:,idx]=np.linspace(0,1,n)
    m=focus_metric(y,pred,A6_VARIANTS["A6-PF"])
    assert m["focus_auc"]==1.0
    assert np.isfinite(selection_score(m))

def test_oof_coverage_guard():
    fold_results=[
      {"uids":["a"],"pred":np.zeros((1,12)),"targets":np.zeros((1,12))},
      {"uids":["b"],"pred":np.ones((1,12)),"targets":np.ones((1,12))},
    ]
    p,y=assemble_oof(fold_results,["a","b"])
    assert p.shape==(2,12)
    with pytest.raises(RuntimeError):
        assemble_oof(fold_results,["a","b","c"])
