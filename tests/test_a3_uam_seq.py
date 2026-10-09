import unittest
import numpy as np
import torch
from src.a3_uam_seq import UAMSeqConfig,UAMSeqHead,masked_soft_bce,pairwise_auc_loss,uam_rank_objective,uid_fold,unstated_aware_weight,validate_feature_batch

class TestA3UAMSeq(unittest.TestCase):
    def test_unstated_weight_zero_at_half(self):
        y=torch.tensor([[0.0,0.25,0.5,0.75,1.0]])
        self.assertTrue(torch.allclose(unstated_aware_weight(y),torch.tensor([[1.0,0.5,0.0,0.5,1.0]])))
    def test_gold_overrides_unstated_mask(self):
        z=torch.zeros((1,2),requires_grad=True); y=torch.tensor([[0.5,1.0]]); g=torch.tensor([[True,False]])
        self.assertGreater(float(masked_soft_bce(z,y,g)),0.0)
    def test_all_unstated_zero_loss(self):
        z=torch.randn((2,12),requires_grad=True); y=torch.full((2,12),0.5)
        loss=masked_soft_bce(z,y); self.assertEqual(float(loss.detach()),0.0); loss.backward(); self.assertIsNotNone(z.grad)
    def test_pairwise_auc_prefers_correct_order(self):
        y=torch.tensor([[1.0],[0.0],[0.5]])
        good=torch.tensor([[3.0],[-2.0],[0.0]],requires_grad=True)
        bad=torch.tensor([[-2.0],[3.0],[0.0]],requires_grad=True)
        self.assertLess(float(pairwise_auc_loss(good,y).detach()),float(pairwise_auc_loss(bad,y).detach()))
    def test_unstated_cannot_create_rank_pair(self):
        y=torch.full((4,2),0.5); z=torch.randn((4,2),requires_grad=True)
        loss=pairwise_auc_loss(z,y); self.assertEqual(float(loss.detach()),0.0); loss.backward(); self.assertIsNotNone(z.grad)
    def test_hybrid_objective_finite(self):
        z=torch.randn((6,12),requires_grad=True)
        y=torch.tensor([[0.,1.,.5,.25,.75,1.,0.,.5,1.,0.,.75,.25]]*6)
        total,parts=uam_rank_objective(z,y); self.assertTrue(torch.isfinite(total)); self.assertIn("pair_rank",parts)
    def test_uid_fold_deterministic(self):
        uid="1.2.840.113619.2.55.3"; self.assertEqual(uid_fold(uid),uid_fold(uid))
    def test_model_forward_shape(self):
        cfg=UAMSeqConfig(feature_dim=32,projection_dim=16,metadata_dim=8,hidden_dim=12,gru_layers=1,num_targets=12,metadata_vocab=8)
        model=UAMSeqHead(cfg); feat=torch.randn(3,7,32); types=torch.randint(0,8,(3,7)); mask=torch.ones((3,7),dtype=torch.bool)
        logits=model(feat,types,mask); self.assertEqual(tuple(logits.shape),(3,12)); self.assertTrue(torch.isfinite(logits).all())
    def test_target_type_router_receives_gradient(self):
        cfg=UAMSeqConfig(feature_dim=16,projection_dim=12,metadata_dim=6,hidden_dim=8,gru_layers=1,num_targets=12,metadata_vocab=5)
        model=UAMSeqHead(cfg); feat=torch.randn(2,6,16); types=torch.tensor([[0,1,2,3,4,0],[1,2,3,4,0,1]]); mask=torch.ones((2,6),dtype=torch.bool)
        model(feat,types,mask).sum().backward(); self.assertIsNotNone(model.target_type_bias.grad)
    def test_feature_validation_rejects_empty_study(self):
        f=np.zeros((2,4,8),dtype=np.float32); t=np.zeros((2,4),dtype=np.int64); m=np.array([[1,1,0,0],[0,0,0,0]],dtype=bool)
        with self.assertRaises(ValueError): validate_feature_batch(f,t,m,feature_dim=8)

if __name__=="__main__": unittest.main()
