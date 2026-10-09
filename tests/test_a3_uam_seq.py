import unittest

import numpy as np
import torch

from src.a3_uam_seq import (
    UAMSeqConfig,
    UAMSeqHead,
    masked_soft_bce,
    uid_fold,
    unstated_aware_weight,
    validate_feature_batch,
)


class TestA3UAMSeq(unittest.TestCase):
    def test_unstated_weight_zero_at_half(self):
        y=torch.tensor([[0.0,0.25,0.5,0.75,1.0]])
        w=unstated_aware_weight(y)
        self.assertTrue(torch.allclose(w,torch.tensor([[1.0,0.5,0.0,0.5,1.0]])))

    def test_gold_overrides_unstated_mask(self):
        logits=torch.zeros((1,2),requires_grad=True)
        y=torch.tensor([[0.5,1.0]])
        gold=torch.tensor([[True,False]])
        loss=masked_soft_bce(logits,y,gold_mask=gold,gold_weight=4.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss),0.0)

    def test_all_unstated_batch_has_zero_differentiable_loss(self):
        logits=torch.randn((2,12),requires_grad=True)
        y=torch.full((2,12),0.5)
        loss=masked_soft_bce(logits,y)
        self.assertEqual(float(loss.detach()),0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_uid_fold_is_deterministic(self):
        uid="1.2.840.113619.2.55.3"
        self.assertEqual(uid_fold(uid),uid_fold(uid))
        self.assertTrue(0 <= uid_fold(uid) < 5)

    def test_model_forward_shape(self):
        cfg=UAMSeqConfig(feature_dim=32,projection_dim=16,metadata_dim=8,
                         hidden_dim=12,gru_layers=1,num_targets=12)
        model=UAMSeqHead(cfg)
        feat=torch.randn(3,7,32)
        types=torch.randint(0,8,(3,7))
        mask=torch.ones((3,7),dtype=torch.bool)
        logits=model(feat,types,mask)
        self.assertEqual(tuple(logits.shape),(3,12))
        self.assertTrue(torch.isfinite(logits).all())

    def test_feature_validation_rejects_empty_study(self):
        f=np.zeros((2,4,8),dtype=np.float32)
        t=np.zeros((2,4),dtype=np.int64)
        m=np.array([[1,1,0,0],[0,0,0,0]],dtype=bool)
        with self.assertRaises(ValueError):
            validate_feature_batch(f,t,m,feature_dim=8)


if __name__=="__main__":
    unittest.main()
