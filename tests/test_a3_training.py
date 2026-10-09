import unittest
import numpy as np
import torch

from src.a3_training import (
    VARIANTS,
    FeatureBankDataset,
    binary_auc,
    build_model,
    collate_feature_bank,
    compute_objective,
    macro_auc,
)


class TestA3Training(unittest.TestCase):
    def test_variants_change_only_intended_axes(self):
        self.assertFalse(VARIANTS["A3-1"].use_rank_loss)
        self.assertFalse(VARIANTS["A3-2"].use_target_router)
        self.assertTrue(VARIANTS["A3-3"].use_target_router)

    def test_variable_length_collate(self):
        records = [
            {"uid":"a","features":np.ones((2,8)),"token_types":np.array([1,2]),"targets":np.zeros(12)},
            {"uid":"b","features":np.ones((4,8)),"token_types":np.array([1,2,3,4]),"targets":np.ones(12)},
        ]
        batch = collate_feature_bank([FeatureBankDataset(records)[0], FeatureBankDataset(records)[1]])
        self.assertEqual(tuple(batch["features"].shape),(2,4,8))
        self.assertEqual(int(batch["token_mask"][0].sum()),2)
        self.assertEqual(int(batch["token_mask"][1].sum()),4)

    def test_auc_perfect_and_reverse(self):
        y=np.array([0,0,1,1])
        self.assertEqual(binary_auc(y,np.array([0.1,0.2,0.8,0.9])),1.0)
        self.assertEqual(binary_auc(y,np.array([0.9,0.8,0.2,0.1])),0.0)

    def test_auc_ties(self):
        y=np.array([0,1])
        self.assertEqual(binary_auc(y,np.array([0.5,0.5])),0.5)

    def test_macro_auc(self):
        y=np.array([[0,1],[1,0],[0,1],[1,0]])
        s=np.array([[0.1,0.9],[0.9,0.1],[0.2,0.8],[0.8,0.2]])
        m,per=macro_auc(y,s)
        self.assertEqual(m,1.0)
        self.assertTrue(np.all(per==1.0))

    def test_objective_ablation(self):
        z=torch.randn(8,12,requires_grad=True)
        y=torch.randint(0,2,(8,12)).float()
        g=torch.zeros_like(y,dtype=torch.bool)
        l1,p1=compute_objective(z,y,g,VARIANTS["A3-1"])
        l2,p2=compute_objective(z,y,g,VARIANTS["A3-2"])
        self.assertTrue(torch.isfinite(l1))
        self.assertTrue(torch.isfinite(l2))
        self.assertEqual(float(p1["pair_rank"]),0.0)

    def test_build_model_router_flag(self):
        m2=build_model(16,VARIANTS["A3-2"],projection_dim=8,metadata_dim=4,hidden_dim=6,gru_layers=1,metadata_vocab=16)
        m3=build_model(16,VARIANTS["A3-3"],projection_dim=8,metadata_dim=4,hidden_dim=6,gru_layers=1,metadata_vocab=16)
        self.assertIsNone(m2.target_type_bias)
        self.assertIsNotNone(m3.target_type_bias)


if __name__=="__main__":
    unittest.main()
