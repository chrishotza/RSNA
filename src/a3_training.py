from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from src.a3_uam_seq import UAMSeqConfig, UAMSeqHead, masked_soft_bce, uam_rank_objective


Mode = Literal["A3-1", "A3-2", "A3-3"]


@dataclass(frozen=True)
class TrainVariant:
    name: Mode
    use_rank_loss: bool
    use_target_router: bool
    point_weight: float = 0.35
    rank_weight: float = 0.65
    min_target_gap: float = 0.35


VARIANTS: dict[Mode, TrainVariant] = {
    "A3-1": TrainVariant("A3-1", use_rank_loss=False, use_target_router=False),
    "A3-2": TrainVariant("A3-2", use_rank_loss=True, use_target_router=False),
    "A3-3": TrainVariant("A3-3", use_rank_loss=True, use_target_router=True),
}


class FeatureBankDataset(Dataset):
    def __init__(self, records: list[dict]):
        if not records:
            raise ValueError("empty dataset")
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "uid": str(r["uid"]),
            "features": torch.as_tensor(r["features"], dtype=torch.float32),
            "token_types": torch.as_tensor(r["token_types"], dtype=torch.long),
            "targets": torch.as_tensor(r["targets"], dtype=torch.float32),
            "gold_mask": torch.as_tensor(r.get("gold_mask", np.zeros(12, dtype=bool)), dtype=torch.bool),
        }


def collate_feature_bank(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("empty batch")
    feature_dim = batch[0]["features"].shape[-1]
    max_t = max(x["features"].shape[0] for x in batch)
    b = len(batch)

    features = torch.zeros((b, max_t, feature_dim), dtype=torch.float32)
    token_types = torch.zeros((b, max_t), dtype=torch.long)
    token_mask = torch.zeros((b, max_t), dtype=torch.bool)
    targets = torch.stack([x["targets"] for x in batch])
    gold_mask = torch.stack([x["gold_mask"] for x in batch])

    for i, x in enumerate(batch):
        t = x["features"].shape[0]
        if x["token_types"].shape[0] != t:
            raise ValueError("token_types/features length mismatch")
        features[i, :t] = x["features"]
        token_types[i, :t] = x["token_types"]
        token_mask[i, :t] = True

    return {
        "uids": [x["uid"] for x in batch],
        "features": features,
        "token_types": token_types,
        "token_mask": token_mask,
        "targets": targets,
        "gold_mask": gold_mask,
    }


def build_model(feature_dim: int, variant: TrainVariant, **overrides) -> UAMSeqHead:
    cfg = UAMSeqConfig(
        feature_dim=feature_dim,
        use_target_type_router=variant.use_target_router,
        **overrides,
    )
    return UAMSeqHead(cfg)


def compute_objective(logits, targets, gold_mask, variant: TrainVariant):
    if variant.use_rank_loss:
        return uam_rank_objective(
            logits,
            targets,
            gold_mask,
            point_weight=variant.point_weight,
            rank_weight=variant.rank_weight,
            min_target_gap=variant.min_target_gap,
        )
    point = masked_soft_bce(logits, targets, gold_mask=gold_mask)
    return point, {"point_bce": point.detach(), "pair_rank": point.detach() * 0.0}


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    # average ranks for ties
    vals = y_score[order]
    start = 0
    while start < len(vals):
        end = start + 1
        while end < len(vals) and vals[end] == vals[start]:
            end += 1
        if end - start > 1:
            avg = (start + 1 + end) / 2.0
            ranks[order[start:end]] = avg
        start = end

    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, np.ndarray]:
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError("expected matching [N,Q] arrays")
    per = np.array([binary_auc(y_true[:, q], y_score[:, q]) for q in range(y_true.shape[1])])
    finite = np.isfinite(per)
    return (float(per[finite].mean()) if finite.any() else float("nan"), per)
