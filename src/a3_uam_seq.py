from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
SEED = 20261009

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def uid_fold(uid: str, n_folds: int = 5) -> int:
    if n_folds < 2: raise ValueError("n_folds must be >= 2")
    digest = hashlib.sha256(str(uid).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds

def unstated_aware_weight(target: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(target).all(): raise ValueError("target contains non-finite values")
    if ((target < 0) | (target > 1)).any(): raise ValueError("target outside [0,1]")
    return torch.clamp(2.0 * torch.abs(target - 0.5), 0.0, 1.0)

def supervision_weight(targets, gold_mask=None, gold_weight: float = 4.0):
    weights = unstated_aware_weight(targets)
    if gold_mask is not None:
        if gold_mask.shape != targets.shape: raise ValueError("gold_mask shape mismatch")
        weights = torch.where(gold_mask.bool(), torch.full_like(weights, float(gold_weight)), weights)
    return weights

def masked_soft_bce(logits, targets, gold_mask=None, gold_weight: float = 4.0):
    if logits.shape != targets.shape: raise ValueError("shape mismatch")
    weights = supervision_weight(targets, gold_mask, gold_weight)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 0: return logits.sum() * 0.0
    return (loss * weights).sum() / denom

def pairwise_auc_loss(
    logits, targets, gold_mask=None, min_target_gap: float = 0.35,
    gold_weight: float = 4.0, temperature: float = 1.0,
):
    """Confidence-weighted AUC surrogate over reliable within-batch pairs."""
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("logits/targets must be equal [B,Q] tensors")
    if temperature <= 0: raise ValueError("temperature must be > 0")
    if not (0.0 < min_target_gap <= 1.0): raise ValueError("invalid min_target_gap")
    weights = supervision_weight(targets, gold_mask, gold_weight)
    total = logits.sum() * 0.0
    total_weight = logits.new_zeros(())
    for q in range(logits.shape[1]):
        y, z, w = targets[:, q], logits[:, q], weights[:, q]
        dy = y[:, None] - y[None, :]
        valid = dy >= min_target_gap
        if not valid.any(): continue
        pair_weight = torch.sqrt(torch.clamp(w[:, None] * w[None, :], min=0.0))
        pair_weight = pair_weight * valid.to(pair_weight.dtype)
        denom = pair_weight.sum()
        if float(denom.detach().cpu()) <= 0: continue
        dz = z[:, None] - z[None, :]
        total = total + (F.softplus(-dz / temperature) * pair_weight).sum()
        total_weight = total_weight + denom
    if float(total_weight.detach().cpu()) <= 0: return logits.sum() * 0.0
    return total / total_weight

def uam_rank_objective(
    logits, targets, gold_mask=None, *, point_weight: float = 0.35,
    rank_weight: float = 0.65, min_target_gap: float = 0.35,
    gold_weight: float = 4.0, temperature: float = 1.0,
):
    if point_weight < 0 or rank_weight < 0 or point_weight + rank_weight <= 0:
        raise ValueError("invalid objective weights")
    point = masked_soft_bce(logits, targets, gold_mask, gold_weight)
    rank = pairwise_auc_loss(logits, targets, gold_mask, min_target_gap, gold_weight, temperature)
    total = point_weight * point + rank_weight * rank
    return total, {"point_bce": point.detach(), "pair_rank": rank.detach()}

@dataclass(frozen=True)
class UAMSeqConfig:
    feature_dim: int = 2048
    hidden_dim: int = 256
    projection_dim: int = 384
    metadata_vocab: int = 32
    metadata_dim: int = 48
    num_targets: int = 12
    gru_layers: int = 2
    dropout: float = 0.15
    use_target_type_router: bool = True

class UAMSeqHead(nn.Module):
    """Ordered-window reader with target-specific acquisition routing."""
    def __init__(self, cfg: UAMSeqConfig = UAMSeqConfig()):
        super().__init__()
        self.cfg = cfg
        self.project = nn.Sequential(nn.LayerNorm(cfg.feature_dim), nn.Linear(cfg.feature_dim, cfg.projection_dim), nn.GELU(), nn.Dropout(cfg.dropout))
        self.meta = nn.Embedding(cfg.metadata_vocab, cfg.metadata_dim)
        self.gru = nn.GRU(cfg.projection_dim + cfg.metadata_dim, cfg.hidden_dim, num_layers=cfg.gru_layers, dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0, batch_first=True, bidirectional=True)
        token_dim = cfg.hidden_dim * 2
        self.token_norm = nn.LayerNorm(token_dim)
        self.queries = nn.Parameter(torch.randn(cfg.num_targets, token_dim) * 0.02)
        self.query_norm = nn.LayerNorm(token_dim)
        if cfg.use_target_type_router:
            self.target_type_bias = nn.Parameter(torch.zeros(cfg.num_targets, cfg.metadata_vocab))
        else:
            self.register_parameter("target_type_bias", None)
        self.classifier = nn.Sequential(nn.LayerNorm(token_dim * 4), nn.Linear(token_dim * 4, token_dim), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(token_dim, 1))

    def forward(self, features, token_types, token_mask, attention_log_prior=None):
        if features.ndim != 3: raise ValueError("features must be [B,T,F]")
        if token_types.shape != features.shape[:2]: raise ValueError("token_types shape mismatch")
        if token_mask.shape != features.shape[:2]: raise ValueError("token_mask shape mismatch")
        if ((token_types < 0) | (token_types >= self.cfg.metadata_vocab)).any(): raise ValueError("token type id outside metadata vocabulary")
        x = self.project(features)
        x = torch.cat([x, self.meta(token_types.long())], dim=-1)
        x, _ = self.gru(x)
        x = self.token_norm(x)
        valid = token_mask.bool()
        if (~valid).all(dim=1).any(): raise ValueError("study has no valid sequence tokens")
        q = self.query_norm(self.queries)
        scores = torch.einsum("btd,qd->bqt", x, q) / math.sqrt(x.shape[-1])
        if attention_log_prior is not None:
            if attention_log_prior.shape != scores.shape:
                raise ValueError("attention_log_prior must be [B,Q,T]")
            if not torch.isfinite(attention_log_prior).all():
                raise ValueError("attention_log_prior contains non-finite values")
            scores = scores + attention_log_prior.to(device=scores.device, dtype=scores.dtype)
        if self.target_type_bias is not None:
            scores = scores + self.target_type_bias[:, token_types.long()].permute(1, 0, 2)
        scores = scores.masked_fill(~valid[:, None, :], torch.finfo(scores.dtype).min)
        attn = scores.softmax(dim=-1)
        attended = torch.einsum("bqt,btd->bqd", attn, x)
        mask_f = valid.float()
        mean = (x * mask_f[..., None]).sum(dim=1) / mask_f.sum(dim=1, keepdim=True)
        mean = mean[:, None, :].expand(-1, self.cfg.num_targets, -1)
        fused = torch.cat([attended, mean, torch.abs(attended - mean), attended * mean], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        if logits.shape[-1] != self.cfg.num_targets: raise RuntimeError("target cardinality drift")
        return logits

def validate_feature_batch(features, token_types, token_mask, feature_dim: int = 2048):
    if features.ndim != 3 or features.shape[-1] != feature_dim: raise ValueError("invalid feature tensor shape")
    if token_types.shape != features.shape[:2]: raise ValueError("token type tensor shape mismatch")
    if token_mask.shape != features.shape[:2]: raise ValueError("token mask shape mismatch")
    if not np.isfinite(features).all(): raise ValueError("non-finite feature bank values")
    if not np.asarray(token_mask).any(axis=1).all(): raise ValueError("one or more studies have zero valid tokens")

def fold_manifest(uids: Iterable[str], n_folds: int = 5) -> dict[str, int]:
    return {str(uid): uid_fold(str(uid), n_folds=n_folds) for uid in uids}
