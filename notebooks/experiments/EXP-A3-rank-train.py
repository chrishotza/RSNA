from __future__ import annotations

# === EMBEDDED A3 CORE ===
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

    def forward(self, features, token_types, token_mask):
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


# === EMBEDDED FEATURE BANK ===
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


PLANE_IDS = {
    "UNKNOWN": 0,
    "SAGITTAL": 1,
    "CORONAL": 2,
    "AXIAL": 3,
}


@dataclass(frozen=True)
class SliceRecord:
    sop_instance_uid: str
    image_position: tuple[float, float, float] | None
    image_orientation: tuple[float, float, float, float, float, float] | None
    instance_number: int | None


def _vec(values: Sequence[float], n: int) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"expected finite vector of length {n}")
    return a


def physical_slice_coordinate(
    image_position: Sequence[float],
    image_orientation: Sequence[float],
) -> float:
    """Project ImagePositionPatient onto the DICOM slice normal."""
    pos = _vec(image_position, 3)
    ori = _vec(image_orientation, 6)
    row = ori[:3]
    col = ori[3:]
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm <= 1e-8:
        raise ValueError("degenerate ImageOrientationPatient")
    normal = normal / norm
    return float(np.dot(pos, normal))


def sort_slice_records(records: Sequence[SliceRecord]) -> list[SliceRecord]:
    """Deterministic anatomical sorting with geometry-first fallback.

    Geometry is used only when every slice has a valid physical coordinate.
    Otherwise the entire series falls back to InstanceNumber, never a mixture
    of incomparable coordinate systems.
    """
    if not records:
        raise ValueError("empty series")

    coords: list[float] = []
    geometry_ok = True
    for r in records:
        if r.image_position is None or r.image_orientation is None:
            geometry_ok = False
            break
        try:
            coords.append(physical_slice_coordinate(r.image_position, r.image_orientation))
        except ValueError:
            geometry_ok = False
            break

    if geometry_ok:
        paired = list(zip(records, coords))
        paired.sort(key=lambda rc: (rc[1], rc[0].sop_instance_uid))
        return [r for r, _ in paired]

    if not all(r.instance_number is not None for r in records):
        raise ValueError("series lacks complete geometry and complete InstanceNumber fallback")

    return sorted(records, key=lambda r: (int(r.instance_number), r.sop_instance_uid))


def deterministic_anchor_centers(
    n_slices: int,
    n_anchors: int = 12,
    low_fraction: float = 0.04,
    high_fraction: float = 0.96,
) -> list[int]:
    """Evenly cover a series while staying deterministic on short stacks."""
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if n_anchors <= 0:
        raise ValueError("n_anchors must be positive")
    if not (0.0 <= low_fraction <= high_fraction <= 1.0):
        raise ValueError("invalid anchor fractions")

    if n_slices == 1:
        return [0]

    lo = low_fraction * (n_slices - 1)
    hi = high_fraction * (n_slices - 1)
    raw = np.linspace(lo, hi, num=min(n_anchors, n_slices))
    centers = np.rint(raw).astype(int)
    centers = np.clip(centers, 0, n_slices - 1)

    # Preserve order while dropping duplicates created on short stacks.
    seen = set()
    out = []
    for c in centers.tolist():
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def adjacent_triplet(center: int, n_slices: int) -> tuple[int, int, int]:
    """Return a 2.5D triplet with edge replication."""
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if not (0 <= center < n_slices):
        raise ValueError("center outside series")
    return (
        max(0, center - 1),
        center,
        min(n_slices - 1, center + 1),
    )


def canonical_plane(value: str | None) -> str:
    s = (value or "").strip().upper()
    if s.startswith("SAG"):
        return "SAGITTAL"
    if s.startswith("COR"):
        return "CORONAL"
    if s.startswith("AX"):
        return "AXIAL"
    return "UNKNOWN"


def _binary_flag(value: bool | int | float | str | None) -> int:
    """Map only explicit truthy protocol values to 1; unknown/NaN -> 0."""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "t"}:
            return 1
        if text in {"0", "false", "no", "n", "f", "", "nan", "none", "<na>", "unknown"}:
            return 0
        try:
            value = float(text)
        except ValueError:
            return 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(number):
        return 0
    return 1 if number == 1.0 else 0


def acquisition_type_id(
    anatomical_plane: str | None,
    fluid_sensitive: bool | int | float | str | None,
    fat_suppression: bool | int | float | str | None,
) -> int:
    """Stable compact ID: plane x fluid x fat suppression.

    Unknown protocol flags map to 0 rather than Python's dangerous bool(NaN).
    IDs occupy [0, 15], leaving room in the model vocabulary for future
    sequence metadata without changing existing IDs.
    """
    plane_id = PLANE_IDS[canonical_plane(anatomical_plane)]
    fluid = _binary_flag(fluid_sensitive)
    fat = _binary_flag(fat_suppression)
    return plane_id * 4 + fluid * 2 + fat


def build_series_window_manifest(
    sorted_sops: Sequence[str],
    n_anchors: int = 12,
) -> list[dict[str, object]]:
    centers = deterministic_anchor_centers(len(sorted_sops), n_anchors=n_anchors)
    out: list[dict[str, object]] = []
    for rank, center in enumerate(centers):
        a, b, c = adjacent_triplet(center, len(sorted_sops))
        out.append(
            {
                "anchor_rank": rank,
                "center_index": center,
                "sop_triplet": [sorted_sops[a], sorted_sops[b], sorted_sops[c]],
            }
        )
    return out


# === EMBEDDED TRAINING PRIMITIVES ===
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset



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


# === A3 TRAINING RUNNER ===
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydicom.pixel_data_handlers.util import apply_modality_lut
from torch.utils.data import DataLoader
from torchvision.models import resnet50


START = time.time()
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
TRAINING_BUDGET_S = 8.5 * 3600
IMG = 224
CROP_MM = 130.0
ANCHORS_PER_SERIES = 6
ENCODER_BATCH = 96
TRAIN_BATCH = 64
EPOCHS = 8
PATIENCE = 2
LR = 3e-4
WEIGHT_DECAY = 0.02
RAD_SHA256 = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"

def log(msg):
    print(f"[A3-RANK {time.time()-START:8.1f}s] {msg}", flush=True)

def sha256(path, chunk=8<<20):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(chunk),b""):
            h.update(b)
    return h.hexdigest()

def deadline(tag):
    if time.time()-START > TRAINING_BUDGET_S:
        raise TimeoutError(f"A3 training budget exceeded during {tag}")

def find_comp_root():
    for p in [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]:
        if (p/"train.csv").is_file() and (p/"train_series.csv").is_file():
            return p
    raise FileNotFoundError("competition root not found")

def normalize_name(s):
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def canonicalize_label_frame(df):
    by_norm={normalize_name(c):c for c in df.columns}
    uid=by_norm.get(normalize_name("StudyInstanceUID"))
    if uid is None:
        return None
    out=pd.DataFrame({"StudyInstanceUID":df[uid].astype(str)})
    for label in LABELS:
        c=by_norm.get(normalize_name(label))
        if c is None:
            return None
        out[label]=pd.to_numeric(df[c],errors="coerce")
    return out

def discover_pseudo_labels():
    roots=[
        Path("/kaggle/input/datasets/pilkwang/rsna-knee-llm-labels"),
        Path("/kaggle/input/rsna-knee-llm-labels"),
    ]
    candidates=[]
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*.csv"):
            try:
                raw=pd.read_csv(p)
                f=canonicalize_label_frame(raw)
                if f is None:
                    continue
                finite=np.isfinite(f[LABELS].to_numpy(float))
                coverage=int(finite.sum())
                uids=int(f.StudyInstanceUID.nunique())
                vals=f[LABELS].to_numpy(float)[finite]
                valid_range=bool(len(vals) and np.nanmin(vals)>=0 and np.nanmax(vals)<=1)
                if valid_range:
                    candidates.append((coverage,uids,p,f))
            except Exception as exc:
                log(f"pseudo candidate skipped {p}: {type(exc).__name__}: {exc}")
    if not candidates:
        raise RuntimeError("no pseudo-label CSV satisfies UID+12-target contract")
    candidates.sort(key=lambda x:(x[0],x[1]),reverse=True)
    coverage,uids,path,frame=candidates[0]
    if uids < 4000:
        raise RuntimeError(f"pseudo-label coverage suspicious: {uids} studies from {path}")
    log(f"pseudo labels: {path} | studies={uids} finite_cells={coverage}")
    return frame.drop_duplicates("StudyInstanceUID",keep="last"), str(path)

def prepare_targets(train, pseudo):
    gold=canonicalize_label_frame(train)
    if gold is None:
        raise RuntimeError("competition train.csv target contract changed")
    p=pseudo.set_index("StudyInstanceUID").reindex(gold.StudyInstanceUID).reset_index()
    if p[LABELS].isna().all(axis=None):
        raise RuntimeError("pseudo labels do not align to competition UIDs")
    weak=p[LABELS].to_numpy(np.float32)
    if np.isnan(weak).any():
        # Missing weak cells are truly unlabeled: encode as 0.5, which A3 masks.
        weak=np.where(np.isfinite(weak),weak,0.5).astype(np.float32)
    gold_values=gold[LABELS].to_numpy(np.float32)
    gold_mask=np.isfinite(gold_values)
    targets=np.where(gold_mask,gold_values,weak).astype(np.float32)
    if not np.isfinite(targets).all() or ((targets<0)|(targets>1)).any():
        raise RuntimeError("invalid target tensor")
    return (
        gold.StudyInstanceUID.astype(str).tolist(),
        targets,
        gold_mask,
        gold_values,
        weak,
    )

class RadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(self,x):
        return self.backbone(x).mean(dim=(2,3))

def find_rad_weight():
    candidates=[
        Path("/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt"),
        Path("/kaggle/input/resnet-50-radimagenet-marwan/ResNet50.pt"),
    ]
    existing=[p for p in candidates if p.is_file()]
    valid=[p for p in existing if sha256(p)==RAD_SHA256]
    if len(valid)!=1:
        raise RuntimeError(f"expected one pinned RadImageNet weight, got {valid}; existing={existing}")
    return valid[0]

def load_encoder(device):
    p=find_rad_weight()
    model=RadEncoder()
    state=torch.load(p,map_location="cpu",weights_only=True)
    expected=model.state_dict()
    if set(state)!=set(expected):
        raise RuntimeError("RadImageNet encoder key contract changed")
    bad=[k for k in state if tuple(state[k].shape)!=tuple(expected[k].shape)]
    if bad:
        raise RuntimeError(f"RadImageNet encoder shape drift: {bad[:10]}")
    model.load_state_dict(state,strict=True)
    model.eval().requires_grad_(False).to(device)
    log(f"RadImageNet encoder verified {p.name} sha={RAD_SHA256[:12]}")
    return model

def dicom_header(path):
    h=pydicom.dcmread(path,stop_before_pixels=True)
    ipp=getattr(h,"ImagePositionPatient",None)
    iop=getattr(h,"ImageOrientationPatient",None)
    inst=getattr(h,"InstanceNumber",None)
    return SliceRecord(
        sop_instance_uid=Path(path).stem,
        image_position=tuple(float(x) for x in ipp) if ipp is not None and len(ipp)==3 else None,
        image_orientation=tuple(float(x) for x in iop) if iop is not None and len(iop)==6 else None,
        instance_number=int(inst) if inst is not None else None,
    )

def read_slice(path):
    d=pydicom.dcmread(path)
    a=apply_modality_lut(d.pixel_array,d).astype(np.float32)
    if str(getattr(d,"PhotometricInterpretation",""))=="MONOCHROME1":
        a=a.max()-a
    if not np.isfinite(a).all():
        a=np.nan_to_num(a,nan=0.0,posinf=0.0,neginf=0.0)
    lo,hi=np.percentile(a,[1,99])
    if hi<=lo:
        lo=float(a.min()); hi=float(a.max())
    if hi<=lo:
        a=np.zeros_like(a,dtype=np.float32)
    else:
        a=np.clip((a-lo)/(hi-lo),0,1)
    ps=getattr(d,"PixelSpacing",None)
    spacing=float(ps[0]) if ps is not None and len(ps)>=1 and float(ps[0])>0 else 0.5
    side=max(8,min(int(round(CROP_MM/spacing)),min(a.shape)))
    y0=(a.shape[0]-side)//2; x0=(a.shape[1]-side)//2
    a=a[y0:y0+side,x0:x0+side]
    a=cv2.resize(a,(IMG,IMG),interpolation=cv2.INTER_AREA)
    return a.astype(np.float32)

@torch.inference_mode()
def encode_windows(model, triplets, device):
    out=[]
    for start in range(0,len(triplets),ENCODER_BATCH):
        deadline("feature encoding")
        chunk=triplets[start:start+ENCODER_BATCH]
        x=np.stack(chunk,axis=0).astype(np.float32)
        t=torch.from_numpy(x).to(device,non_blocking=True)
        t=t.mul_(2.0).sub_(1.0)
        with torch.autocast("cuda",dtype=torch.float16,enabled=device.type=="cuda"):
            f=model(t)
        f=f.float().cpu().numpy()
        if not np.isfinite(f).all():
            raise RuntimeError("non-finite RadImageNet feature")
        out.append(f.astype(np.float16))
    return np.concatenate(out,axis=0)

def build_feature_bank(comp, series_df, uids, targets, gold_mask):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda":
        raise RuntimeError("A3 feature-bank generation requires GPU")
    encoder=load_encoder(device)
    series_by_uid={uid:g.to_dict("records") for uid,g in series_df.groupby("StudyInstanceUID",sort=False)}
    records=[]
    total_windows=0
    geometry_fallback_series=0
    skipped_series=0

    for idx,uid in enumerate(uids):
        deadline("feature-bank study loop")
        token_features=[]
        token_types=[]
        rows=series_by_uid.get(uid,[])
        for row in rows:
            suid=str(row["SeriesInstanceUID"])
            sdir=comp/"train_series"/uid/suid
            files=sorted(sdir.glob("*.dcm"))
            if not files:
                skipped_series+=1
                continue
            headers=[]
            by_sop={}
            for f in files:
                try:
                    rec=dicom_header(f)
                    headers.append(rec); by_sop[rec.sop_instance_uid]=f
                except Exception:
                    continue
            if len(headers)<2:
                skipped_series+=1
                continue
            if not all(r.image_position is not None and r.image_orientation is not None for r in headers):
                geometry_fallback_series+=1
            try:
                ordered=sort_slice_records(headers)
            except Exception:
                skipped_series+=1
                continue
            sops=[r.sop_instance_uid for r in ordered]
            manifest=build_series_window_manifest(sops,n_anchors=ANCHORS_PER_SERIES)
            plane=str(row.get("Anatomical_Plane",""))
            fluid=row.get("Fluid_Sensitive",0)
            fat=row.get("Fat_Suppression",0)
            type_id=acquisition_type_id(plane,fluid,fat)
            triplets=[]
            for m in manifest:
                try:
                    planes=[read_slice(by_sop[s]) for s in m["sop_triplet"]]
                    triplets.append(np.stack(planes,axis=0))
                except Exception:
                    continue
            if not triplets:
                skipped_series+=1
                continue
            feats=encode_windows(encoder,triplets,device)
            token_features.append(feats)
            token_types.extend([type_id]*len(feats))
            total_windows+=len(feats)

        if not token_features:
            raise RuntimeError(f"study {uid} produced zero valid tokens")
        feat=np.concatenate(token_features,axis=0)
        records.append({
            "uid":uid,
            "features":feat,
            "token_types":np.asarray(token_types,dtype=np.int16),
            "targets":targets[idx].astype(np.float32),
            "gold_mask":gold_mask[idx].astype(bool),
            "fold":uid_fold(uid,5),
        })
        if (idx+1)%100==0:
            log(f"feature bank {idx+1}/{len(uids)} studies | windows={total_windows}")

    bank_path=WORK/"a3_feature_bank.pt"
    torch.save(records,bank_path)
    receipt={
        "studies":len(records),
        "windows":total_windows,
        "anchors_per_series":ANCHORS_PER_SERIES,
        "img":IMG,
        "crop_mm":CROP_MM,
        "feature_dim":2048,
        "dtype":"float16",
        "geometry_fallback_series":geometry_fallback_series,
        "skipped_series":skipped_series,
        "rad_sha256":RAD_SHA256,
        "bank_sha256":sha256(bank_path),
    }
    (WORK/"a3_feature_bank_receipt.json").write_text(json.dumps(receipt,indent=2))
    log(f"feature bank complete: {receipt}")
    del encoder
    gc.collect(); torch.cuda.empty_cache()
    return records,receipt

def high_conf_binary_targets(targets):
    y=np.full_like(targets,np.nan,dtype=np.float32)
    y[targets<=0.10]=0
    y[targets>=0.90]=1
    return y

def macro_auc_missing(y_true,y_score):
    per=[]
    for q in range(y_true.shape[1]):
        mask=np.isfinite(y_true[:,q])
        if mask.sum()<2 or len(np.unique(y_true[mask,q]))<2:
            per.append(np.nan)
        else:
            _,one=macro_auc(y_true[mask,q:q+1],y_score[mask,q:q+1])
            per.append(one[0])
    per=np.asarray(per,float)
    return (float(np.nanmean(per)) if np.isfinite(per).any() else float("nan"),per)

def train_one_fold(records,fold,variant_name,device):
    variant=VARIANTS[variant_name]
    train_records=[r for r in records if int(r["fold"])!=fold]
    val_records=[r for r in records if int(r["fold"])==fold]
    train_loader=DataLoader(
        FeatureBankDataset(train_records),batch_size=TRAIN_BATCH,shuffle=True,
        num_workers=0,collate_fn=collate_feature_bank,drop_last=False,
        generator=torch.Generator().manual_seed(SEED+fold),
    )
    val_loader=DataLoader(
        FeatureBankDataset(val_records),batch_size=TRAIN_BATCH,shuffle=False,
        num_workers=0,collate_fn=collate_feature_bank,
    )
    model=build_model(
        2048,variant,projection_dim=384,metadata_dim=48,hidden_dim=256,
        gru_layers=2,metadata_vocab=32,dropout=0.15,
    ).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best_score=-1.0
    best_state=None
    bad_epochs=0
    history=[]

    for epoch in range(EPOCHS):
        deadline(f"{variant_name} fold {fold} epoch {epoch}")
        model.train()
        losses=[]
        for batch in train_loader:
            feat=batch["features"].to(device,non_blocking=True)
            typ=batch["token_types"].to(device,non_blocking=True)
            mask=batch["token_mask"].to(device,non_blocking=True)
            target=batch["targets"].to(device,non_blocking=True)
            gold=batch["gold_mask"].to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=device.type=="cuda"):
                logits=model(feat,typ,mask)
                loss,_=compute_objective(logits,target,gold,variant)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        pred=[]; val_target=[]
        with torch.inference_mode():
            for batch in val_loader:
                feat=batch["features"].to(device,non_blocking=True)
                typ=batch["token_types"].to(device,non_blocking=True)
                mask=batch["token_mask"].to(device,non_blocking=True)
                with torch.autocast("cuda",dtype=torch.float16,enabled=device.type=="cuda"):
                    z=model(feat,typ,mask)
                pred.append(z.float().cpu().numpy())
                val_target.append(batch["targets"].numpy())
        pred=np.concatenate(pred); val_target=np.concatenate(val_target)
        binary=high_conf_binary_targets(val_target)
        score,_=macro_auc_missing(binary,pred)
        history.append({"epoch":epoch,"loss":float(np.mean(losses)),"weak_extreme_auc":score})
        log(f"{variant_name} fold={fold} epoch={epoch} loss={np.mean(losses):.5f} weakAUC={score:.5f}")
        selection=score if np.isfinite(score) else -1
        if selection>best_score+1e-5:
            best_score=selection
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            bad_epochs=0
        else:
            bad_epochs+=1
            if bad_epochs>=PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(f"{variant_name} fold {fold}: no checkpoint")
    model.load_state_dict(best_state,strict=True)
    ckpt=WORK/f"a3_{variant_name.lower()}_fold{fold}.pt"
    torch.save({
        "variant":variant_name,"fold":fold,"state_dict":best_state,
        "config":model.cfg.__dict__,"history":history,"seed":SEED,
    },ckpt)

    model.eval(); pred=[]; ids=[]
    with torch.inference_mode():
        for batch in val_loader:
            feat=batch["features"].to(device); typ=batch["token_types"].to(device); mask=batch["token_mask"].to(device)
            with torch.autocast("cuda",dtype=torch.float16,enabled=device.type=="cuda"):
                z=model(feat,typ,mask)
            pred.append(z.float().cpu().numpy()); ids.extend(batch["uids"])
    del model,opt,scaler
    gc.collect(); torch.cuda.empty_cache()
    return ids,np.concatenate(pred),str(ckpt),history

def evaluate_variant(records,uids,targets,gold_values,variant_name):
    device=torch.device("cuda:0")
    pred_by_uid={}
    checkpoints=[]
    histories={}
    for fold in range(5):
        ids,pred,ckpt,hist=train_one_fold(records,fold,variant_name,device)
        pred_by_uid.update({u:p for u,p in zip(ids,pred)})
        checkpoints.append({"path":ckpt,"sha256":sha256(ckpt)})
        histories[str(fold)]=hist
    if set(pred_by_uid)!=set(uids):
        raise RuntimeError(f"{variant_name}: incomplete OOF UID coverage")
    oof=np.stack([pred_by_uid[u] for u in uids])
    if not np.isfinite(oof).all():
        raise RuntimeError(f"{variant_name}: non-finite OOF")

    weak_binary=high_conf_binary_targets(targets)
    weak_macro,weak_per=macro_auc_missing(weak_binary,oof)
    gold_macro,gold_per=macro_auc_missing(gold_values,oof)
    report={
        "variant":variant_name,
        "weak_extreme_macro_auc":weak_macro,
        "weak_extreme_per_target":dict(zip(LABELS,[None if not np.isfinite(x) else float(x) for x in weak_per])),
        "gold58_macro_auc":gold_macro,
        "gold58_per_target":dict(zip(LABELS,[None if not np.isfinite(x) else float(x) for x in gold_per])),
        "checkpoints":checkpoints,
        "histories":histories,
    }
    pd.DataFrame(oof,columns=LABELS).assign(StudyInstanceUID=uids).to_csv(WORK/f"a3_{variant_name.lower()}_oof.csv",index=False)
    (WORK/f"a3_{variant_name.lower()}_report.json").write_text(json.dumps(report,indent=2))
    log(f"{variant_name} COMPLETE weak={weak_macro:.6f} gold58={gold_macro:.6f}")
    return oof,report

def rank01(x):
    return pd.DataFrame(x).rank(method="average",pct=True).to_numpy(float)

def complementarity(reports_preds):
    names=list(reports_preds)
    out={}
    for i,a in enumerate(names):
        for b in names[i+1:]:
            ra=rank01(reports_preds[a]); rb=rank01(reports_preds[b])
            corr=[]
            for q in range(ra.shape[1]):
                corr.append(float(np.corrcoef(ra[:,q],rb[:,q])[0,1]))
            out[f"{a}__{b}"]={"mean_target_rank_corr":float(np.mean(corr)),"per_target":dict(zip(LABELS,corr))}
    return out

def main():
    seed_everything(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    log(f"GPU(s): {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    comp=find_comp_root()
    train=pd.read_csv(comp/"train.csv",dtype={"StudyInstanceUID":str})
    series=pd.read_csv(comp/"train_series.csv",dtype={"StudyInstanceUID":str,"SeriesInstanceUID":str})
    pseudo,pseudo_path=discover_pseudo_labels()
    uids,targets,gold_mask,gold_values,weak=prepare_targets(train,pseudo)
    if len(uids)!=4407:
        log(f"WARNING competition train study count is {len(uids)}, expected historical 4407")
    records,bank_receipt=build_feature_bank(comp,series,uids,targets,gold_mask)

    preds={}; reports={}
    for variant in ("A3-1","A3-2","A3-3"):
        oof,rep=evaluate_variant(records,uids,targets,gold_values,variant)
        preds[variant]=oof; reports[variant]=rep

    comparison={
        "experiment":"EXP-A3 A3-RANK",
        "seed":SEED,
        "pseudo_label_path":pseudo_path,
        "feature_bank":bank_receipt,
        "variants":reports,
        "complementarity_internal":complementarity(preds),
        "elapsed_seconds":time.time()-START,
    }
    (WORK/"a3_rank_comparison.json").write_text(json.dumps(comparison,indent=2))
    log("A3-RANK TRAINING COMPLETE")
    print(json.dumps({
        k:{"weak":v["weak_extreme_macro_auc"],"gold58":v["gold58_macro_auc"]}
        for k,v in reports.items()
    },indent=2))

if __name__=="__main__":
    main()

