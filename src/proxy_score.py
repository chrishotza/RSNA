"""Leak-free macro ROC-AUC proxy and Kaggle-LB calibration helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    """Return exact competition-style macro ROC-AUC and per-label AUCs."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)

    if y.ndim != 2 or p.shape != y.shape or y.shape[1] != len(LABELS):
        raise ValueError("expected (n_studies, 12) arrays with identical shape")

    if np.isinf(y).any():
        raise ValueError("targets contain infinity; use NaN only for unaddressed labels")

    aucs = np.full(len(LABELS), np.nan, dtype=float)

    for j, label in enumerate(LABELS):
        yy = y[:, j]
        pp = p[:, j]
        mask = np.isfinite(yy)
        if not np.isfinite(pp[mask]).all():
            raise ValueError("prediction coverage is incomplete for labeled targets")
        yy, pp = yy[mask], pp[mask]

        if np.unique(yy).size < 2:
            continue

        aucs[j] = roc_auc_score(yy, pp)

    if not np.isfinite(aucs).all():
        missing = [LABELS[i] for i, a in enumerate(aucs) if not np.isfinite(a)]
        raise ValueError(f"labels without valid binary variation: {missing}")

    return float(aucs.mean()), aucs


def validate_oof(
    study_ids: Iterable[str],
    fold_ids: Iterable[int],
) -> None:
    """Fail loudly if folds are malformed."""
    ids = pd.Series(list(study_ids), dtype="string")
    folds = pd.Series(list(fold_ids), dtype="Int64")

    if len(ids) != len(folds):
        raise ValueError("study_ids and fold_ids have different lengths")
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("study IDs must be non-null and unique")
    if folds.isna().any() or (folds < 0).any():
        raise ValueError("fold IDs must be non-negative integers")


def score_oof_file(
    path: str | Path,
    target_prefix: str = "y_",
    pred_prefix: str = "p_",
) -> dict:
    """Score OOF predictions with one unique row per study and fixed fold provenance."""
    df = pd.read_csv(path)

    provenance = ["StudyInstanceUID", "fold"]
    missing_provenance = [column for column in provenance if column not in df.columns]
    if missing_provenance:
        raise ValueError(f"missing OOF provenance columns: {missing_provenance}")
    validate_oof(df["StudyInstanceUID"], df["fold"])

    true_cols = [target_prefix + label for label in LABELS]
    pred_cols = [pred_prefix + label for label in LABELS]

    missing = [c for c in true_cols + pred_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    macro, per_label = macro_auc(
        df[true_cols].to_numpy(),
        df[pred_cols].to_numpy(),
    )

    return {
        "macro_auc": macro,
        "per_label_auc": dict(zip(LABELS, per_label)),
        "n_studies": int(len(df)),
    }


def append_submission_record(
    ledger_path: str | Path,
    row: dict,
) -> None:
    """Append a manually verified Kaggle result to the experiment ledger."""
    path = Path(ledger_path)
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()

    add = pd.DataFrame([row])
    out = pd.concat([frame, add], ignore_index=True)

    if "submission_id" in out.columns:
        dup = out["submission_id"].dropna().astype(str)
        if dup.duplicated().any():
            raise ValueError("duplicate Kaggle submission_id")

    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="RSNA macro-AUC proxy scorer")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score a single OOF CSV")
    score.add_argument("--oof", required=True)

    args = parser.parse_args()

    if args.command == "score":
        result = score_oof_file(args.oof)
        print(f"studies: {result['n_studies']}")
        print(f"macro_auc: {result['macro_auc']:.8f}")
        print("per_label_auc:")
        for label, auc in result["per_label_auc"].items():
            print(f"  {label}: {auc:.8f}")


if __name__ == "__main__":
    main()
