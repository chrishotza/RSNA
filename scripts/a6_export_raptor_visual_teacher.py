"""A6-R1 image-only teacher export from original RSNA DICOM; no Kaggle submission.

Run inside Kaggle with competition train.csv / train_series.csv / train_series,
the frozen Raptor V5 checkpoint, and the reviewed rsna_knee source on sys.path.
This is IN-SAMPLE teacher supervision, never an independent evaluation set.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
UID = "StudyInstanceUID"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def training_ids(frame):
    if frame[UID].isna().any() or frame[UID].duplicated().any():
        raise ValueError("Train CSV has null/duplicate UIDs")
    labeled = frame[LABELS].notna().all(axis=1)
    if int(labeled.sum()) != 58 or int((~labeled).sum()) != 4349:
        raise ValueError(f"Unexpected gold/non-gold counts: {int(labeled.sum())}/{int((~labeled).sum())}")
    return frame.loc[~labeled, UID].astype(str).tolist(), frame.loc[labeled, UID].astype(str).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--competition-root", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--source-root", required=True, help="Path containing rsna_knee/ package (usually src)")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0, help="For smoke tests; 0=all 4349")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.source_root).resolve()))
    from rsna_knee.raptor import (
        MAXSPAN_CHECKPOINT, MAXSPAN_K_EVAL, RAPTOR_LABELS,
        devices_for_inference, load_raptor, predict_studies, series_by_study,
    )
    if LABELS != list(RAPTOR_LABELS):
        raise ValueError("Raptor target order mismatch")
    root = Path(a.competition_root)
    weights = Path(a.weights)
    if weights.name != MAXSPAN_CHECKPOINT:
        raise ValueError(f"Expected exact frozen checkpoint filename {MAXSPAN_CHECKPOINT}")
    frame = pd.read_csv(root / "train.csv", dtype={UID: str})
    ids, gold = training_ids(frame)
    if a.limit:
        if a.limit < 0:
            raise ValueError("limit must be positive or zero")
        ids = ids[:a.limit]
    if set(ids) & set(gold):
        raise ValueError("Gold leakage")
    series = series_by_study(pd.read_csv(root / "train_series.csv", dtype={UID: str, "SeriesInstanceUID": str}))
    missing = set(ids) - set(series)
    if missing:
        raise ValueError(f"Missing series metadata for {len(missing)} studies")
    # Reject studies without a usable metadata match before expensive inference.
    from rsna_knee.raptor import pick_series_for_slot, MAXSPAN_SLOTS
    unslotted = [study for study in ids if not any(
        pick_series_for_slot(series[study], plane, fluid, set()) is not None
        for plane, fluid, _ in MAXSPAN_SLOTS
    )]
    if unslotted:
        raise ValueError(f"{len(unslotted)} studies have no selectable MR series: {unslotted[:5]}")
    devices = devices_for_inference()
    if not devices or devices == ["cpu"]:
        raise RuntimeError("GPU required for full Raptor extraction")
    replicas = []
    resolution = None
    for device in devices:
        model, res = load_raptor(str(weights), device)
        if resolution is not None and res != resolution:
            raise ValueError("Resolution mismatch between model replicas")
        resolution = res
        replicas.append((model, device))
    scores, failures = predict_studies(
        ids, series, str(root / "train_series"),
        replicas, resolution, k_eval=MAXSPAN_K_EVAL,
        reverse=False, workers=a.workers, log_every=25,
    )
    # Upstream implementation returns 0.5 on decoding errors: reject ALL such rows.
    if failures:
        raise RuntimeError(f"{len(failures)} teacher fallbacks detected; no pseudo-label artifact accepted: {failures[:8]}")
    if scores.shape != (len(ids), 12) or not np.isfinite(scores).all():
        raise ValueError(f"Bad teacher shape / finite values: {scores.shape}")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Teacher predictions outside [0,1]")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(scores, columns=LABELS)
    df.insert(0, UID, ids)
    df.to_csv(out / "a6_raptor_v5_visual.csv", index=False)
    manifest = {
        "teacher": MAXSPAN_CHECKPOINT,
        "sha256": sha256(weights),
        "studies": len(ids),
        "gold_excluded": len(gold),
        "target_order": LABELS,
        "k_eval": MAXSPAN_K_EVAL,
        "resolution": resolution,
        "devices": devices,
        "in_sample_teacher": True,
        "no_fallbacks": True,
        "gold_not_used_as_label": True,
        "note": "No OOF/independent validation claims; recheck teacher provenance before fitting student.",
    }
    (out / "a6_raptor_v5_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
