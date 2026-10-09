from __future__ import annotations

import json, hashlib, time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.a4_ortho import OrthoRouterConfig, ortho_route, validate_submission_frame

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

START = time.time()
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[A4-ORTHO {time.time()-START:8.1f}s] {msg}", flush=True)


def find_comp_root() -> Path:
    for p in [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]:
        if (p/"test.csv").is_file() and (p/"sample_submission.csv").is_file():
            return p
    raise FileNotFoundError("competition root not found")


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def discover_asset(patterns: list[str]) -> Path:
    roots = [Path("/kaggle/input")]
    hits = []
    pats = [p.lower() for p in patterns]
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            low = p.name.lower()
            if any(x in low for x in pats):
                hits.append(p)
    if not hits:
        raise FileNotFoundError(f"no asset matched {patterns}")
    hits = sorted(set(hits))
    log("asset candidates: " + ", ".join(map(str, hits[:20])))
    return hits[0]


def load_anchor_predictions(comp: Path, test_uids: list[str]) -> np.ndarray:
    """
    A4 intentionally keeps the anchor interface separate from specialist loaders.
    First runnable version may use a reproducible public parent prediction file
    if present; otherwise a later contract extraction will wire the exact public
    anchor runtime.
    """
    candidates = []
    for p in Path("/kaggle/input").rglob("*.csv"):
        try:
            head = pd.read_csv(p, nrows=2)
        except Exception:
            continue
        if head.columns.tolist() == ["StudyInstanceUID", *LABELS]:
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError("no public anchor prediction CSV with exact submission schema found")
    # deterministic choice: largest row count, then lexical path
    scored = []
    for p in candidates:
        try:
            n = len(pd.read_csv(p, usecols=["StudyInstanceUID"]))
            scored.append((n, str(p), p))
        except Exception:
            pass
    if not scored:
        raise RuntimeError("anchor CSV candidates unreadable")
    scored.sort(key=lambda x: (-x[0], x[1]))
    p = scored[0][2]
    frame = pd.read_csv(p, dtype={"StudyInstanceUID": str})
    frame = frame.set_index("StudyInstanceUID").reindex(test_uids)
    if frame[LABELS].isna().any().any():
        raise RuntimeError(f"anchor CSV does not cover all test UIDs: {p}")
    log(f"anchor predictions={p}")
    return frame[LABELS].to_numpy(np.float64)


def infer_dinov3_specialist(comp: Path, test_uids: list[str]) -> np.ndarray:
    """
    Placeholder contract gate.

    This function deliberately fails until the exact public checkpoint architecture
    and preprocessing contract are extracted. A4 must never silently guess a loader.
    """
    ckpt = discover_asset(["best_dinov3_finetuned_384"])
    raise RuntimeError(
        "DINOv3 specialist checkpoint discovered but exact public loader contract "
        f"is not yet frozen: {ckpt}"
    )


def infer_second_orthogonal_specialist(comp: Path, test_uids: list[str]) -> np.ndarray:
    """
    Placeholder for BiomedCLIP / second structurally distinct public arm.
    """
    raise RuntimeError("second orthogonal specialist contract not frozen yet")


def main() -> None:
    comp = find_comp_root()
    test = pd.read_csv(comp/"test.csv", dtype={"StudyInstanceUID": str})
    uids = test.StudyInstanceUID.astype(str).tolist()

    anchor = load_anchor_predictions(comp, uids)
    specialist_b = infer_dinov3_specialist(comp, uids)
    specialist_c = infer_second_orthogonal_specialist(comp, uids)

    cfg = OrthoRouterConfig()
    routed, audit = ortho_route(anchor, specialist_b, specialist_c, cfg)

    sub = pd.DataFrame(routed, columns=LABELS)
    sub.insert(0, "StudyInstanceUID", uids)
    validate_submission_frame(sub, LABELS, uids)

    out = WORK/"submission.csv"
    sub.to_csv(out, index=False)

    receipt = {
        "experiment": "EXP-A4",
        "codename": "A4-ORTHO",
        "inference_only": True,
        "router": asdict(cfg),
        "studies": len(uids),
        "submission_sha256": sha256(out),
        "elapsed_seconds": time.time()-START,
        "gate_mean": float(np.mean(audit["gate"])),
        "gate_max": float(np.max(audit["gate"])),
    }
    (WORK/"a4_ortho_receipt.json").write_text(json.dumps(receipt, indent=2))
    log(f"COMPLETE {out} sha={receipt['submission_sha256']}")


if __name__ == "__main__":
    main()
