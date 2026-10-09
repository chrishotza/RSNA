"""A6 R1 teacher target ablation. Never consumes gold58 for label selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

UID = "StudyInstanceUID"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
           "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]


def load_table(path):
    df = pd.read_csv(path, dtype={UID: str})
    if UID not in df or df[UID].isna().any() or df[UID].duplicated().any():
        raise ValueError(f"Missing/duplicate UID in {path}")
    missing = set(TARGETS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing target columns in {path}: {sorted(missing)}")
    return df.set_index(UID)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--visual", required=True)
    p.add_argument("--gold-uids", required=True, help="one StudyInstanceUID per line; authoritative gold exclusion")
    p.add_argument("--out", required=True)
    p.add_argument("--expected", type=int, default=4349)
    a = p.parse_args()
    text = load_table(a.text)
    visual = load_table(a.visual)
    gold = set(Path(a.gold_uids).read_text().split())
    if len(gold) != 58:
        raise ValueError(f"Expected 58 gold UIDs, received {len(gold)}")
    if not text.index.equals(visual.index):
        if set(text.index) != set(visual.index):
            raise ValueError("Teacher study UID sets differ")
        visual = visual.reindex(text.index)
    if gold.intersection(text.index):
        raise ValueError("Gold58 leakage detected in teacher inputs")
    if len(text) != a.expected:
        raise ValueError(f"Expected {a.expected} non-gold studies, got {len(text)}")
    ta = text[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    va = visual[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(va).all() or (va < 0).any() or (va > 1).any():
        raise ValueError("Visual teacher must supply finite [0,1] probabilities")
    if np.isinf(ta).any() or not np.isfinite(ta).any():
        raise ValueError("Invalid text probabilities")
    if np.isfinite(ta).any() and ((ta[np.isfinite(ta)] < 0).any() or (ta[np.isfinite(ta)] > 1).any()):
        raise ValueError("Text probabilities outside [0,1]")
    confcols = [f"{x}__conf" for x in TARGETS]
    if all(x in text for x in confcols):
        conf = text[confcols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        conf = np.nan_to_num(conf, nan=0, posinf=0, neginf=0).clip(0, 1)
    else:
        conf = np.isfinite(ta).astype(float)
    present = np.isfinite(ta)
    ta = np.where(present, ta, va)
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    variants = {"text": 0.0, "visual": 1.0, "mixed": 0.5}
    for name, alpha in variants.items():
        probs = ((1-alpha)*ta + alpha*va).clip(0,1)
        if name == "text":
            probs = np.where(present, probs, np.nan)
        df = pd.DataFrame(probs, columns=TARGETS)
        df.insert(0, UID, text.index)
        # Per-cell weights are explicit; never silently train absent text as clean negative.
        weights = (conf if name == "text" else
                   np.ones_like(conf) if name == "visual" else
                   np.where(present, 0.5 * conf + 0.5, 1.0))
        for j, target in enumerate(TARGETS):
            df[f"{target}__conf"] = weights[:, j]
        df.to_csv(root/f"a6_{name}_targets.csv", index=False)
    (root/"manifest.json").write_text(json.dumps({
        "variants": variants, "studies": len(text), "gold_excluded": len(gold),
        "target_order": TARGETS, "note": "Teacher training provenance and leak-safe validation must be audited separately."
    }, indent=2))
    print(f"Generated three training target files for {len(text)} non-gold studies.")


if __name__ == "__main__":
    main()
