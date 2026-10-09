"""A6-R1 synthetic regression checks; never uses patient data."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "a6_teacher_target_ablation.py"
UID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def test_generation_and_guards():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        ids = ["s1", "s2", "s3"]
        textual = pd.DataFrame({UID: ids, **{t: [0.2, 0.8, 0.4] for t in TARGETS}})
        visual = pd.DataFrame({UID: ids[::-1], **{t: [0.6, 0.1, 0.7] for t in TARGETS}})
        textual.loc[0, "ACL"] = np.nan
        for target in TARGETS:
            textual[f"{target}__conf"] = 0.6
        textual.to_csv(p / "text.csv", index=False)
        visual.to_csv(p / "visual.csv", index=False)
        gold = p / "gold.txt"
        gold.write_text("\n".join(f"gold{i}" for i in range(58)))
        args = [
            sys.executable, str(SCRIPT),
            "--text", str(p / "text.csv"),
            "--visual", str(p / "visual.csv"),
            "--gold-uids", str(gold), "--out", str(p / "out"), "--expected", "3",
        ]

        result = subprocess.run(args, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        mixed = pd.read_csv(p / "out" / "a6_mixed_targets.csv")
        text_only = pd.read_csv(p / "out" / "a6_text_targets.csv")
        visual_only = pd.read_csv(p / "out" / "a6_visual_targets.csv")
        assert np.isclose(mixed.loc[0, "ACL"], 0.7)
        assert np.isclose(mixed.loc[0, "ACL__conf"], 1.0)
        assert np.isnan(text_only.loc[0, "ACL"])
        assert np.isclose(visual_only.loc[0, "ACL"], 0.7)
        assert list(mixed[UID]) == ids

        gold.write_text("s1\n" + "\n".join(f"gold{i}" for i in range(57)))
        blocked = subprocess.run(args, capture_output=True, text=True)
        assert blocked.returncode != 0 and "Gold58 leakage" in blocked.stderr

        gold.write_text("\n".join(f"gold{i}" for i in range(58)))
        textual.loc[:, TARGETS] = np.nan
        textual.to_csv(p / "text.csv", index=False)
        blocked = subprocess.run(args, capture_output=True, text=True)
        assert blocked.returncode != 0 and "Invalid text probabilities" in blocked.stderr
