"""Kaggle entrypoint for RSNA A6 Raptor pseudo-label extraction.

Requires attached competition RSNA knee dataset, public datasets
dreaddevelopment/raptor-knee-maxspan and mathischmp/rsna-knee-src, and
this RSNA code as an attached dataset. No model training or submission.
"""
from pathlib import Path
import subprocess
import sys

def find(root, name):
    matches = sorted(Path(root).rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name} under {root}, got {len(matches)}")
    return matches[0]

root=Path("/kaggle/input")
if not root.exists():
    raise RuntimeError("This script is designed for Kaggle with attached inputs")
competition=find(root,"train.csv").parent
if not (competition/"train_series.csv").exists() or not (competition/"train_series").exists():
    raise RuntimeError(f"Detected {competition} does not contain RSNA train_series DICOM")
checkpoint=find(root,"raptor_ft_coatnet_v5_full_swa.pt")
package=find(root,"raptor.py").parent.parent
if not (package/"rsna_knee"/"raptor.py").exists():
    raise RuntimeError("Expected source root containing rsna_knee/raptor.py")
exporter=find(root,"a6_export_raptor_visual_teacher.py")
limit=int(__import__("os").environ.get("A6_SMOKE_LIMIT","3"))
if limit < 0:
    raise ValueError("A6_SMOKE_LIMIT must be >= 0")
print(f"RSNA dataset: {competition}")
print(f"Checkpoint: {checkpoint}")
print(f"Model source: {package}")
print(f"A6 script: {exporter}")
print(f"Study limit: {limit} (0 means all non-gold)")
cmd=[sys.executable,str(exporter),"--competition-root",str(competition),
     "--weights",str(checkpoint),"--source-root",str(package),
     "--output","/kaggle/working/a6_raptor_visual",
     "--limit",str(limit)]
subprocess.run(cmd,check=True)
