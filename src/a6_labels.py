from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

LABELS=[
    "ACL","MCL","Medial Meniscus","Lateral Meniscus",
    "Medial OA","Lateral OA","PF OA","Effusion",
    "Synovitis","Baker's","Contusion","Fracture",
]

@dataclass(frozen=True)
class TeacherPlan:
    source_by_target: dict[str,str]
    source_files: dict[str,str]

    def validate(self):
        if set(self.source_by_target)!=set(LABELS):
            missing=sorted(set(LABELS)-set(self.source_by_target))
            extra=sorted(set(self.source_by_target)-set(LABELS))
            raise ValueError(f"teacher target mismatch missing={missing} extra={extra}")
        unknown=sorted(set(self.source_by_target.values())-set(self.source_files))
        if unknown:
            raise ValueError(f"teacher source missing file mapping: {unknown}")


def _norm(s): return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def canonical_label_frame(df:pd.DataFrame)->pd.DataFrame:
    by={_norm(c):c for c in df.columns}
    uid=by.get(_norm("StudyInstanceUID"))
    if uid is None:
        raise ValueError("missing StudyInstanceUID")
    out=pd.DataFrame({"StudyInstanceUID":df[uid].astype(str)})
    for lab in LABELS:
        exact=by.get(_norm(lab))
        candidates=[]
        for c in df.columns:
            nc=_norm(c)
            if _norm(lab) in nc and ("confidence" in nc or "probability" in nc or "pseudo" in nc):
                candidates.append(c)
        c=exact or (candidates[0] if candidates else None)
        if c is None:
            raise ValueError(f"missing target {lab}")
        out[lab]=pd.to_numeric(df[c],errors="coerce")
    return out


def load_table(path:str|Path)->pd.DataFrame:
    p=Path(path)
    raw=pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p)
    return canonical_label_frame(raw).drop_duplicates("StudyInstanceUID",keep="last")


def build_teacher_targets(
    uids:list[str],
    *,
    gold_uids:set[str],
    plan:TeacherPlan,
)->tuple[np.ndarray,dict]:
    """Construct A6 soft targets from fixed per-target public teachers.

    Any gold UID in the requested training set is a hard error.
    """
    plan.validate()
    if set(map(str,uids)) & set(map(str,gold_uids)):
        raise RuntimeError("gold UID requested from A6 teacher plan")

    tables={name:load_table(path).set_index("StudyInstanceUID") for name,path in plan.source_files.items()}
    out=np.full((len(uids),len(LABELS)),.5,dtype=np.float32)
    missing={}
    for j,lab in enumerate(LABELS):
        source=plan.source_by_target[lab]
        s=tables[source][lab].reindex(list(map(str,uids)))
        vals=pd.to_numeric(s,errors="coerce").to_numpy(float)
        good=np.isfinite(vals)&(vals>=0)&(vals<=1)
        out[good,j]=vals[good].astype(np.float32)
        missing[lab]=int((~good).sum())
    receipt={
        "n_uids":len(uids),
        "source_by_target":dict(plan.source_by_target),
        "missing_cells_by_target":missing,
        "neutral_fill":0.5,
    }
    return out,receipt


def fixed_default_plan(source_files:dict[str,str])->TeacherPlan:
    """Conservative pre-audit default.

    Until the read-only source audit completes, use the best published hybrid
    label table globally. A target-specific plan is frozen only by an explicit
    registry update after the audit; it is never learned inside training.
    """
    if "hybrid" not in source_files:
        raise ValueError("hybrid source required")
    return TeacherPlan(
        source_by_target={lab:"hybrid" for lab in LABELS},
        source_files=source_files,
    )
