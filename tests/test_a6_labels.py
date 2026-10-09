import numpy as np
import pandas as pd
import pytest

from src.a6_labels import LABELS,TeacherPlan,build_teacher_targets,fixed_default_plan

def _write(path,offset):
    rows=[]
    for i in range(5):
        r={"StudyInstanceUID":f"u{i}"}
        for j,l in enumerate(LABELS):
            r[l]=float(np.clip(.1+.15*((i+j+offset)%6),0,1))
        rows.append(r)
    pd.DataFrame(rows).to_csv(path,index=False)

def test_teacher_plan_can_route_by_target(tmp_path):
    a=tmp_path/"a.csv"; b=tmp_path/"b.csv"
    _write(a,0); _write(b,1)
    mapping={l:"a" for l in LABELS}
    mapping["Synovitis"]="b"
    plan=TeacherPlan(mapping,{"a":str(a),"b":str(b)})
    y,receipt=build_teacher_targets(["u0","u2"],gold_uids={"g"},plan=plan)
    assert y.shape==(2,12)
    assert receipt["source_by_target"]["Synovitis"]=="b"

def test_gold_uid_is_refused(tmp_path):
    p=tmp_path/"h.csv"; _write(p,0)
    plan=fixed_default_plan({"hybrid":str(p)})
    with pytest.raises(RuntimeError):
        build_teacher_targets(["u1"],gold_uids={"u1"},plan=plan)

def test_missing_value_becomes_neutral(tmp_path):
    p=tmp_path/"h.csv"; _write(p,0)
    d=pd.read_csv(p); d.loc[d.StudyInstanceUID=="u2","PF OA"]=np.nan; d.to_csv(p,index=False)
    plan=fixed_default_plan({"hybrid":str(p)})
    y,r=build_teacher_targets(["u2"],gold_uids=set(),plan=plan)
    assert y[0,LABELS.index("PF OA")]==.5
    assert r["missing_cells_by_target"]["PF OA"]==1
