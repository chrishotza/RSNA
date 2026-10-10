"""Build an A6 training variant from audited A3 RadImageNet reader.
Hard gold exclusion from optimization; gold used for final reporting only.
"""
from pathlib import Path
import sys
root=Path(sys.argv[1])
src=(root/"notebooks/experiments/EXP-A3-rank-train.py").read_text()
start=src.index("def discover_pseudo_labels():")
end=src.index("\ndef prepare_targets(",start)
replace='''def discover_pseudo_labels():
    root=Path("/kaggle/input")
    matches=sorted(root.rglob("a6_raptor_v5_visual.csv"))
    if len(matches)!=1:
        raise RuntimeError(f"Expected exactly one Raptor V5 teacher CSV, got {matches}")
    path=matches[0]
    raw=pd.read_csv(path,dtype={"StudyInstanceUID":str})
    pseudo=canonicalize_label_frame(raw)
    if pseudo is None or pseudo.StudyInstanceUID.duplicated().any():
        raise RuntimeError("Invalid/duplicate teacher UID")
    vals=pseudo[LABELS].to_numpy(float)
    if len(pseudo)!=4349 or not np.isfinite(vals).all() or ((vals<0)|(vals>1)).any():
        raise RuntimeError("Teacher count/probability violation")
    log(f"A6 validated teacher: {path} ({len(pseudo)} studies)")
    return pseudo,str(path)
'''
src=src[:start]+replace+src[end:]
# A3's original target construction can be used to align gold and weak.
# But never let gold UIDs enter optimization or early stopping.
old='train_records=[r for r in records if int(r["fold"])!=fold]'
new='train_records=[r for r in records if int(r["fold"])!=fold and not np.any(r["gold_mask"])]'
assert old in src
src=src.replace(old,new)
old='binary=high_conf_binary_targets(val_target)'
new='''# Gold labels must not select epochs: mask them from the weak proxy.
        val_gold=np.stack([r["gold_mask"] for r in val_records]).astype(bool)
        val_target=np.where(val_gold,np.nan,val_target)
        binary=high_conf_binary_targets(val_target)'''
assert old in src
src=src.replace(old,new)
old='for variant in ("A3-1","A3-2","A3-3"):'
assert old in src
src=src.replace(old,'for variant in ("A3-1",):')
src=src.replace('"EXP-A3 A3-RANK"','"A6-R1 Raptor teacher student (RadImageNet)"')
src=src.replace('A3-RANK TRAINING COMPLETE','A6-R1 STUDENT TRAINING COMPLETE')
# Make artifacts distinctive.
src=src.replace('a3_rank_comparison.json','a6_raptor_student_comparison.json')
src=src.replace('a3_feature_bank.pt','a6_raptor_feature_bank.pt')
src=src.replace('a3_{variant_name.lower()}','a6_{variant_name.lower()}')
# Strict test for non-gold training and teacher origin before emitting code.
assert "and not np.any(r[\"gold_mask\"])" in src
assert "val_target=np.where(val_gold,np.nan,val_target)" in src
assert 'a6_raptor_v5_visual.csv' in src
compile(src,"a6-r1-student.py","exec")
out=root/"build/a6";out.mkdir(parents=True,exist_ok=True)
(out/"a6-r1-student.py").write_text(src)
print("A6 script generated with hard gold exclusion and pseudo-only epoch selection")
