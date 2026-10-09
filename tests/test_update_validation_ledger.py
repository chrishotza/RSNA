import json, subprocess, sys
from pathlib import Path

def test_update_validation_ledger(tmp_path):
    ledger=tmp_path/"ledger.json"
    report=tmp_path/"report.json"
    ledger.write_text(json.dumps({"experiments":{}}))
    report.write_text(json.dumps({
        "schema":"a5_internal_promotion_v1",
        "source_sha256":"abc",
        "parent":"original",
        "candidate":"eam12",
        "n_studies":58,
        "parent_macro_auc":0.91,
        "candidate_macro_auc":0.92,
        "macro_delta":0.01,
        "rank_macro_delta":0.01,
        "targets_improved":8,
        "targets_degraded":4,
        "paired_bootstrap":{"p_gt_0":0.9},
        "study_leave_one_out":{"sign_flips":0},
        "target_leave_one_out":{"min_without_one_target":0.001},
        "promotion_gates":{"macro_positive":True},
        "promote":True,
    }))
    subprocess.run([
        sys.executable,"scripts/update_validation_ledger.py",
        "--ledger",str(ledger),
        "--report",str(report),
        "--experiment","A5_EAM",
    ],check=True)
    out=json.loads(ledger.read_text())
    assert out["experiments"]["A5_EAM"]["status"]=="PROMOTE_INTERNAL"
    assert out["experiments"]["A5_EAM"]["internal_validation"]["promote"] is True
