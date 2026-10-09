#!/usr/bin/env python3
"""Merge an internal validation report into the canonical RSNA validation ledger."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from datetime import datetime, timezone

def load(p):
    return json.loads(Path(p).read_text())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ledger",default="validation/A5_VALIDATION_LEDGER.json")
    ap.add_argument("--report",required=True)
    ap.add_argument("--experiment",required=True)
    ap.add_argument("--artifact",default="")
    args=ap.parse_args()

    lp=Path(args.ledger)
    rp=Path(args.report)
    ledger=load(lp)
    report=load(rp)
    if report.get("schema")!="a5_internal_promotion_v1":
        raise SystemExit("unexpected validation schema")

    exp=ledger.setdefault("experiments",{}).setdefault(args.experiment,{})
    exp["internal_validation"]={
        "report_path":str(rp),
        "report_sha256":hashlib.sha256(rp.read_bytes()).hexdigest(),
        "source_npz_sha256":report["source_sha256"],
        "parent":report["parent"],
        "candidate":report["candidate"],
        "n_studies":report["n_studies"],
        "parent_macro_auc":report["parent_macro_auc"],
        "candidate_macro_auc":report["candidate_macro_auc"],
        "macro_delta":report["macro_delta"],
        "rank_macro_delta":report["rank_macro_delta"],
        "targets_improved":report["targets_improved"],
        "targets_degraded":report["targets_degraded"],
        "paired_bootstrap":report["paired_bootstrap"],
        "study_leave_one_out":report["study_leave_one_out"],
        "target_leave_one_out":report["target_leave_one_out"],
        "promotion_gates":report["promotion_gates"],
        "promote":report["promote"],
        "artifact":args.artifact or None,
        "recorded_at_utc":datetime.now(timezone.utc).isoformat(),
    }
    exp["status"]="PROMOTE_INTERNAL" if report["promote"] else "REJECT_INTERNAL"
    lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+"\n")
    print(json.dumps(exp["internal_validation"],indent=2))

if __name__=="__main__":
    main()
