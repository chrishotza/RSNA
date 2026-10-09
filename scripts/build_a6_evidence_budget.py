#!/usr/bin/env python3
"""Bookkeep A6 evidence without double-counting correlated levers."""
from __future__ import annotations
import argparse,json
from pathlib import Path

PARENT=0.950
TARGET=0.970

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="validation/A6_EVIDENCE_BUDGET.json")
    a=ap.parse_args()

    levers=[
      {
        "name":"resolution_router_mcl_effusion",
        "status":"unconfirmed",
        "evidence_surface":"same clean gold58 checkpoint metadata",
        "macro_delta_if_exact_transfer":0.00407670314502613,
        "countable_expected_delta":0.0,
        "correlation_group":"coatnet_resolution",
      },
      {
        "name":"apex_target_slot_attention",
        "status":"awaiting_holdout",
        "evidence_surface":"anatomical prior + exact public Apex attention contract",
        "macro_delta_if_exact_transfer":None,
        "countable_expected_delta":0.0,
        "correlation_group":"coatnet_attention",
      },
      {
        "name":"convnext_target_fusion",
        "status":"already_in_parent",
        "evidence_surface":"public LB 0.950 parent",
        "macro_delta_if_exact_transfer":None,
        "countable_expected_delta":0.0,
        "correlation_group":"parent",
      },
      {
        "name":"teacher_upgrade",
        "status":"frozen_for_retraining_only",
        "evidence_surface":"gold agreement audit run 37983404784",
        "macro_delta_if_exact_transfer":None,
        "countable_expected_delta":0.0,
        "correlation_group":"supervision",
      },
    ]

    counted=sum(float(x["countable_expected_delta"]) for x in levers)
    report={
      "schema":"a6_evidence_budget_v1",
      "parent_public_lb":PARENT,
      "stretch_target":TARGET,
      "gap":TARGET-PARENT,
      "counted_expected_delta":counted,
      "current_evidence_based_estimate":PARENT+counted,
      "levers":levers,
      "rule":"A lever contributes to expected score only after independent validation. Diagnostic/oracle deltas remain zero in the countable column.",
      "warning":"Do not arithmetically add levers from correlated families until joint predictions are evaluated."
    }
    Path(a.out).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
