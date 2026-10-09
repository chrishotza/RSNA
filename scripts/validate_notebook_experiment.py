#!/usr/bin/env python3
"""Cheap, deterministic validation gate for controlled RSNA notebook experiments.

This script never contacts Kaggle and never runs model inference.
It validates that a candidate notebook has exactly one allowed functional code change
against the frozen A0 source. The known A0 recovery no-op in cell 23 is normalized
on both sides so the comparison reflects the actually measured 0.940 baseline.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

A0_ARM_NOOP = [
    "# A0 RECOVERY BASELINE\n",
    "# The documented arm checkpoints are currently unavailable from Kaggle.\n",
    "# Preserve the untouched parent pipeline so the first external score measures\n",
    "# the reproducible 0.940/0.941 reference path rather than a failed blend.\n",
    "print('[A0_RECOVERY] Arm checkpoints unavailable; leaving _pipeline_stage.csv unchanged.')\n",
]


def text(cell: dict[str, Any]) -> str:
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else str(value or "")


def set_text(cell: dict[str, Any], value: str) -> None:
    cell["source"] = re.findall(r"[^\n]*\n|[^\n]+$", value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_notebook(path: Path) -> dict[str, Any]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(notebook.get("cells"), list):
        raise ValueError(f"{path}: missing notebook cells")
    return notebook


def normalize_a0_recovery_cell(notebook: dict[str, Any]) -> None:
    if len(notebook["cells"]) <= 23:
        raise ValueError("Expected 25-cell A0 notebook with cell 23 arm-blend cell")
    cell = notebook["cells"][23]
    if cell.get("cell_type") != "code":
        raise ValueError("Expected cell 23 to be the arm-blend code cell")
    set_text(cell, "".join(A0_ARM_NOOP))
    cell["execution_count"] = None
    cell["outputs"] = []


def check_python_syntax(notebook: dict[str, Any]) -> None:
    """Compile each ordinary Python cell without executing it; tolerate notebook magics."""
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = text(cell)
        if not source.strip():
            continue
        if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
            continue
        try:
            ast.parse(source, filename=f"notebook-cell-{index}")
        except SyntaxError as exc:
            raise ValueError(
                f"Python syntax error in code cell {index}, line {exc.lineno}: {exc.msg}"
            ) from exc


def validate(
    baseline_path: Path,
    candidate_path: Path,
    experiment_id: str,
    changed_cell: int,
    before: str,
    after: str,
    output_path: Path | None,
) -> dict[str, Any]:
    baseline_original = load_notebook(baseline_path)
    candidate = load_notebook(candidate_path)

    if len(baseline_original["cells"]) != 25 or len(candidate["cells"]) != 25:
        raise ValueError("A0/EXP-A1 gate expects exactly 25 cells in both notebooks")

    meta = candidate.get("metadata", {}).get("rsna_experiment", {})
    if meta.get("id") != experiment_id:
        raise ValueError(
            f"candidate metadata experiment id {meta.get('id')!r} != {experiment_id!r}"
        )

    baseline = json.loads(json.dumps(baseline_original))
    normalize_a0_recovery_cell(baseline)
    if text(candidate["cells"][23]) != "".join(A0_ARM_NOOP):
        raise ValueError("Candidate cell 23 must preserve the measured A0 recovery no-op")

    if not 0 <= changed_cell < len(candidate["cells"]):
        raise ValueError("changed cell index is outside notebook")
    base_lines = text(baseline["cells"][changed_cell]).splitlines()
    cand_lines = text(candidate["cells"][changed_cell]).splitlines()
    if not base_lines or not cand_lines:
        raise ValueError("Declared changed cell must be non-empty")
    if base_lines[0].strip() != before:
        raise ValueError(
            f"baseline cell {changed_cell} first line is {base_lines[0]!r}, expected {before!r}"
        )
    if cand_lines[0].strip() != after:
        raise ValueError(
            f"candidate cell {changed_cell} first line is {cand_lines[0]!r}, expected {after!r}"
        )
    if base_lines[1:] != cand_lines[1:]:
        raise ValueError("The changed cell contains additional changes beyond its declared first line")

    changed_code_cells: list[int] = []
    for index, (base_cell, candidate_cell) in enumerate(
        zip(baseline["cells"], candidate["cells"])
    ):
        if base_cell.get("cell_type") != candidate_cell.get("cell_type"):
            raise ValueError(f"cell {index} type changed")
        if base_cell.get("cell_type") != "code":
            continue
        if text(base_cell) != text(candidate_cell):
            changed_code_cells.append(index)
    if changed_code_cells != [changed_cell]:
        raise ValueError(
            f"Expected only code cell {changed_cell} to differ; got {changed_code_cells}"
        )

    changed_source = text(candidate["cells"][changed_cell])
    if experiment_id == "EXP-A1":
        if "AMP_PREF = 'auto'" not in changed_source.splitlines()[0]:
            raise ValueError("EXP-A1 must change AMP_PREF to 'auto'")
        if "if AMP_PREF == 'auto':" not in changed_source:
            raise ValueError("amp_for() auto branch is missing")
        if "torch.bfloat16 if cc >= (8, 0) else torch.float16" not in changed_source:
            raise ValueError("hardware-aware AMP selection is missing")
        expected_t4_precision = "float16 (T4 compute capability 7.5)"
    else:
        expected_t4_precision = "not inferred by this experiment-specific gate"

    check_python_syntax(candidate)

    manifest = {
        "status": "PASS",
        "experiment_id": experiment_id,
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "baseline_sha256": sha256(baseline_path),
        "candidate_sha256": sha256(candidate_path),
        "changed_code_cells": changed_code_cells,
        "changed_cell": changed_cell,
        "before": before,
        "after": after,
        "expected_t4_precision": expected_t4_precision,
        "model_inference_run": False,
        "kaggle_api_called": False,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--changed-cell", type=int, required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        manifest = validate(
            args.baseline,
            args.candidate,
            args.experiment_id,
            args.changed_cell,
            args.before,
            args.after,
            args.output,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"EXPERIMENT_GATE_FAIL: {exc}", file=sys.stderr)
        return 2

    print("EXPERIMENT_GATE_PASS")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
