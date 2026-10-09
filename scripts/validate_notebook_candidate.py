#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def source(cell: dict[str, Any]) -> str:
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else str(value or "")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(nb.get("cells"), list):
        raise ValueError(f"{path}: missing cells")
    return nb


def parse_indices(raw: str) -> set[int]:
    raw = raw.strip()
    if not raw:
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def compile_code(nb: dict[str, Any]) -> None:
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        text = source(cell)
        if not text.strip():
            continue
        if any(line.lstrip().startswith(("%", "!")) for line in text.splitlines()):
            continue
        try:
            ast.parse(text, filename=f"cell-{i}")
        except SyntaxError as exc:
            raise ValueError(f"syntax error in cell {i}, line {exc.lineno}: {exc.msg}") from exc


def validate(
    baseline_path: Path,
    candidate_path: Path,
    experiment_id: str,
    allowed_code_cells: set[int],
    allowed_markdown_cells: set[int],
    required_tokens: list[str],
    output: Path | None,
) -> dict[str, Any]:
    base = load(baseline_path)
    cand = load(candidate_path)

    if len(base["cells"]) != len(cand["cells"]):
        raise ValueError("cell count changed")

    meta = cand.get("metadata", {}).get("rsna_experiment", {})
    if meta.get("id") != experiment_id:
        raise ValueError(f"metadata experiment id {meta.get('id')!r} != {experiment_id!r}")

    changed_code: list[int] = []
    changed_markdown: list[int] = []
    for i, (a, b) in enumerate(zip(base["cells"], cand["cells"])):
        if a.get("cell_type") != b.get("cell_type"):
            raise ValueError(f"cell {i} type changed")
        if source(a) == source(b):
            continue
        if a.get("cell_type") == "code":
            changed_code.append(i)
        else:
            changed_markdown.append(i)

    if set(changed_code) != allowed_code_cells:
        raise ValueError(f"code diff mismatch: expected {sorted(allowed_code_cells)}, got {changed_code}")
    if set(changed_markdown) != allowed_markdown_cells:
        raise ValueError(f"markdown diff mismatch: expected {sorted(allowed_markdown_cells)}, got {changed_markdown}")

    all_candidate = "\n".join(source(c) for c in cand["cells"])
    missing = [token for token in required_tokens if token not in all_candidate]
    if missing:
        raise ValueError(f"required candidate tokens missing: {missing}")

    compile_code(cand)

    manifest = {
        "status": "PASS",
        "experiment_id": experiment_id,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "baseline_sha256": sha256(baseline_path),
        "candidate_sha256": sha256(candidate_path),
        "changed_code_cells": changed_code,
        "changed_markdown_cells": changed_markdown,
        "required_tokens": required_tokens,
        "model_inference_run": False,
        "kaggle_api_called": False,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--allowed-code-cells", default="")
    p.add_argument("--allowed-markdown-cells", default="")
    p.add_argument("--require-token", action="append", default=[])
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    manifest = validate(
        args.baseline,
        args.candidate,
        args.experiment_id,
        parse_indices(args.allowed_code_cells),
        parse_indices(args.allowed_markdown_cells),
        args.require_token,
        args.output,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
