# RSNA - Knee Abnormality Detection

Private research repository for the RSNA Knee Abnormality Detection 2026 competition.

## Objective

Build an image-only inference system that maximizes macro-averaged ROC AUC across the 12 targets, within the nine-hour Kaggle notebook runtime and offline submission rules.

## Research rule

Every experiment gets a hypothesis, reproducible configuration, offline metrics, per-label AUC, rank correlation against the current parent, runtime/memory, and a logged Kaggle result when submitted.

The Kaggle leaderboard is treated as an external validation signal, not the only source of truth.

## Initial anchor

The strongest public GitHub implementation located for this competition is NTejas-1/RSNA-Knee-Abnormality-Detection, reporting public LB 0.941 and a 0.940 parent public-reference ensemble.

We use it as a reference anchor while preserving attribution and third-party licenses. Its own experiment log says its additional arms were effectively redundant with the parent ensemble.

## Layout

- docs/BASELINE.md - public baseline inventory
- docs/METHOD.md - improvement algorithm and experiment gates
- experiments/ledger.csv - experiment/submission ledger
- configs/ - experiment configurations
- src/ - reusable training/evaluation/inference code
- notebooks/ - Kaggle notebooks

## Non-clinical use

Competition research code only. Not a clinical diagnostic system.
