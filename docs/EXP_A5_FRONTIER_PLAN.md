# EXP-A5 — FRONTIER FUSION PLAN

## Objective

Use the final daily Kaggle submission only after a reproducible candidate has stronger evidence than the measured 0.940 A0/A1 baseline.

The target is not a cosmetic blend. The experiment should maximize justified deviation from A0 by introducing a genuinely different high-performing public representation while preserving strict validation and runtime controls.

## Evidence available before implementation

### Measured internal competition evidence

- A0/A1 public LB: 0.940.
- A4 ORTHO public LB: 0.908.
- A3 A3-2 rank-first public LB: 0.843.
- A2 first code-submission produced no score because of a single-GPU execution bug; repaired code exists but is not yet re-submitted.
- One daily competition submission remains after A3.

### Fresh public frontier evidence

A newly surfaced public Kaggle notebook:

- `goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944`
- displayed public LB: **0.944**
- runtime shown by Kaggle: about **4m51s on T4 x2**
- visible input: **Knee MRI - Max-Span Dense Corpus**
- architecture label: **Stack + 2.5D ConvNeXt MIL**

This is important because ConvNeXt MIL is not simply another copy of A3's RadImageNet-ResNet sequence head or A4's Raptor/DINO specialists.

### Raptor geometry evidence from public author discussion

Dread Development reports that inference coverage itself materially changes public LB even with identical weights:

- 42 windows: ~0.924
- 62 windows: ~0.927
- full coverage: higher
- Fine Spacing corpus: 80 slices/study, span 2–98%, 140 mm crop, 336 px
- slots: 22 sagittal-fluid, 18 sagittal, 15 coronal-fluid, 10 coronal, 15 axial
- full coverage: 78 windows
- reported score for that configuration: ~0.932

The author explicitly states that density, not simply span, is the lever: widening span at fixed slice count degraded score, while adding slices to preserve sampling density restored/improved performance.

## Scientific support

Recent knee-MRI work independently supports three design principles relevant to A5:

1. **multi-sequence fusion** rather than treating every series as interchangeable;
2. **slice/label-specific attention**, because different abnormalities localize to different slices/sequences;
3. **coarse-to-fine / target-aware modeling**, especially across meniscus, ligament, cartilage/OA and fluid-related findings.

These principles agree with the public 0.944 ConvNeXt-MIL direction and with the observed Raptor density ablations.

## A5 hypothesis

> A high-scoring ConvNeXt 2.5D MIL arm with dense physical coverage contains useful ranking errors that are different from the 0.940 BTKD/Rad/DINO/CoAt family. A large but evidence-constrained fusion can outperform both.

This is intentionally the opposite of A4's failure:
- A4 used a weaker ~0.914 anchor and allowed correlated specialists to move it.
- A5 starts from two independently strong public-LB families: 0.940 and 0.944.

## Candidate hierarchy

### Candidate F0 — exact public frontier
Reproduce the public 0.944 notebook exactly.

Purpose:
- verify assets and inference contract;
- establish an independent strong arm;
- no Kaggle submission required during discovery.

### Candidate F1 — rank-average A0 + ConvNeXt
Global rank blend across all 12 targets.

Grid for offline/gold validation only:
- ConvNeXt weight 0.25 / 0.50 / 0.75.

No public submission used for grid search.

### Candidate F2 — target-aware maximum justified deviation
Per-target choose among:
- A0,
- ConvNeXt,
- rank-average,
- extrapolated residual `A0 + alpha*(ConvNeXt-A0)` only when held-out evidence supports it.

The macro-AUC metric allows each target to be optimized independently before averaging. Any target-specific choice must be derived from held-out predictions, not public-LB probing.

### Candidate F3 — dense-coverage Raptor residual
Only if the exact Fine Spacing public checkpoint and its 80-slice/78-window preprocessing can be reproduced inside the runtime budget.

This arm is lower priority than ConvNeXt because its standalone public score is lower, but it can provide useful target-specific residual signal.

## Promotion gates for the final submission

The final A5 candidate may consume the last daily submission only if all are true:

1. the 0.944 public notebook source and inputs resolve reproducibly;
2. its output can be generated under the 9-hour code limit;
3. all prediction arms cover identical test UIDs and 12 targets;
4. no fallback/constant substitution is used;
5. target-specific choices are based on held-out evidence or exact public score evidence, not invented weights;
6. expected runtime leaves at least one hour of safety margin;
7. final `submission.csv` passes identity, finite-value, [0,1], and deterministic-hash checks;
8. A5 preserves a receipt containing source kernel IDs, dataset/model identities, blend coefficients, and prediction hashes.

## Current action

A GitHub Actions discovery job is pulling and statically auditing:

`goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944`

No competition submission will be made during this discovery phase.
