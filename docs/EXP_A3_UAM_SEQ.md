# EXP-A3 — UAM-SEQ Independent Reader

Status: DESIGN_LOCKED / NOT SUBMITTED

## Thesis

A3 is deliberately orthogonal to EXP-A2.

EXP-A2 changes how existing model families are fused. A3 creates a new image reader whose training objective and study aggregation differ from every currently scored parent component.

The reader is trained from report-derived soft labels, but **report silence is not treated as a target**.

## Core intervention

### Unstated-Aware Masked supervision

For each soft target y in [0,1]:

```
confidence(y) = min(1, 2 * abs(y - 0.5))
```

Exact 0.5 receives zero weak-label loss.

Known competition gold labels override the pseudo-label and receive weight 4.0.

This makes the training objective explicitly distinguish:

- positive evidence;
- negative evidence;
- report did not address the finding.

It is especially important for targets such as Synovitis, Baker's cyst and Fracture where report non-mention is frequent.

## Visual model

Encoder:
- RadImageNet ResNet-50 weights already used as a legal attached competition dependency.
- Frozen during the initial feature-bank stage.
- The independent signal comes from a new sampling and aggregation graph, not another copy of the current Rad head.

Input representation:
- DICOM slices sorted by physical geometry (ImagePositionPatient + ImageOrientationPatient projection; InstanceNumber fallback only when geometry is unavailable).
- 2.5D adjacent slice triplets preserve the pretrained 3-channel interface.
- deterministic anchors within each acquisition;
- sequence metadata retained: anatomical plane, fluid-sensitive flag, fat-suppression flag.

Study representation:
- token projection;
- bidirectional GRU over ordered local 2.5D windows;
- learned acquisition-type embeddings;
- 12 target queries attend independently over the study token bank;
- one binary logit per finding.

This differs materially from:
- rank blending;
- static target routing;
- simple global pooling;
- the current frozen RadImageNet attention head;
- DINO member averaging.

## Training plan

1. Attach competition data, `pilkwang/rsna-knee-llm-labels`, and the pinned RadImageNet ResNet-50 artifact.
2. Build one deterministic frozen feature bank from all 4,407 training studies.
3. Override pseudo labels with gold labels wherever competition gold is present.
4. Use deterministic five-fold UID hashing.
5. Train five small UAM-SEQ heads from the shared frozen feature bank.
6. Save fold checkpoints plus feature/preprocessing manifest and hashes.
7. Produce OOF predictions for all train studies.
8. Score the 58 gold studies separately.
9. Measure target-wise rank correlation against available parent OOF predictions when they become available.

## Loss

For weak-label row i,target j:

```
w_ij = 2 * abs(y_ij - 0.5)
loss_ij = w_ij * BCEWithLogits(logit_ij, y_ij)
```

For gold cells:

```
w_ij = 4.0
y_ij = exact gold 0/1
```

Loss is normalized by total active weight, not tensor size.

A target with no active supervision in a minibatch contributes zero rather than NaN.

## Determinism contract

- seed = 20261009;
- deterministic fold assignment from SHA256(StudyInstanceUID);
- no random test-time sampling;
- fixed window anchors;
- physical slice ordering;
- fixed resize/crop contract;
- no stochastic augmentation during feature-bank generation;
- exact checkpoint SHA256 recorded.

## Promotion criteria to scoring candidate

A3 inference is not eligible for competition submit until the training run produces:

- all five fold checkpoints;
- zero non-finite OOF predictions;
- complete StudyInstanceUID coverage;
- macro AUC on the 58 gold rows;
- per-target AUC on gold where both classes exist;
- runtime extrapolation to ~1,322 hidden studies;
- rank-correlation matrix versus the parent when comparable predictions are available.

## Why this can be complementary

The current parent is dominated by ensembles trained from conventional soft targets and by static study aggregation. A3 changes two causal axes at once on purpose:

1. **what counts as supervision** — report silence is masked instead of optimized toward 0.5;
2. **how MRI evidence is represented** — ordered local 2.5D windows are aggregated by a sequential target-query model.

This is intended as a new family, not as another member of the existing public ancestry.

## No competition submission authorization

This document authorizes training/research preparation only. It does not authorize a competition submission while EXP-A2 is being scored.
