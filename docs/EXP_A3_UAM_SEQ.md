# EXP-A3 — A3-RANK: Rank-First Anatomical Evidence Reader

Status: RADICAL_REDESIGN_LOCKED / IMPLEMENTATION_STARTED / NOT SUBMITTED

## Objective

The competition scores twelve independent ROC-AUC rankings. A3 therefore does not assume that calibrated pointwise classification is the best training geometry.

The central question is: for each pathology, which study must rank above which other study, and which MRI evidence justifies that order?

A3 is deliberately orthogonal to EXP-A2. A2 routes already-scored families. A3 creates a new image reader, a new supervision geometry and ideally a different error distribution.

## Radical hypothesis

Three mismatches can cap the usual public recipe:

1. Kaggle scores ranking while most training remains pointwise BCE.
2. Report silence is numerically encoded even when it is not a true negative.
3. One generic study representation is asked to serve twelve anatomically different findings.

A3-RANK attacks all three.

## AUC-native supervision

Weak confidence is min(1, 2*abs(y-0.5)). Exact 0.5 has zero weak confidence.

Pointwise BCE remains an auxiliary stabilizer. The primary objective is within-target pairwise ordering. A pair becomes active only when the pseudo-target gap is at least 0.35. For an ordered pair yi > yj, the model minimizes softplus(-(logit_i-logit_j)/temperature).

Gold labels override weak confidence with weight 4.

Default objective:
- 0.35 pointwise masked soft BCE
- 0.65 confidence-weighted pairwise rank loss

These coefficients are hypotheses, not protected constants.

## Unstated means unlabeled

A report-derived 0.5:
- contributes zero weak BCE;
- cannot manufacture a weak ranking pair;
- can later participate in image-only representation learning;
- can become supervised through gold or a future cross-fitted teacher.

This matters especially for report-silent findings.

## Target-specific anatomical routing

Pipeline:

DICOM physical geometry -> adjacent 2.5D windows -> frozen RadImageNet features -> ordered tokens -> BiGRU -> twelve target queries.

A3-RANK adds a learned target-by-acquisition-type attention bias. ACL, MCL, menisci, OA, effusion, synovitis, Baker's cyst, contusion and fracture can therefore learn different preferences over plane and sequence type instead of sharing one fixed study pooling rule.

## Feature-bank contract

Initial encoder: frozen RadImageNet ResNet-50.

Required preprocessing:
- slice order from ImagePositionPatient projected onto the normal derived from ImageOrientationPatient;
- InstanceNumber only as fallback;
- deterministic adjacent 2.5D triplets;
- deterministic anchors;
- fixed resize/crop contract;
- no stochastic augmentation during bank generation.

Per-token metadata must preserve at least anatomical plane, fluid sensitivity, fat suppression and a stable combined acquisition type ID.

Every bank must retain StudyInstanceUID, SeriesInstanceUID, preprocessing receipt and SHA256.

## Validation that attacks shortcuts

Random UID folds are useful for engineering but are not enough evidence. Promotion requires:
- deterministic five-fold UID OOF;
- scanner/site stress validation derived from non-identifying acquisition fingerprints when available;
- gold-58 sanity metrics;
- target-wise rank correlation against the parent.

A gain that vanishes under scanner-held-out validation is shortcut-prone, not progress.

## Experimental ladder

A3-0: feature bank correctness.
A3-1: original pointwise UAM head.
A3-2: same bank/head, rank-first objective only.
A3-3: same objective plus target-specific acquisition router.
A3-4: scanner/site stress validation.
A3-5: complementarity and conditional gain versus parent families.

The point is causal speed: the exact same frozen bank lets us test the risky claims without repeatedly decoding 570 GB or retraining a backbone.

## Promotion criteria

No scoring candidate until we have:
- all five fold checkpoints;
- complete OOF UID coverage;
- zero non-finite outputs;
- feature/preprocessing hashes;
- A3-1 versus A3-2 versus A3-3 controlled comparison;
- per-target OOF AUC;
- gold-58 sanity metrics;
- scanner stress result where possible;
- parent rank-correlation matrix;
- hidden-test runtime extrapolation.

## Falsification

Kill or redesign a claim if pairwise ranking cannot beat pointwise on the same bank, target routing adds no conditional gain, improvements vanish on scanner-held-out validation, predictions are effectively redundant with the parent, or runtime prevents a full hidden pass.

## Current implementation

src/a3_uam_seq.py now implements unstated-aware weights, gold override, masked BCE, confidence-weighted pairwise AUC surrogate, the hybrid rank-first objective, BiGRU ordered-window reading and target-specific acquisition routing.

The next irreversible step is the deterministic DICOM-to-2.5D-to-RadImageNet feature bank.

Submission remains unauthorized while EXP-A2 is being scored.
