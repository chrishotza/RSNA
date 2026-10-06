# RSNA improvement method

## 1. Core idea

We are not searching for one giant model. We are building a sequence of non-redundant prediction operators whose errors can be measured and whose ranks improve the final 12-column ROC-AUC.

Pipeline:

raw DICOM -> deterministic geometry -> representation -> study aggregation -> per-label prediction -> rank ensemble -> submission

Each stage can be changed independently.

## 2. Experimental gates

### Gate A - integrity

- exact 12-label schema
- no NaN or inf
- correct StudyInstanceUID order
- deterministic preprocessing
- no train/test leakage
- test-time path works with internet disabled

### Gate B - offline quality

Measure macro ROC-AUC, 12 per-label AUCs, runtime and peak GPU memory.

The 58 expert-labelled studies are the hard gold subset. Report-derived labels may be used for scale, but they are not treated as equivalent ground truth.

### Gate C - novelty

For a new model C, compute rank correlation against the current parent ensemble P, per label.

A high-AUC model that is almost perfectly correlated with P is usually a weak ensemble candidate.

Prefer candidates with higher standalone AUC or complementary errors with meaningful rank disagreement.

### Gate D - Kaggle

Only survivors become leaderboard submissions.

Log public score, score delta, exact notebook or commit, runtime and attached datasets or model sources.

## 3. Search order

### Wave 1 - input geometry

Test one variable at a time:

- physical crop: 130, 140, 150 mm
- slice band
- slices per slot
- plane and sequence slot definition
- laterality normalization

No model change in this wave.

### Wave 2 - study aggregation

Compare masked mean, gated attention, per-label attention, hierarchical slice-to-slot-to-study pooling, and window MIL.

The key question is whether different abnormalities need different spatial evidence.

### Wave 3 - pretrained visual encoders

Benchmark candidates such as RadImageNet ResNet50, DINOv2, DINO-family alternatives available offline in Kaggle, and efficient convolutional backbones that fit the runtime cap.

We care about complementarity with the parent, not just standalone score.

### Wave 4 - supervision

Construct report-derived soft targets, but validate extraction directly on the 58 gold studies.

Never equate unstated, explicitly absent and explicitly present.

Do not perform uncontrolled self-training on model predictions. Public experiments show this can amplify systematic errors.

### Wave 5 - per-label ensemble

Because the metric is macro ROC-AUC, optimize each target independently.

For each label:

1. convert candidate predictions to ranks;
2. test pairwise blends;
3. search a small convex weight grid;
4. keep the smallest change that improves held-out evidence;
5. reject weights that only fit the tiny gold subset without support elsewhere.

## 4. Main ensemble rule

The parent remains the senior signal until evidence proves otherwise.

For candidate C:

P_new[label] = rank_blend(P_parent[label], C[label], w_label)

Initially constrain w_label to a small range. Do not use a global weight when the candidate only helps a subset of labels.

## 5. Experiment selection algorithm

Each next experiment receives:

Priority = ExpectedAUCGain * Complementarity / GPUHours

ExpectedAUCGain comes from prior experiments on the same stage.

Complementarity comes from rank disagreement against the parent.

GPUHours is measured wall time under the actual Kaggle configuration.

This makes the process an evidence-driven search rather than random hyperparameter sweeps.

## 6. Anti-overfitting rule

The 58 gold studies are small.

Therefore:

- use them primarily as a regression and sanity set;
- avoid repeatedly fitting many weights directly to them;
- prefer structural changes supported by multiple signals;
- never treat a single tiny AUC delta as decisive.

The public leaderboard is also not the private leaderboard. A public improvement is evidence, not proof.

## 7. Stop conditions

Stop early when the candidate is clearly redundant, memory or runtime makes the full run infeasible, per-label results show broad regression, the inference path violates the offline rules, or a noisy proxy improves without complementary evidence.

Keep negative results in the ledger so we never pay twice for the same mistake.
