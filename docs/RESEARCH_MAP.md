# RSNA research map - 2026-10-06

## Goal

A reproducible path from the current public 0.94x region toward the highest possible macro-ROC-AUC.

A perfect 1.0000 is an aspirational optimization target, not a guaranteed attainable result. We must distinguish leaderboard improvement from genuine generalization.

## Evidence hierarchy

1. Official competition rules/data/evaluation.
2. Reproducible public code with exact reported score.
3. Public Kaggle notebook with a visible public score.
4. Controlled Kaggle discussion experiments with OOF details.
5. Peer-reviewed / arXiv medical-imaging evidence.
6. Anecdotal claims without reproducible details.

Claims from levels 4-6 are hypotheses until reproduced.

## Current public anchors

### Code anchor

NTejas-1/RSNA-Knee-Abnormality-Detection reports:

- parent public ensemble: 0.940
- repository result: 0.941
- 12-target macro ROC-AUC
- 5-fold training
- rank-based blending
- 6 MRI input slots
- 140 mm physical crop
- laterality normalization
- gated attention pooling
- soft report-derived targets

Most important negative result: its extra RadImageNet and ImageNet ResNet50 arms were different from one another but largely redundant with the parent ensemble.

### Visible public notebook anchor

Aman Atar's Kaggle notebook currently reports:

- public score: 0.943
- best version: V6
- runtime: 4m38s on T4 x2
- DINOv2 plus several public knee-MRI / fold-weight / notebook sources

This is a score anchor, not automatically a method to copy blindly.

### Reported current ceiling

The NTejas project page reported a leader around 0.958 at the time of writing; a later public post by the same author reported the current #1 around 0.959. We treat ~0.959 as an externally reported benchmark, not as a substitute for direct leaderboard verification.

## High-value competition evidence

### 1. Labels are necessary but extraction is not obviously the final bottleneck

A public report-label study measured LLM extraction around 0.878 macro AUC against the 58 gold studies versus about 0.814 for regex/lexicon extraction.

The more important finding was distributional: 25.4% of label cells were "not addressed" by the report. This varies dramatically by finding; synovitis was reported as not addressed for about 83.7% of gold cells.

Implication:

- keep a strong soft-label teacher;
- model silence explicitly;
- do not assume better wording alone can recover findings absent from the report;
- use image supervision to learn what the report omits.

### 2. HARD -> SOFT needs replication and uncertainty accounting

A controlled replication using DINOv2 and three paired seeds found SOFT targets above HARD targets in all three seeds and a +0.0143 macro-AUC rank-mean ensemble on the 58 gold studies.

However, the paired bootstrap 95% interval crossed zero: [-0.0041, +0.0330].

Implication:

- SOFT is a serious candidate;
- do not call +0.0143 a settled effect;
- evaluate on the full report-derived corpus and preserve the 58 gold studies as an external sanity set.

### 3. Input geometry is a first-class model component

Controlled competition discussions identify four recurring failure modes:

- DICOM filenames are not anatomical order; use ImagePositionPatient and ImageOrientationPatient geometry.
- fixed-pixel crops change physical field of view across scanners; fixed physical crops are preferable.
- laterality must be normalized without silently swapping medial/lateral labels.
- stacking many slices as channels into a pretrained 2D encoder can destroy the pretrained input interface; 2.5D adjacent triplets preserve the native interface.

Independent public experiments report useful settings around 224-288 px and physical crops around 130-150 mm. A separate participant reported gains from retaining 24 -> 32 slices per series. Another reported 392 px / 150 mm / 32-slice study bags. These are hypotheses, not universal optima.

### 4. Slice sampling structure matters

One controlled implementation uses three anchors across each sorted series, with three physically adjacent slices around each anchor, yielding 3x3 slices. The motivation is to give each encoder call a local 3D neighborhood rather than nine widely separated channels.

Do not assume "more slices" is automatically better. Test:

- 3 adjacent
- 5 adjacent
- 3x3 anchor windows
- 9/16/24/32 study-level samples
- deterministic versus stochastic sampling

### 5. Model size is not sufficient

A competition participant reported a near-null DINOv2-S -> B change of about +0.0011 under paired OOF evaluation. Another reported that a small ResNet at 224 px reached 0.936 single-fold and 0.943 after later changes.

Implication:

- do not spend GPU budget simply scaling backbone size;
- compare initialization, architecture and training strategy as separate causal variables;
- use small models for geometry/label ablations because they are cheap.

### 6. Complementarity is a primary objective

A team at ~0.943 reported an external arm with rank correlation about 0.83 against its eight-signal ensemble and a public gain around +0.004 across four submissions.

Multiple competitors independently emphasize that disagreement / decorrelation matters more than adding another model from the same family.

Implication:

- compute per-label Spearman/rank correlation against the entire parent;
- reject candidates that are both expensive and redundant;
- prefer a modest standalone model with genuinely different errors.

### 7. TTA is worth testing, but only under strict runtime accounting

A participant reported 0.942 raw and 0.943 with TTA at 288x288.

Another showed that a bf16 mistake on T4/P100 turned a nominal 0.86 held-out model into a 0.71 public result because inference exceeded the runtime budget and many studies received a fallback prediction.

Implication:

- TTA must be benchmarked on the exact scoring GPU;
- use fp16 where appropriate on T4;
- measure marginal seconds/study, not total elapsed time divided by a tiny test subset;
- never allow partial-test fallback to silently pass.

## External data / license gate

Do not train on an external MRI dataset merely because it is downloadable.

The current competition rules allow freely and publicly available external data, subject to the competition's specific requirements. Public discussions raise unresolved or restrictive access/licensing concerns for several knee datasets.

Examples:

- fastMRI requires an application and data-sharing agreement; the official site states internal research/education use and no redistribution without permission.
- MRNet requires a Stanford research-use agreement.
- SKM-TEA requires account access and provides dense pathology/tissue annotations.
- OAI access involves a data-use certification / institutional requirements.

Because the competition also has winner obligations involving public release of code and weights, we should use external data only after a dataset-by-dataset legality/provenance gate confirms it is safe for the competition.

## New research direction: knee-specific self-supervision

KneePreM (2026) reports a knee-specific 3D masked-autoencoder pretrained on 19,011 unlabeled OAI MRI series and improved transfer on several downstream knee tasks.

The dataset used for its original pretraining is not automatically competition-eligible. The important reusable idea is the method:

- self-supervised 3D masked reconstruction
- knee-specific representation
- label-efficient fine-tuning

Our legal competition analogue is to pretrain on unlabeled RSNA competition images themselves, or on a dataset that passes the external-data gate, then fine-tune for the 12 targets.

## Medical evidence: why specialized branches are plausible

Peer-reviewed knee-MRI studies show strong task-specific performance for ACL, meniscus, cartilage and bone findings.

The literature supports:

- structure-specific localization before classification;
- 3D context for cartilage / meniscus / ACL;
- external validation across scanners;
- AI outputs that can complement radiologists.

This supports exploring task-specialist branches rather than assuming one shared head is optimal for all twelve labels.

## Research program

### Phase A - reproduce

A0. Public 0.941 code anchor.
A1. Public 0.943 Kaggle score anchor.
A2. Exact baseline inference harness.
A3. OOF prediction cache for every baseline component.

### Phase B - cheap causal ablations

B1. slice sorting.
B2. physical crop.
B3. laterality.
B4. 2.5D grouping.
B5. number of slices.
B6. image resolution.
B7. soft vs hard targets.
B8. report-label source.

### Phase C - strong single-model families

C1. ResNet/EfficientNet.
C2. DINOv2.
C3. DINOv3.
C4. ConvNeXt / ConvNeXt-V2.
C5. MAE-pretrained ViT or Swin.
C6. knee-specific SSL pretraining.
C7. structure-aware / specialist branches.

### Phase D - study aggregation

D1. masked mean.
D2. gated attention.
D3. hierarchical slice -> window -> series -> study.
D4. label-specific attention.
D5. mixture-of-experts across sequences.
D6. uncertainty-aware aggregation.

### Phase E - target-aware supervision

E1. soft report labels.
E2. explicit present/absent/unstated.
E3. confidence-aware BCE.
E4. label smoothing only where justified.
E5. specialist label models.
E6. OOF pseudo-labeling with strict teacher/student separation.

### Phase F - ensemble

For each target independently:

1. compute OOF predictions for every candidate;
2. compute rank correlation against the parent;
3. estimate conditional gain;
4. fit constrained blend weights in nested folds;
5. reject unstable weights;
6. submit only a small number of high-information candidates.

### Phase G - leaderboard

Every submission is tagged with:

- experiment ID
- parent commit
- candidate checkpoints
- input geometry
- label version
- TTA recipe
- estimated runtime
- observed runtime
- public score
- delta from parent

No unexplained submission is allowed into the final ensemble.

## Decision rule

A candidate survives when it passes at least two independent checks:

1. out-of-fold improvement on report-derived targets;
2. improvement or stable non-regression on the 58 gold sanity set;
3. low rank correlation / complementary residuals to the parent;
4. public-LB improvement;
5. no unacceptable runtime/licensing risk.

A public-LB gain without complementary OOF evidence is considered suspicious.

## Core hypothesis

The path from ~0.94 to the top tier is more likely to come from stacking several partially independent sources of signal than from a single oversized backbone.

Candidate sources:

- better physical sampling
- local 2.5D geometry
- report-derived soft supervision
- image-only signal on report-silent findings
- task-specialist heads
- a genuinely different visual pretraining family
- TTA
- carefully constrained per-target blending

The final system should look less like "one model" and more like an auditable evidence-weighted set of complementary predictors.
