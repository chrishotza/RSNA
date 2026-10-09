# Baseline inventory

## Competition facts

The task is to predict 12 confidence scores per knee MRI study. The metric is the macro-average of twelve ROC-AUC values.

Submission columns:

StudyInstanceUID, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture

Submission notebooks must run with internet disabled and within the nine-hour execution cap. Public external data and pretrained models are allowed by the competition rules.

## Public anchor

Repository: NTejas-1/RSNA-Knee-Abnormality-Detection

Reported public leaderboard: 0.941

Reported parent public-reference ensemble: 0.940

Our measured A0 also scored 0.940 across refs `56904847`, `56904871`, `56906382` and `56906411`; these are repeated evaluations of one kernel version, not independent experiments. The exact tested artifact is `notebooks/A0_public_0941/a0-submitted-freeze.ipynb`. The source notebook still contains an experimental cell-23 arm blend and is not the measured artifact.

Reported components include RadImageNet ResNet50, ImageNet ResNet50, gated attention pooling, six fixed MRI slots, physical-FOV cropping, laterality normalization, soft report-derived targets, five-fold training and rank-based blending.

## Important negative result

The anchor repository reports that its additional arms did not improve the public score over the 0.940 parent ensemble despite being different from each other. Its analysis attributes this to architectural redundancy with components already present in the parent ensemble.

Therefore our first principle is:

Measure novelty against the complete parent ensemble, not merely between our own candidate models.

## Reproduction requirements

Before modifying the method we must verify:

1. parent inference path;
2. input geometry;
3. every backbone and checkpoint source;
4. fold composition;
5. report-label generation source;
6. per-label baseline metrics where available;
7. runtime and memory;
8. submission integrity.

A candidate that cannot reproduce the anchor is an environment problem, not an experiment.

## Provenance

The anchor repo states that its own code and checkpoints are MIT licensed. It separately identifies third-party assets including the public reference ensemble, RadImageNet, DINOv2 and other Kaggle datasets/models.

This repository preserves provenance and does not relicense third-party material.
