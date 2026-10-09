# Proxy score contract

## Why the proxy cannot equal Kaggle exactly

The competition leaderboard score is computed on hidden test labels. The public leaderboard covers only about 30% of the test data, while the final/private result uses the remaining 70%.

Therefore no local scorer can reproduce the hidden submission score exactly unless hidden labels are available.

Our goal is stricter and more useful:

1. make the local scorer deterministic;
2. use fixed, leak-free study folds;
3. keep expert 58-study evaluation separate from model selection;
4. measure the proxy's ability to predict *LB deltas*;
5. recalibrate the proxy after every controlled Kaggle submission;
6. stop trusting the proxy if its rank correlation with LB degrades.

## Proxy layers

### P0 - silver OOF

All available training studies with report-derived targets.

Use only out-of-fold predictions.

Purpose: fast model-selection signal.

This is not treated as ground truth.

### P1 - gold OOF sanity

The 58 expert-labelled studies.

They are never used as training data or blend-weight optimization data.

Purpose: sanity check and regression detector.

Because n=58 is small, P1 cannot be the sole experiment selector.

### P2 - public-LB calibration

Every manual Kaggle submission records:

- local P0
- local P1
- rank correlation / residual diagnostics
- Kaggle public score
- submission ID
- exact Git commit
- model/config hash

After enough controlled submissions we fit a lightweight calibration of the form:

LB_hat = f(P0, P1, per-label features, complementarity)

The calibration is evaluated leave-one-submission-out.

The proxy is considered useful when it predicts the direction and approximate size of Kaggle changes, not merely when its absolute number looks pretty.

## First calibration protocol

Do not immediately submit ten variants.

Submit a small set of deliberately different anchors:

A. public 0.941 GitHub baseline;
B. public 0.943 stack if reproducible;
C. one controlled geometry modification;
D. one controlled training/supervision modification.

For each:

- calculate P0/P1;
- submit manually through Kaggle;
- record the public score;
- compare deltas.

Four points are only an initial calibration. Prefer >=8 controlled submissions before trusting a fitted mapping.

## Selection rule

For experiment selection, rank these signals:

1. nested-CV delta;
2. stability across seeds/folds;
3. complementarity against parent;
4. public-LB delta;
5. calibrated LB prediction.

A candidate that only wins P1 but does not generalize to other evidence is not promoted.

## Important negative evidence

Kaggle participants have explicitly reported cases where label-set quality measured on 58 gold studies did not reliably predict LB performance, and cases where changing models caused validation/LB correlation to disappear. That is why the calibration layer is mandatory rather than optional.


## Strict OOF inputs

OOF files require unique, non-blank `StudyInstanceUID` values and non-negative integer `fold` IDs. If a target is known, its prediction must be finite; the scorer now raises instead of silently omitting a failed prediction from AUC. Only `NaN` targets are treated as unaddressed cells. Infinite targets are rejected.

## Exact metric

For each of the 12 columns:

ROC-AUC(prediction, target)

Final proxy:

mean(column_auc[12])

No thresholding.

No accuracy.

No F1.

No probability calibration before scoring.

Rank-preserving transformations are allowed because ROC-AUC depends on ordering.
