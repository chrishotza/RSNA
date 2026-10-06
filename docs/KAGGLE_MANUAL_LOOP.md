# Manual Kaggle experiment loop

## Why manual first

The current environment has GitHub access but no authenticated Kaggle connector. We therefore keep Kaggle execution manual until the submission path is proven stable.

Do not create or rotate Kaggle API keys yet. They are not required for the first calibration cycle.

## Cycle 0 - calibrate the proxy

### A0: public 0.941 anchor

Use the public reproducible GitHub solution as the reference implementation.

Run it as a Kaggle Competition Notebook without changing the model or preprocessing.

Record:

- source commit
- notebook version
- runtime
- public score
- submission ID
- local proxy P0
- local proxy P1

### A1: public 0.943 stack

Use Kaggle Copy & Edit on:

`amanatar/rsna-knee-abnormality-detection`

The currently visible notebook reports public score 0.943, best version V6, and runtime 4m38s on T4 x2.

Do not modify the inference path for this calibration submission.

Record the same fields as A0.

### A2: controlled change #1

One geometry change only.

Do not combine it with label, architecture or TTA changes.

### A3: controlled change #2

One supervision/training change only.

## Submission discipline

One submission must answer one scientific question.

Every notebook gets an experiment ID in its title and in the output metadata.

Example:

`RSNA EXP-A2 geometry-150mm`

## Calibration rule

After at least four controlled submissions, compare:

- Spearman(proxy, public-LB)
- Pearson(proxy, public-LB)
- Spearman(proxy_delta, LB_delta)
- absolute prediction error of a simple calibrated linear model

Prefer delta calibration because absolute OOF and hidden-test AUC can have different offsets.

After eight or more controlled submissions, freeze the calibration formula for a search epoch. Refit only after the next independent anchor set.

## Stop conditions

If the proxy ranks experiments differently from public LB repeatedly, stop using it for selection and diagnose the mismatch.

Possible causes:

- silver-label bias
- gold subset sampling noise
- train/test distribution shift
- leakage in folds
- preprocessing mismatch
- test-time aggregation mismatch
- leaderboard overfitting

## Final principle

The proxy is a filter, not an oracle.

Kaggle remains the external measurement instrument.
