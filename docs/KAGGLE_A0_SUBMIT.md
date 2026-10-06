# A0 - Kaggle submission checklist

## Artifact

Use exactly:

notebooks/A0_public_0941/rsna-knee-blend-full.ipynb

Do not edit its code for A0.

## Kaggle setup

1. Open Kaggle.
2. Open the RSNA Knee Abnormality Detection competition.
3. Create a new Competition Notebook, or use Copy & Edit from the public notebook lineage.
4. Upload/open the frozen A0 notebook.
5. Confirm GPU is enabled and select the Tesla T4 environment where available.
6. Confirm Internet is disabled.
7. Attach the competition dataset.
8. Attach the exact public dataset/model/kernel sources from the frozen kernel-metadata.json.
9. Commit/run the notebook.
10. Inspect the final output before submitting.

## Required integrity checks in the run

The notebook must reach the publication cell.

Do not proceed if:

- a required dataset is missing;
- a checkpoint is missing;
- _DINOV2_MATCHED_MEMBERS is not 20;
- V18_CALIBRATOR_APPLIED is false;
- the final submission schema is wrong;
- submission.csv is absent;
- any prediction is NaN or inf.

## Submission

Use the notebook's generated submission.csv.

Record immediately:

- Kaggle submission ID
- public leaderboard score
- displayed rank if available
- notebook version
- total runtime
- any warnings/errors
- exact Kaggle notebook URL

## Return to GitHub

After Kaggle evaluates A0, update:

experiments/submissions.csv

and:

experiments/ledger.csv

with the real public score.

Then save the exact submitted notebook version under:

notebooks/A0_public_0941/kaggle_committed_v1.ipynb

Do not overwrite the frozen source notebook.

## Interpretation

A0 is not an optimization.

Its only job is to establish:

frozen source -> Kaggle execution -> public score

Once A0 validates, A1 and all future experiments can be compared against a known external anchor.

## First calibration target

We expect the A0 result to be near the public score reported by the source project, approximately 0.941, but the result must be measured rather than assumed.

A discrepancy is useful diagnostic information and should be investigated before any model change.
