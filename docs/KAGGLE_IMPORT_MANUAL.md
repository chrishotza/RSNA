# Manual Kaggle import

## A0

1. Open the public GitHub anchor repository NTejas-1/RSNA-Knee-Abnormality-Detection.
2. In Kaggle, create a Competition Notebook for the RSNA competition.
3. Recreate or copy the exact input and model sources from the public implementation.
4. Run once without modifications.
5. Download the notebook JSON and commit it under notebooks/a0_anchor/.
6. Save the resulting submission.csv under artifacts/submissions/A0/.
7. Record the public score and submission ID in experiments/submissions.csv.

## A1

1. Open amanatar/rsna-knee-abnormality-detection.
2. Use Copy and Edit.
3. Do not change any code.
4. Commit, run and submit.
5. Record the public score, submission ID, runtime and notebook version.
6. Download the notebook JSON into notebooks/a1_0943_reference/.

## Authentication

Do not generate new API tokens for the first calibration cycle.

The official Kaggle API supports kernels pull and kernels push, but browser Copy and Edit gives us the cleaner control path while we establish the scoring relationship.

After the manual path is validated, API automation can be introduced without changing experiment semantics.

## Evidence

The current public Aman Atar notebook reports public score 0.943, best version V6, and 4m38s on T4 x2.

For any copied notebook, record the exact notebook version because Kaggle notebooks can change while retaining the same URL.
