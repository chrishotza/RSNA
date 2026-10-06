# Current public ceiling

As of 2026-10-06, two different public references matter:

## Code anchor

NTejas-1/RSNA-Knee-Abnormality-Detection

- public GitHub repository
- reported public LB: 0.941
- reports a 0.940 parent public-reference ensemble
- repository-authored code/checkpoints are stated as MIT licensed

## Score anchor

Kaggle notebook by Aman Atar:

- public score: 0.943
- best public version shown: V6
- reported runtime: 4m38s on GPU T4 x2
- inputs include the competition data, several public knee-MRI datasets, fold weights and DINOv2/model/notebook sources
- notebook license shown by Kaggle: Apache 2.0

## Strategy

We do not replace the 0.941 code anchor with an unverifiable copy of the 0.943 notebook.

Instead:

1. reproduce the 0.941 GitHub pipeline inside vendor/anchor;
2. reproduce or reconstruct the 0.943 Kaggle notebook as an external score anchor;
3. diff their preprocessing, model sources, aggregation and blending;
4. identify changes that explain the score delta;
5. test each change independently against our parent predictions;
6. only then build a new ensemble.

The competition page currently describes macro ROC-AUC over 12 targets and requires submissions through offline notebooks with a nine-hour runtime cap.
