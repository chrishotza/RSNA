# ANCHOR-B audit — pre-EXP-A2

Date: 2026-10-09

## Purpose

ANCHOR-B is the separate public Kaggle notebook `amanatar/rsna-knee-abnormality-detection`, whose current visible title is:

`RSNA Knee Abnormality Detection — Beyond the Public SOTA · FINAL Build (v6.0.0 A2PA)`

Project notes previously record a visible public score of 0.943 and runtime 4m38s on T4 x2. Treat these as an external score reference until the exact scored source/version is captured and reproduced.

ANCHOR-B is not EXP-A1.

## What our measured parent already does

Frozen parent:
`notebooks/A0_public_0941/a0-submitted-freeze.ipynb`

Measured public AUC: 0.940.

The parent already contains multiple families and guarded paths:
- DINOv2 ensemble, 20 fingerprint-matched members.
- A5 five-fold component.
- RadImageNet E11/E13 family.
- Raptor family.
- CoAtNet residual-gated and D4 paths.

Current Raptor contracts are not one generic preprocessing recipe. The frozen parent explicitly routes:
- `maxspan-v5`: 336 px, 2–98% span, 62 eval windows.
- `native384dense-v10`: 384 px, 2–98% span, 62 eval windows.
- `native384-v8`: 384 px, 6–94% span, 42 eval windows.

The parent also already uses average ranks for ties through `rsna_rank01` / `rsna_rankpct`.

Therefore two commonly proposed public fixes are not automatically new here:
1. replacing unstable double-argsort tie ranking;
2. assigning every Raptor checkpoint a single shared 336/2–98%/62-window contract.

Any candidate based on either idea must first prove that the measured parent still violates the exact checkpoint contract.

## Required source capture for ANCHOR-B

Before promoting EXP-A2:
1. capture exact scored notebook version/source;
2. inventory all dataset, model and notebook inputs;
3. identify checkpoint hashes where available;
4. record preprocessing per family: resolution, physical crop/FOV, slice span, window count, ordering and laterality;
5. record output routing and per-target blend weights;
6. record runtime assumptions and GPU topology;
7. distinguish source code that is merely present from model arms that actually contribute to the scored output.

## Diff questions

Compare ANCHOR-B to A0 on these axes only:

| Stage | Question |
|---|---|
| Input | Does ANCHOR-B use different physical FOV, slice span, sequence slots or ordering? |
| DINO | Different checkpoints, windows, pooling, TTA, or merely the same ancestry? |
| Raptor | Different checkpoint family or just different routing/weights? |
| CoAtNet | Different model/checkpoint/input contract or same public D4/resgated family? |
| Labels | Any change to training label source/soft targets? |
| Blend | Which per-label ranks/weights differ from A0? |
| Runtime | Which arms actually execute on hidden scale within the cap? |

## Promotion rule

EXP-A2 stays `RESEARCH_BLOCKED` until one attributable intervention is identified.

A candidate may be promoted only if:
- its source/checkpoint provenance is clear;
- the change is isolated;
- all known-label predictions are finite;
- study coverage is complete;
- runtime is feasible;
- paired OOF/local evidence exists where obtainable;
- the expected benefit is not based only on public-LB weight fitting.

No Kaggle submission is authorized by this document.
