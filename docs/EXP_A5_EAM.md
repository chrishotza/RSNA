# EXP-A5 — EAM (Expectation over Acquisition Marginalization)

## Motivation

The public 0.944 frontier uses a strong community stack (~0.943 public LB) plus a 2.5D ConvNeXt reader (~0.929 standalone) blended in rank space at 30%.

Inspection of the exact reader revealed a train/inference mismatch:

### Training
- if multiple series exist for one acquisition slot, a candidate series is sampled at random;
- slice-window centres are sampled randomly inside equal-depth bins;
- geometric/intensity augmentation is applied;
- the model therefore learns under acquisition/sampling nuisance.

### Inference
- only the largest series in each slot is used;
- a single deterministic linear slice grid is used;
- no marginalization is performed over the nuisance variables seen during training.

The model is therefore trained to approximate an expectation over acquisition choices, but test inference uses one point estimate.

## Core hypothesis

For study x and target q, instead of

    p_q = f_q(x, z0)

for one acquisition realization z0, approximate

    E_z[f_q(x, z)]

over plausible acquisition/sampling realizations z drawn from the same family seen during training.

This is inference-only and does not alter trained weights.

## Realizations

### Original
- largest series per slot;
- original deterministic linspace centres.

### Mid-grid
- largest series per slot;
- bin-midpoint centres, matching the expectation of the training sampler.

### Grid3
- largest series per slot;
- three within-bin phases: 0.2, 0.5, 0.8;
- mean probability across grids.

### Series4
- four deterministic, study/slot-specific candidate-series draws;
- bin-midpoint grid;
- mean probability.

### EAM12
- four deterministic candidate-series realizations × three grid phases;
- 12 predictions per study;
- mean probability.

The deterministic series sampling uses a stable hash of study UID, slot and realization ID, avoiding run-to-run randomness.

## Confidence signal

EAM12 also yields a prediction standard deviation across realizations for every study/target.

This can act as an acquisition uncertainty signal:

- low variance: the target prediction is stable under plausible acquisition choices;
- high variance: the target is sensitive to series/slice selection.

A later fusion can therefore replace a global fixed ConvNeXt weight with confidence-aware target/study weights, but only if held-out evidence supports it.

## Why this is different from A4

A4 inferred confidence from agreement between separate weak/correlated models.

EAM derives confidence from perturbations of the **same trained model under nuisance variables it explicitly saw during training**.

That distinction matters:
- the perturbations have a direct training-time interpretation;
- no assumption is made that two weak specialists agreeing implies correctness;
- uncertainty is local to each study and each target.

## Evidence

The public frontier notebook reports:
- community stack: 0.943 public LB;
- reader alone: 0.929;
- stack + 15% reader: 0.944;
- stack + 30% reader: 0.944 (with another 30% run at 0.943);
- stack + 45% reader: 0.942.

Reader held-out gold-58 per-target AUC reported in the notebook:
- Baker's: 0.976
- Contusion: 0.966
- Medial OA: 0.966
- ACL: 0.965
- Medial Meniscus: 0.963
- MCL: 0.946
- Fracture: 0.911
- Lateral Meniscus: 0.889
- Effusion: 0.880
- Lateral OA: 0.865
- PF OA: 0.843
- Synovitis: 0.761
- macro: ~0.911/0.912

This spread makes a single global reader weight biologically and statistically suspicious: the reader is extremely strong on some findings and much weaker on others.

## Offline promotion gate

A Kaggle dry-run kernel evaluates Original, Mid-grid, Grid3, Series4 and EAM12 on the 58 expert-labeled studies.

It records:
- macro ROC-AUC;
- 12 per-target AUCs;
- realization variance per target;
- paired study-level bootstrap distribution for EAM12 minus Original.

EAM is promoted only if:
1. macro AUC improves over exact Original;
2. gains are not concentrated in one pathological target while broadly degrading others;
3. bootstrap evidence is directionally positive;
4. runtime is compatible with hidden-test execution.

## Disruptive follow-up if EAM passes

The final candidate becomes a two-layer decision system:

1. **Frontier predictor**
   - exact community 0.943 stack;
   - EAM-improved ConvNeXt reader.

2. **Confidence-aware fusion**
   - larger reader contribution for targets/studies with stable EAM predictions;
   - smaller reader contribution when acquisition variance is high;
   - target priors constrained by held-out gold evidence, not leaderboard probing.

This is a maximum-justified deviation: it can move substantially away from the stack where the reader is demonstrably reliable, while falling back toward the strong parent where acquisition uncertainty is high.

No competition submission is consumed during EAM validation.
