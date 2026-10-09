# EXP-A6 — 0.97 View-Diversity Program

## Goal

Stretch target: **0.970 public LB**.

The immediate goal is not to tune another global blend. It is to identify or train
one genuinely new imaging pipeline that improves the current public frontier while
remaining orthogonal to the dominant Raptor/CoAt/DINO/Rad family.

As of 2026-10-09:
- reproducible public frontier in our audit: 0.944;
- D4-lite public frontier: 0.945;
- newly surfaced Apex Grandmaster Stack: 0.950 public LB;
- public medal context around Oct 6: gold threshold approximately 0.958.

## Why A6 exists

Independent analyses converge on the same failure mode:

- more checkpoints of the same Raptor/CoAt pipeline are highly correlated and add little;
- diversity of labels/preprocessing/view is what has measurably improved strong ensembles;
- the weak findings are concentrated around:
  - Synovitis,
  - PF OA,
  - Lateral OA,
  - Lateral Meniscus.

A +0.020 move from 0.950 to 0.970 requires +0.24 cumulative target-AUC.
That can be achieved, for example, by +0.06 on four weak targets while preserving
the other eight.

## Stage A — audit the 0.950 Apex parent

Read-only audit of:
- exact notebook source,
- dataset/model/kernel dependencies,
- rank/probability blend,
- per-target weighting,
- runtime,
- provenance of any newly added arm.

Promotion rule:

1. If Apex 0.950 contains a genuinely new model/pipeline family not already in D4,
   freeze it as the new public parent.
2. If it is only a reblend/reweight of D4-era members, keep D4/0.945 as the
   scientific parent and treat 0.950 as an external score target, not evidence of
   a new representation.

No Kaggle write is required for this stage.

## Stage B — candidate view-diversity arm

If Stage A finds no new independent representation, train one arm with all of the
following frozen before training:

- 2.5D neighbouring-slice inputs;
- physical-scale crop;
- explicit plane and fluid-sensitive metadata;
- target-specific attention;
- loss optimized for ranking quality;
- gold58 excluded from all training and model selection;
- one alternative view/preprocessing family, not another backbone swap.

The primary alternative-view candidates are:

### B1 — axial / patellofemoral emphasis
Purpose: PF OA.

- denser axial sampling;
- wider patellofemoral depth coverage;
- target-specific axial prior only as an attention bias, not as a hard routing rule.

### B2 — fluid-sensitive inflammatory emphasis
Purpose: Synovitis.

- fluid-sensitive series prioritized;
- denser superior joint/recess coverage;
- separate attention query for inflammatory findings;
- teacher labels pre-registered before training.

### B3 — lateral compartment / meniscus view
Purpose: Lateral OA and Lateral Meniscus.

- sagittal + coronal high-density windows;
- alternative depth span/crop from the dominant public corpus;
- no anatomical mirroring until a clean side-estimation experiment supports it.

## Stage C — validation

A6 is not promoted by one gold58 macro number.

Required evidence:

1. fixed-fold OOF over all trainable/report-supervised studies;
2. gold58 used as a regression guard only;
3. target-specific improvement must transfer across folds;
4. per-target rank correlation against existing clean specialists;
5. paired bootstrap and split-half target-swap gate;
6. no target may be selected using predictions from a model trained on that target's
   gold study;
7. runtime leaves >= 60 minutes safety margin under the 9h notebook limit.

## Quantitative gates

A6 view arm proceeds to final integration only if at least one condition holds:

- >= +0.030 AUC on two of the four weak targets with no >0.010 macro regression;
- >= +0.015 average gain across the four weak targets;
- or >= +0.005 macro gain when fused with the strongest clean parent under
  cross-fitted fixed weights.

For a final competition submission aimed at 0.970:

- expected public score from evidence should be >= 0.958;
- stretch estimate should include 0.970;
- candidate must outperform the 0.950 public frontier structurally, not merely by
  changing weights on the same correlated members.

## Stop rules

Reject A6 if:
- improvements exist only on gold58 and not fixed-fold OOF;
- mean rank correlation with the strongest parent is >0.95 on all four weak targets;
- any gain depends on a leaked/unknown-provenance checkpoint;
- the new view hurts two or more previously strong targets by >0.02;
- projected runtime exceeds the competition limit.

## Current action

Workflow `.github/workflows/a6-audit-0950-frontier.yml` is performing the
read-only audit of:
- `sujanmajhisuzan/rsna-knee-apex-grandmaster-stack` (0.950 public LB),
- `tomdifiore/rsna-knee-credible-checkpoints`.

No Kaggle competition submission is authorized by this document.
