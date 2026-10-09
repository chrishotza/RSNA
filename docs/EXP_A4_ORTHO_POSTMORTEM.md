# EXP-A4 ORTHO — Postmortem and lessons

## Outcome

A4-ORTHO was intentionally exploratory: a lightweight, inference-only three-arm public-model experiment designed to test whether structurally different public signals could correct a stronger family through disagreement routing.

The experiment underperformed. The failure is useful because it identifies several concrete anti-patterns for future RSNA submissions.

## What A4 actually used

### Anchor
- Public Raptor CoAtNet:
  - checkpoint: `raptor_ft_coatnet_v4_full.pt`
  - architecture: `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`
  - public notebook notes place this family around ~0.914 live LB.

### Specialist B
- `loop5_v3_track5a_fold0.pth`
- Despite its public naming, this is **not a native BiomedCLIP image encoder**.
- It is a three-view ResNet50 system with a learned contrastive/alignment head called `BiomedCLIPAlignmentHead`.
- Therefore the assumed representation diversity was overestimated.

### Specialist C
- `best_dinov3_finetuned_384.pth`
- Despite the filename, the model class loads:
  `vit_small_patch14_dinov2.lvd142m`
- Therefore this is a DINOv2-Small backbone with a slot-attention head, not a true DINOv3 backbone.

## Router behavior observed

A4 used a capped agreement-gated rank displacement router.

Observed dry-run receipt:
- mean gate activation: ~0.428
- maximum gate activation: ~0.977
- displacement cap: 0.20 rank units

This was too aggressive for specialists whose independent quality was not established.

## Failure modes

### 1. Weak anchor selection

A4 anchored on a public ~0.914 CoAtNet instead of the already measured ~0.940 A0 family.

This created a structural handicap before routing began.

Rule going forward:
> A diversity experiment must preserve the strongest verified predictor as anchor. Diversity is allowed only as a correction term.

### 2. False orthogonality from labels/names

Model filenames and public notebook labels were treated as evidence of architectural independence.

This was wrong:
- “BiomedCLIP” arm was ResNet50-based with a learned alignment head.
- “DINO-v3” arm actually instantiated DINOv2-Small.

Rule going forward:
> Orthogonality must be established from actual model graph, backbone, preprocessing and prediction correlation—not names.

### 3. Agreement was mistaken for correctness

A4 assumed:
> if two specialists agree against the anchor, their agreement is evidence.

This is invalid when specialists are weak, correlated, or share preprocessing artifacts.

Rule going forward:
> Agreement is only actionable when the agreeing models have demonstrated target-specific incremental value against the anchor on held-out validation.

### 4. Router movement was too large

A maximum 0.20 rank displacement allowed weak specialists to materially reorder a strong prediction vector.

Rule going forward:
> Unproven auxiliary signals must start in the 0.03–0.08 correction range, not 0.20.

### 5. Preprocessing contracts were not comparable

Raptor anchor:
- physical crop ~140 mm;
- fluid-sensitive slot routing;
- multiple Sagittal/Coronal subtypes;
- dense slice/window coverage.

V36 specialist path:
- one series per anatomical plane;
- center crop based on image dimensions rather than true physical spacing;
- only five sliding windows;
- materially different information coverage.

Thus model disagreement mixed representation differences with acquisition/preprocessing differences.

Rule going forward:
> Before interpreting disagreement as semantic complementarity, normalize or at least explicitly model acquisition/preprocessing differences.

## Strategic conclusion

A4 does **not** show that heterogeneous models are useless.

It shows that diversity should not replace a strong anchor.

The next experiment should test the opposite extreme:

### Strong-anchor conservative correction

- Anchor: best externally verified family (A0 / corrected A2 / best scored candidate).
- Auxiliary model influence: minimal.
- Per-target gates only.
- Auxiliary weight initialized at 0.03–0.08.
- No auxiliary signal may alter a target unless held-out evidence shows improvement or useful complementarity.
- Prefer reverting toward the strong parent over extrapolating away from it.

## Reusable principle

> Preserve strong signal. Add diversity only as a small, validated residual.

This postmortem is part of the experiment record and should be consulted before any future ensemble/router submission.
