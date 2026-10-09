# A5 / 0.97 Target Specialist Matrix

## Objective

Raise the public frontier from the reproducible 0.944–0.945 region toward a stretch target of 0.970 by replacing the idea of one universal blend with evidence-gated target specialists.

A +0.025 macro-AUC improvement requires +0.30 cumulative AUC across the 12 targets. One plausible path is +0.05 on six weak targets while preserving the six strong targets.

## Public frontier evidence

### D4-lite 0.945

The exact public notebook `pjmathematician/rsna-knee-d4-lite` reports 0.945 LB and uses a large rank-space deviation from the public parent:

- public parent weight: 0.55
- private/core weight: 0.45
- optional student leg default: 0.00

Inside the CoAt-family outer blend, default CoAt weight is 0.60 with stronger target-specific weights:

- ACL: 0.75
- Medial Meniscus: 0.80
- Lateral Meniscus: 1.00
- Lateral OA: 0.75
- Fracture: 0.75
- all other targets: 0.60

The legacy public member also contains target-specific window/member emphasis for Lateral Meniscus, Medial OA, Lateral OA and Contusion, plus soft-window pooling on ACL, MCL, both menisci, Baker's, Contusion and Fracture.

This is direct evidence that target-specialized routing is part of a stronger public frontier.

### D4 depth-zone arm

The notebook validates a `d4_zones` arm with:
- 384 px inference;
- canonical sagittal handling;
- rank after probability averaging;
- a three-zone depth adapter;
- expanded FSX attention;
- SWA parent epochs [15,16];
- depth adapter SWA epochs [11,9,5];
- no fallback studies permitted.

## Validation rule adopted from the public gold harness

The public `rsna-blend-gate-harness-58-golds-12-targets` uses:

- 400 paired study-level bootstrap resamples;
- target swap selection only when bootstrap win fraction >= 0.90;
- Holm family-wise correction at alpha=0.05 as an additional high bar;
- split-half selection/evaluation to reveal winner's curse.

Our implementation is `scripts/validate_target_swaps.py`.

## Scientific priors for specialist families

These are priors for search, not weights to be hard-coded without validation.

| Target | Strongest evidence direction | Candidate specialist family |
|---|---|---|
| ACL | Sagittal PD/PDFS; multi-plane attention also helps; literature reports ~0.96–0.99 AUC in dedicated systems | D4 depth-zone / sagittal-dense CoAt + dedicated ligament arm |
| MCL | Coronal-dominant with multi-plane support | CoAt target attention / coronal specialist |
| Medial Meniscus | Sagittal + coronal, slice/label attention | D4/CoAt + ConvNeXt label-aware MIL |
| Lateral Meniscus | Sagittal + coronal; strongest public D4 target weighting | D4 CoAt family, allowed to dominate if cross-fit passes |
| Medial OA | Coronal structural/cartilage evidence | D4/Global96 + OA specialist |
| Lateral OA | Coronal structural/cartilage evidence; explicit D4 emphasis | D4/Global96 |
| PF OA | Axial/patellofemoral evidence | axial/OA specialist; current ConvNeXt reader is weak here |
| Effusion | Fluid-sensitive sequences; multi-plane | fluid-sensitive specialist / segmentation-derived signal |
| Synovitis | Fluid-sensitive, difficult and label-noisy | dedicated fluid/synovial arm; avoid trusting generic reader |
| Baker's | Posterior fluid lesion; sagittal/axial evidence | ConvNeXt reader already strong; preserve unless challenger wins |
| Contusion | Fluid-sensitive bone marrow signal; depth coverage matters | D4/FineSpacing/Global96 |
| Fracture | multi-plane cortical/trabecular signal; dedicated texture model plausible | D4 CoAt + RadImageNet/ConvNeXt texture arm |

## Promotion policy toward 0.970

A specialist can replace the parent for a target only if:

1. paired bootstrap win fraction >= 0.90;
2. the direction is positive in split-half cross-fitting;
3. the improvement is not produced by fallback or UID mismatch;
4. inference runtime remains within the 9h limit;
5. the final decision is recorded in the validation ledger.

The 58 gold studies are not used to continuously tune arbitrary weights. They are a gate, not a leaderboard.
