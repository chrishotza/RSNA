# A6-RR — Resolution Router hypothesis

## Discovery

The official Apex publication package contains two independently submitted CoAtNet
checkpoints trained on the expanded non-gold corpus:

- accuracy arm: 384 px, public LB 0.949, gold macro 0.9189517821;
- efficiency arm: 224 px, public LB 0.945, gold macro 0.9147251129.

Although 384 wins macro, the target-level errors are not nested.

### Gold58 target deltas: 224 minus 384

| Target | 224 | 384 | Delta |
|---|---:|---:|---:|
| ACL | 0.96446 | 0.95956 | +0.00490 |
| MCL | 0.97506 | 0.94104 | **+0.03401** |
| Medial Meniscus | 0.96514 | 0.98438 | -0.01923 |
| Lateral Meniscus | 0.86832 | 0.85963 | +0.00870 |
| Medial OA | 0.97519 | 0.97054 | +0.00465 |
| Lateral OA | 0.78917 | 0.77756 | +0.01161 |
| PF OA | 0.84942 | 0.84041 | +0.00901 |
| Effusion | 0.97516 | 0.96025 | **+0.01491** |
| Synovitis | 0.78136 | 0.83871 | -0.05735 |
| Baker's | 0.94565 | 0.97464 | -0.02899 |
| Contusion | 0.94332 | 0.95682 | -0.01350 |
| Fracture | 0.94444 | 0.96389 | -0.01944 |

The same-data oracle (choosing the better resolution for each target) is 0.92627
versus 0.91895 for 384 alone. **This is diagnostic only, not an unbiased score.**

## Hypothesis

Resolution changes the effective texture/context bias enough to create useful
per-target diversity. A constrained router may therefore improve the 384 parent
without introducing a new backbone.

## Guardrail

Do not choose all positive deltas. That would be direct gold overfit.

Pre-registered high-margin candidates for independent confirmation:
- MCL: 224 candidate (gold margin +0.034);
- Effusion: 224 candidate (gold margin +0.0149).

Lateral OA / PF OA / Lateral Meniscus remain research signals until an independent
validation surface supports them.

Synovitis, Baker's, Contusion, Fracture and Medial Meniscus remain 384 by default.

## Evidence needed

Promotion requires one of:
1. OOF/cross-fitted predictions from both resolution arms;
2. an independent public leaderboard ablation with target swaps;
3. clean study-level gold predictions enabling split-half/bootstrap confirmation.

Checkpoint metadata alone is insufficient for promotion.
