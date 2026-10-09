# EXP-A4 — ORTHO

## Objective

Build a lightweight, inference-only candidate that adds **structurally different public-model signal** rather than another variant of the A0/A2 family.

A4 is deliberately orthogonal to:

- A0: DINO / RadImageNet / Raptor / CoAt-heavy public anchor.
- A2: deterministic specialist routing over those existing families.
- A3: newly trained rank-first sequence reader.

A4 must not train on Kaggle. It must be ready to submit as a code-competition notebook from the first runnable version.

## Codename

**A4-ORTHO — Public Foundation Disagreement Router**

## Core hypothesis

A weaker but genuinely different public model can improve a strong ensemble when it is used only where its ranking signal is complementary.

A4 therefore does **not** average every public model uniformly. It:

1. converts every arm to per-target percentile ranks;
2. measures pairwise disagreement per target;
3. treats the strongest reproducible public arm as anchor;
4. allows an orthogonal arm to alter the anchor only when a second independent arm supports the same direction;
5. caps the maximum displacement so one weak specialist cannot destroy a strong target.

This creates a deterministic disagreement router rather than a plain blend.

## Candidate public sources to audit

1. `dreaddevelopment/raptor-knee-widedense`
   - public single-model CoAtNet checkpoint;
   - known public LB ~0.924;
   - useful mainly as a calibrated visual anchor / sanity reference.

2. `tonylica/rsna2026-models`
   - public DINO-derived four-fold model bundle;
   - different acquisition/preprocessing path from current A0 internals;
   - public standalone score reported around 0.832.

3. `tim9510019/rsna-knee-v34-master-models`
   - public notebook logs show 8 foundation models plus BiomedCLIP and DINO-v3 specialist checkpoints;
   - expected use in A4: disagreement/specialist signals, not dominant weight.

4. Public BiomedCLIP Kaggle model/source used by public RSNA notebooks.
   - target role: semantically different medical-image representation.

## Non-goals

- no new training;
- no pseudo-label generation;
- no full A0 reproduction inside A4;
- no blind equal-weight super-ensemble;
- no unbounded target-specific hand tuning to public LB.

## Initial router contract

For each target q:

- anchor rank: A_q
- orthogonal ranks: B_q, C_q
- consensus specialist: S_q = median(B_q, C_q)
- specialist agreement: G_q = 1 - |B_q - C_q|
- anchor disagreement: D_q = |S_q - A_q|
- gate: sigmoid((G_q - g0)/tau_g) * sigmoid((D_q - d0)/tau_d)
- final: A_q + cap_q * gate * (S_q - A_q)

All predictions are ranked again after routing.

Initial defaults:
- g0 = 0.70
- d0 = 0.10
- tau_g = 0.08
- tau_d = 0.05
- global cap = 0.20 rank units

No per-target cap is accepted until supported by an explicit validation source.

## Runtime contract

A4 submission must:
- be inference-only;
- stay below the 9h code-competition limit with margin;
- attach only public Kaggle datasets/models;
- generate exactly `/kaggle/working/submission.csv`;
- never require Internet;
- validate 13 columns, UID order, finite values and [0,1] range;
- emit a receipt containing model identities, hashes where available, runtime and router parameters.

## Promotion gates

A4 can be submitted only if:
1. all public assets resolve uniquely;
2. all checkpoint schemas are validated;
3. sample/test inference completes without fallback;
4. no arm silently substitutes constant predictions;
5. router output differs from every individual arm;
6. estimated hidden-test runtime remains < 8h;
7. submission.csv passes schema and finite-range checks.

## Current state

Scaffold created. Public-asset discovery is the next executable step.
