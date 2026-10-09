# A6 External Frontier Audit — TianK003 line

Date: 2026-10-09

Independent public repository audited:
`TianK003/RSNA-KneeMRI-kaggle-competition`.

This is used as external experimental evidence, not as our own validation.

## Critical provenance correction

The public 0.949 CoAtNet checkpoint underlying the Apex 0.950 stack is **not**
a clean 4,349-only RSNA model.

The external ledger documents:
- 2,399 OAI knees were added to training;
- the corresponding no-OAI level was approximately 0.942;
- the OAI/data-expansion step was reported around +0.005, with confounding;
- gold58 was repeatedly used during development / model selection and is explicitly
  described as directional rather than independent CV.

Therefore:
- Apex 0.950 remains a valid hidden-public-LB anchor;
- its gold58 component metrics are not admissible as independent promotion evidence;
- OAI eligibility/access remains a separate final-submission risk.

Our arm registry was corrected accordingly.

## Blend ceiling observed externally

The strongest public 0.949 anchor plus a ~0.943 independent leg at beta 0.35
produced C4 = 0.950: only +0.001 over the anchor.

The external ledger's repeated empirical rule is:
- same-family additions usually add nothing or dilute;
- cross-family diversity often contributes ~+0.002 to +0.003;
- fork lift is approximately diversity gain minus beta times the anchor-leg gap;
- improving a much weaker blend leg by +0.001 has only a fraction-of-a-tick effect
  on a 0.949 anchor.

Implication: **0.970 cannot plausibly be reached by adding more ~0.94 models to
the existing 0.949/0.950 parent.**

## Strongest transferred target-side result

The most important external result for A6 is not an architecture change.

A Raptor visual teacher was run over the 4,349 report-only RSNA studies and mixed
50/50 with the LLM targets. In the external experiment lineage, this target-source
change produced a substantial single-model public improvement (approximately
0.918 -> 0.927). Same-family self-distillation and several other label variants
did not show comparable transfer.

This raises the priority of **cross-family visual-teacher supervision** for A6.

## Other negative knowledge

- test-time ConvNeXt resolution 288 -> 320 read the same public score;
- a second ConvNeXt seed did not improve the family level;
- many extra same-family ensemble members did not improve the best blend;
- gold58 has predicted the direction of public-LB changes incorrectly multiple times.

## A6 consequence

Primary non-OAI path to investigate:
1. frozen stronger public text teacher plan;
2. regenerate a public visual teacher from a genuinely different family;
3. train only on the 4,349 non-gold studies;
4. specialize representation for the remaining weak targets;
5. use gold only as a regression diagnostic, never as the sole selection surface.

The stretch goal remains 0.970, but no unvalidated diagnostic delta is counted
toward the expected score.
