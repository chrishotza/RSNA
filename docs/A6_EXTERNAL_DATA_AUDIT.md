# A6 clean-data decision

The public 8,631-study "Balanced Labels" table is **not approved as an A6
training source as a whole**.

Reason: its provenance explicitly includes RSNA, MRNet and OAI. The RSNA
competition allows freely/publicly available external data, but current host
clarifications make institutionally gated access a material eligibility risk.
OAI/MRNet therefore cannot silently enter the final training corpus.

A6 remains based on the 4,349 non-gold RSNA studies used by the clean Apex
CoAtNet parent. External datasets can be reconsidered only source-by-source,
with explicit access/license provenance.

## A6 view intervention

The intervention is deliberately orthogonal to the Apex parent:

- PF OA: denser axial sampling;
- Synovitis: denser fluid-sensitive / fat-suppressed sampling;
- Lateral OA + Lateral Meniscus: denser sagittal/coronal coverage;
- all other targets retain the balanced acquisition policy.

Implementation: `src/a6_view_policy.py`.

The policy changes sampling/attention evidence, not labels, gold membership, or
leaderboard-derived target weights.
