# A0 source and measured snapshot

Source repository: NTejas-1/RSNA-Knee-Abnormality-Detection  
Source file: `out/notebook_blend_full.ipynb`  
Source blob SHA: `eb92181eb4a7cf6c5da59a357b61b0c9255b7d1e`  
Reported upstream public LB: 0.941

## Exact artifact that produced our A0 score

Use `notebooks/A0_public_0941/a0-submitted-freeze.ipynb` for future experiment diffs. It retains the measured configuration: cell 19 uses `AMP_PREF = 'bf16'`, and cell 23 is the A0 recovery no-op. The final integrity/publication gates remain in cell 24.

Four submissions of the same A0 kernel Version 2 scored 0.940 public AUC: `56904847`, `56904871`, `56906382`, `56906411`. These were repeated evaluations of the same code, not separate experiments.

## Original source copy — not the measured A0 artifact

`notebooks/A0_public_0941/rsna-knee-blend-full.ipynb` retains the original source, including the experimental `APPENDED ARM BLEND` in cell 23. The cell's own comments estimate ~143 seconds per study (~52 hours for 1,300 studies), above the nine-hour runtime budget. Do not submit this file as the A0 reproduction or use it as the measured baseline.

Keep both files: the source copy is for research; `a0-submitted-freeze.ipynb` is the measured parent.