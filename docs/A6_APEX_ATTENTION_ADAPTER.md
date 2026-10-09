# A6-Apex inference-only attention adapter

The public Apex CoAtNet 0.949 already evaluates essentially full depth coverage:

- 96-slice canonical volume;
- slots: sagittal FS 26, sagittal non-FS 22, coronal FS 18,
  coronal non-FS 12, axial 18;
- `K_EVAL=94`, i.e. every valid adjacent 3-slice centre from 1..94;
- target-specific learned MIL attention;
- mirror TTA.

Therefore A6 does **not** attempt another window-density sweep. The remaining
inference-only intervention is to change the *relative attention prior* over the
already-complete set of windows for the four target gaps.

Implementation: `src/a6_apex_attention.py`.

The adapter adds a fixed target×slot log-prior to the public attention logits
before softmax. With strength=0 the head is exactly neutral. No model parameter
is changed and no gold label is read at inference.

Pre-registered target priors:
- PF OA: axial emphasis;
- Synovitis: fluid-sensitive sagittal/coronal + axial emphasis;
- Lateral OA: sagittal/coronal emphasis;
- Lateral Meniscus: sagittal/coronal emphasis.

This is a hypothesis until evaluated on a clean holdout. It must not be promoted
from anatomical plausibility alone.
