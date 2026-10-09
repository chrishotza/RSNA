# A0 anchor and controlled experiment protocol

## Objective

Freeze a reproducible parent before modifying anything.

## ANCHOR-A0 - measured parent

Source:
NTejas-1/RSNA-Knee-Abnormality-Detection

Reported public LB: 0.941.

Run the public implementation as a Kaggle Competition Notebook with no source modifications.

Local artifacts we need:

- exact notebook JSON;
- exact dataset/model sources;
- inference log;
- submission.csv;
- all available OOF predictions;
- per-label OOF AUC;
- proxy_silver;
- proxy_gold when the 58-study targets are available;
- runtime and peak GPU memory.

## ANCHOR-B - visible public score reference (not EXP-A1)

Source:
Aman Atar, `amanatar/rsna-knee-abnormality-detection`.

Current visible notebook metadata reports public score 0.943, best version V6 and runtime 4m38s on T4 x2.

Use Kaggle Copy & Edit. Do not modify the notebook before recording its unchanged score.

## Do not create Kaggle API keys yet

Manual Copy & Edit is preferred for the first calibration cycle. API automation is optional later.

## Required submission record

For each submission save:

- experiment_id
- git_commit
- notebook_version
- public score
- submission ID
- runtime
- proxy_silver
- proxy_gold
- per-label local metrics
- exact candidate description

## Promotion rule

No experiment is promoted from A0/A1 unless its change is isolated and its local proxy is available.

The first four calibration points should be intentionally simple:

A0 unchanged anchor
A1 unchanged public 0.943 stack
A2 one geometry change
A3 one supervision change

After A0-A3, calculate the correlation between proxy deltas and Kaggle public-LB deltas.

## Runtime guard

Before any submission candidate is accepted, benchmark the exact inference path at hidden-test scale. A public test with only three visible studies is not sufficient.

T4/P100 do not have native bf16 acceleration. Use fp16 where appropriate and explicitly check `torch.cuda.is_bf16_supported(including_emulation=False)` before any bf16 path.

A recent competition failure showed 50x slower inference from bf16 emulation, causing a partial test fallback and a severe LB collapse. Therefore runtime is a hard gate, not an optimization after scoring.
