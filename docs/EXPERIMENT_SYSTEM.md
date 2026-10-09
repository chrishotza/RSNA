# RSNA experiment operating system

## Operating rule

**No Kaggle commit, push, run, or competition submission is part of the inner development loop.** The hidden competition rerun is expensive, cannot be used as a fast debugger, and must be reserved for a candidate that passes every cheap gate.

As of 2026-10-09:
- EXP-A0 is the measured anchor: public score 0.940.
- Refs 56904847, 56904871, 56906382 and 56906411 all scored 0.940, but they are repeated submissions of the same Version 2, not independent experiments.
- EXP-A1 has already been submitted once as ref 56988112. Its result is pending; do not resubmit or repush it.
- The active queue is marked IN_FLIGHT until that result is recorded.


### Frozen artifact identity

The file `notebooks/A0_public_0941/rsna-knee-blend-full.ipynb` is the public/source notebook and still contains the expensive `APPENDED ARM BLEND` in cell 23. It is **not** the exact code artifact that produced the measured A0 0.940 score. That evaluation used the cell-23 recovery no-op because those optional arm checkpoints were not available to the submitted kernel.

The measured parent is now frozen separately at `notebooks/A0_public_0941/a0-submitted-freeze.ipynb`: cell 19 keeps `AMP_PREF = 'bf16'`, cell 23 is the exact recovery no-op, and the runtime/data metadata is retained. All future candidate diffs must use this snapshot, not silently rewrite/normalize the public source notebook.

The experiment gate now compares the first Python statement by AST (so inline comments do not cause a false mismatch), freezes execution metadata except the `rsna_experiment` annotation, and rejects content changes outside the declared code cell and the presentation banner. Regression tests were added. GitHub Actions has repeatedly failed before allocating a runner (`runner_id=0`, no steps), so passing CI has **not** yet been observed; do not mark these tests green until an actual run executes them.

## Why submission is expensive

Kaggle's code-competition workflow privately reruns the notebook end-to-end against a hidden version of the competition data, then checks the output and computes the score. It is not just scoring an existing CSV. The public notebook run and hidden scoring run are separate operations.

For all future experiments, the controlled workflow uses `kaggle kernels push --no-run` to save a version without a preliminary execution. It then calls `kaggle competitions submit` at most once. That submission itself performs the hidden rerun. The workflow must never wait for a regular run and then submit the same candidate afterward.

## Required experiment lifecycle

### 1. Formulate

Before touching a notebook, create one experiment record with:
- parent experiment and exact frozen artifact;
- one scientific question and one primary intervention;
- target labels and affected pipeline stage;
- what evidence would support/refute the hypothesis;
- compute cost and stop/keep/revert rule.

No “small tweak” gets a leaderboard submission without a written rationale.

### 2. Cheap offline gates

Use CPU-only checks first:
1. JSON integrity and exactly the expected notebook cell count.
2. Python syntax compilation without executing any model.
3. Exact code diff against the measured baseline; fail if unapproved cells changed.
4. Confirm checkpoint/data hashes, dataset inputs, offline internet setting, submission schema and output-validation gates have not silently changed.
5. Run unit tests and any local/OOF proxy tests available.
6. Estimate runtime and make sure all expected studies are processed; no fallback or partial-test results.
7. Save a validation manifest with baseline/candidate SHA-256 and the exact allowed diff.

The current `scripts/validate_notebook_experiment.py` is a starter implementation of the code-diff/syntax gate. `.github/workflows/validate-experiments.yml` runs it on a cheap CPU runner and has no Kaggle secret or GPU.

**Important limitation:** syntax and code-diff checks are not a substitute for local/OOF metric validation. We must add data-backed OOF tests for an intervention before calling it scientifically validated. If the relevant OOF predictions or data are not available in the repo, explicitly record that gap rather than claiming a metric improvement.

### 3. Freeze and queue

Only after the gates pass:
- freeze the exact candidate notebook;
- set `experiments/queue/active.json` to `READY`;
- use a unique experiment ID, kernel slug, and submission message;
- commit the dispatch marker with a message containing `RUN EXPERIMENT`.

The default queue is intentionally `IN_FLIGHT` for EXP-A1/ref 56988112, so it prevents any new experiment from being pushed right now. Ordinary code commits do not trigger a Kaggle submission.

### 4. Single hidden evaluation

The controlled workflow:
- rechecks queue state;
- rechecks the exact code diff and syntax;
- checks Kaggle submission history for the unique message;
- saves a notebook version with `--no-run`;
- checks history again;
- invokes `kaggle competitions submit` once;
- captures the ref from submission history without retrying the submit command;
- records the ref and version in the queue.

If a command fails after the one submit invocation, inspect history and recover its ref. Never rerun `competitions submit` because the CLI failed to print a ref.

### 5. Poll, don't resubmit

Status checking is read-only. Use `kaggle competitions submissions <competition> -v -q` and locate the exact numeric ref. `PENDING` is not a score. Do not call submit again to make a pending evaluation finish.

When a result reaches COMPLETE, record:
- public score and delta from 0.940;
- submission ref, kernel slug/version, commit and artifact hashes;
- elapsed time, warnings, failure/fallback diagnostics;
- 12-label OOF/P0/P1 values if available.

### 6. Scientific decision

For a candidate `C) against parent `P`:
1. compare complete OOF predictions on fixed folds;
2. inspect per-label ROC-AUC and confidence intervals/seed stability;
3. calculate per-label rank correlation and error complementarity to `P`;
4. select blend weights on nested folds, not the hidden leaderboard;
5. promote only with converging evidence, not one noisy public delta.

Public LB measures one hidden subset and is not a local debugger. Repeated random weight changes are prohibited.

## EXP-A1 interpretation

EXP-A1 changes only cell 19's `AMP_PREF` from `bf16` to `auto`, which should select FP16 on T4. This is a runtime/numerical hypothesis motivated by a reported BF16 issue on T4 in this competition, not a proven explanation for our 0.940. Its single existing ref is 56988112. Wait for that result; do not create another version while it is running.

## Stop conditions

Never submit if:
- the queue is not READY;
- the experiment ID/message already exists in history;
- the diff gate reports more than the declared intervention;
- candidate schema or final-output checks fail;
- a previous evaluation of the same experiment is still PENDING;
- the previous result has not yet been interpreted and recorded.

These are hard stops, not warnings.
