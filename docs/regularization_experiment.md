# Round 3: EXPLORATORY regularization sensitivity

Run `src/regularization_experiment.py` with F5 in Spyder using the existing
`D:\Anaconda\envs\icsi436-p\python.exe` interpreter. No package installation,
environment update, model change, or CUDA configuration change is needed.
The real sweep is intended for manual execution; this document supplies no
invented sweep results.

## Controlled question and design

How does the downstream Logistic Regression regularization strength affect
classification with TF-IDF versus frozen MiniLM features?

Only **C** changes: **0.1, 1.0, 10.0**. These are factor-of-ten steps around the
existing baseline. Smaller C means stronger L2 regularization, discouraging
large coefficients; larger C permits a closer training fit but can overfit.
We do not assume increasing C improves validation performance.

| Component | Fixed setting |
| --- | --- |
| Dataset | Pinned AG News original training split |
| Training | Existing 400-row sets, 100/class, seeds 42/123/2026 |
| Validation | Same existing ordered 800 rows, 200/class, disjoint from every training set |
| TF-IDF | Existing preprocessing/vocabulary settings; fit only on each seed's training text |
| MiniLM | Existing frozen, normalized float32 384-D embeddings; same model revision/settings |
| Classifier | Multinomial LR, L2 (`l1_ratio=0`), lbfgs, max_iter=2000, tol=1e-4, intercept enabled |
| Classifier random_state | 42, unchanged; the experimental seeds identify saved training samples |
| Official test set | Never requested or evaluated |

There are **18 fits**: 2 representations x 3 seeds x 3 C values. Each is a new
classifier, with no warm start. TF-IDF vocabulary/IDF and MiniLM features are held
fixed across C. The classifier learns weights and intercepts through sklearn's
standard multiclass cross-entropy plus L2 penalty. No custom loss or Transformer
training is implemented.

Regularization can behave differently in sparse vocabulary features and dense
pretrained coordinates, even with row normalization. Their dimensions,
correlations, and information content differ. This controlled probe measures
responses under these specific settings, not an optimized global comparison.

## Exact reference and reuse

The pinned source is `results/repeated/20261019T225846_619676Z`, configured in
`REFERENCE_RUN`. The script reads its metadata, sample manifest, individual
results, and embedding-cache records. It verifies completion, all 12 reference
cells, settings/revisions, dependency versions, dataset fingerprint, row balance,
row labels, and separation before using the six 400-example baselines.

No sampling function is called. The original manifest is copied verbatim into
the new run as `source_sample_manifest.json`; the old 100-row entries remain in
that provenance copy but are not used as training budgets in this sweep.

The four required MiniLM blocks already exist: validation plus one 400-row block
per seed. Their ordered text/row identities, metadata, dimensions, finite values,
and checksums are checked with the established pilot cache helper. All three C
values reuse those exact arrays. **No MiniLM model load, download, warmup, CUDA
operation, or re-encoding is needed by this script.** Existing embeddings were
computed on CUDA in the prior experiment; LR fitting remains on CPU.

The script intentionally fails on missing/invalid embeddings instead of silently
recomputing them. Restore missing caches before rerunning; the prior repeated
script can recreate its deterministic blocks while writing a new result folder.
Do not loosen baseline tolerances to conceal mismatched inputs or settings.

The prior experiment did not save TF-IDF feature matrices/vectorizers. TF-IDF is
therefore refit **once per seed**, with unchanged training-only settings, and its
training/validation matrices are reused at all three C values. The short
`HF_DATASETS_CACHE` default remains `D:\hf_cache\datasets`.

## C=1 reproduction gate

All six C=1 fits run first. Their accuracy and Macro-F1 are compared with the
saved repeated-baseline values using **absolute tolerance 1e-10, relative
tolerance 0**. This is floating-point tolerance, not permission to change a
prediction: one different label out of 800 is much larger than the tolerance.
Training times are remeasured, not required to match historical timings.

`baseline_reproduction.csv` and `.json` show old/current scores, differences,
tolerance, and match flags. Any mismatch stops the sweep before C=0.1 or 10,
marks the run failed, and retains the discrepancy report. Review settings,
features, package versions, and samples before continuing. A successful gate
verifies metric reproduction; it does not assert bit-identical fitted weights.

## Outputs

Each run creates a fresh `results/regularization/<UTC timestamp>/` directory.
Previous experiment scripts, caches, and result folders are not modified.

- `individual_results.csv` / `.json`: 18 rows with seed, representation, C,
  validation accuracy/Macro-F1, LR fit time, iterations, and feature dimension.
- `summary.csv` / `.json`: six representation/C groups; means and sample SDs
  (`ddof=1`) for both scores and LR fit times, plus mean/sample SD of paired score
  differences relative to the same seed/representation's C=1 result.
- `paired_differences.csv` / `.json`: all individual score changes relative to C=1.
- `accuracy_vs_C.png`, `macro_f1_vs_C.png`: log-C plots showing means +/- sample
  SD and thin individual-seed lines. SD bars are not confidence intervals.
- `results_table.md`: compact six-row mean +/- SD table, including paired changes.
- `analysis.md`: data-derived responses to the research questions, including
  improvements/declines/ties, differing responses, per-seed consistency, and
  MiniLM-versus-TF-IDF comparisons at matched C values.
- `baseline_reproduction.csv` / `.json`: explicit C=1 checks.
- `predictions_<representation>_seed<seed>_C<value>.csv`: ordered validation row
  IDs and true/predicted labels for each fit (for example C0p1 means C=0.1).
- `source_sample_manifest.json`, `run_metadata.json`: exact source samples,
  reference file hashes, status, settings, versions, and cache/setup provenance.

Require `EXPLORATORY REGULARIZATION SWEEP COMPLETE` and metadata
`status: complete`. Partial files from failed runs are not a completed study.
Aggregation rejects missing, duplicate, non-finite, or out-of-range results.

Scores in CSV/JSON use 0..1; table scores use percentages and score differences
use percentage points. LR training time measures only `classifier.fit`, excluding
TF-IDF preparation, embedding reads, prediction, and output writing. Preparation
and cache-read provenance is recorded separately; historical encoding time is not
counted as current computation. Runtime means/SDs are descriptive and affected by
hardware load and fixed execution order.

## Interpretation and final-study planning

The C sweep is **EXPLORATORY** because the fixed validation set has already been
inspected during earlier rounds. Choosing the best observed C on this set can
produce selection bias. The highest validation score is not an unbiased final
estimate, a significance result, or evidence that the model is fully optimized.
The analysis explicitly reports losses and ties and may favor different C values
for different representations/metrics. No direction of improvement is assumed.

Three training seeds provide only limited descriptive variability. The models
share validation rows and training subsets can overlap, so these are not three
independent final test evaluations. No statistical significance test is performed.

For the final project, predefine the selection metric, budgets, and candidate
settings. Select C separately for each representation using inner validation or
stratified cross-validation within training data. Fit TF-IDF inside each inner
training fold. Once choices are fixed, refit using the allowed training data and
evaluate on the untouched official test set. Keep tuning decisions separate from
that final evaluation. Consider more seeds/budgets and error/calibration analysis.

## Offline verification

`tests/test_regularization_experiment.py` uses synthetic data, temporary files,
and the existing helpers. It checks configuration isolation, exact sample/cache
reuse, mean/sample SD, paired changes, invalid aggregate rejection, output files,
and stopping before exploration if C=1 fails reproduction. It forbids calls to
sampling/model-loading/CUDA functions during the sweep check. These checks are
software verification, not new AG News results.

From the project directory:

```powershell
& 'D:\Anaconda\envs\icsi436-p\python.exe' -B -m unittest discover -s tests -v
```

No presentation or one-page report is generated. Review the actual sweep outputs
before deciding what belongs in the midterm materials.
