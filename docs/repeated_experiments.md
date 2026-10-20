# Repeated baselines: three training-subset seeds

Run `src/repeated_experiment.py` with F5 in Spyder using the existing `icsi436-p`
interpreter. The real three-seed experiments are intended for manual execution;
their analysis and figures are generated from measured results, not predictions
about which model should win. CUDA is required, as in the pilot.

## What stays fixed

- Dataset/revision: the pilot's AG News original training split. No test access.
- Model/revision: frozen all-MiniLM-L6-v2, normalized float32 embeddings, batch 32.
- TF-IDF: all pilot settings, fitted separately on each training subset only.
- Classifier: multinomial Logistic Regression, lbfgs, L2, C=1.0, max_iter=2000,
  tol=1e-4, intercept enabled. No hyperparameter search.
- Validation: the **exact ordered 800 rows from the completed seed-42 pilot**,
  200 per class, identical for every experiment.
- Training sizes: 100 (25/class) and 400 (100/class), nested within each seed.

`REFERENCE_PILOT_RUN` explicitly names `results/pilot/20261019T223707_829532Z`.
The script checks its completion status, dataset fingerprint/revision, model
revision, classifier/vectorizer settings, row counts, class balance, labels, and
disjointness. If the reference is absent or incompatible, it fails visibly;
it never silently draws a different validation set. Keep the reference folder.

## What the seeds change

Seed 42 uses the reference manifest's exact training rows and ordering. Seeds
123 and 2026 draw new balanced training subsets after removing all fixed validation
rows. Training sets may overlap **across seeds**, which is allowed; no training
set overlaps validation. The larger set includes every row in its smaller set.
Both representations use identical ordered samples within each seed.

Here, "seed" means the training-sampling seed. The pilot classifier's
`random_state=42` setting is retained for all fits; lbfgs does not use it. Frozen
MiniLM remains in evaluation mode with the original setup seed. We are measuring
variation from the labeled examples, not intentionally randomizing model weights.
Exact floating-point reproducibility across machines/software versions is not
guaranteed. Row IDs, labels, package versions, settings, and code hashes are saved.

## Preservation and caching

The original pilot/feasibility scripts and previous results are unchanged. The
new script imports the existing pilot's helpers without executing its main().
It writes each run to a fresh `results/repeated/<UTC timestamp>/` folder.

The cache identity matches the pilot, so compatible validation and seed-42
embeddings are reused. Seeds 123/2026 get their own content/configuration-checked
blocks. A second repeated run can reuse all blocks. Hashes, shapes, and finite
values are checked. No existing scores are copied into the aggregate: all twelve
classifiers are fitted/evaluated in the new run, avoiding double-counting seed 42.

The same short `HF_DATASETS_CACHE` default (`D:\hf_cache\datasets`) and existing
model/embedding cache directories are used. No packages or CUDA settings change.
To refresh selected embedding blocks deliberately, change the new script's
`FORCE_RECOMPUTE_EMBEDDINGS` constant to True; normally leave it False. Do not run
concurrent scripts against the same embedding cache.

Only selected texts are processed; the label column is scanned for sampling.
At most 2,000 unique articles participate (800 validation plus up to 3 x 400
training rows), fewer if training sets overlap. Smaller blocks are encoded
separately for genuine measured timings, not prorated estimates.

## Outputs after all experiments finish

| File | Contents |
| --- | --- |
| individual_results.csv / .json | 12 rows: 3 seeds x 2 representations x 2 sizes; scores, timings, errors, cache flags |
| summary.csv / .json | Four representation/size groups; mean, sample SD, minimum and maximum for accuracy/Macro-F1 |
| paired_differences.csv | MiniLM minus TF-IDF per seed/size and 400 minus 100 per seed/representation |
| mean_std_comparison.png | Two metric panels, means with +/- sample SD bars |
| individual_seed_comparison.png | Six panels showing both representations for each seed/metric |
| analysis.md | Concise answers to the five requested research questions, calculated from this run |
| sample_manifest.json | Exact common validation and per-seed training row IDs/labels |
| run_metadata.json | Status, provenance, reference manifest hash, configuration, and cache/timing details |
| seed_42/, seed_123/, seed_2026/ | Four individual metrics plus prediction/error CSVs and class/confusion diagnostics per seed |

Require `REPEATED BASELINES COMPLETE` and metadata `status: complete`. Failed runs
retain diagnostic/partial files but are not valid completed comparisons. Aggregate
statistics require all 12 unique experiments and reject missing/duplicate rows.

## How to read the statistics and analysis

CSV/JSON accuracy and Macro-F1 use a 0..1 scale. Their mean summarizes the three
training subsets; sample SD uses `ddof=1`, dividing squared deviations by n-1.
The analysis converts scores to percentages and changes/spread to percentage
points. Error bars show SD, **not confidence intervals or standard error**.

The generated analysis reports wins/ties/losses separately for each training size
and metric, and signed 100-to-400 changes for each seed. It does not assume that
the pilot's patterns persist. "Limited improvement" has an explicit descriptive
definition: nonnegative gain of at most **1 percentage point** (configurable via
`LIMITED_GAIN_THRESHOLD=0.01`). Losses and larger gains are reported separately.
This convention is not a statistical equivalence test and does not prove a plateau.

Holding validation fixed isolates training-sample variation but leaves uncertainty
about new validation/test data. Three seeds are too few for strong statistical
claims; nested training sizes, possible cross-seed overlap, and shared validation
make results paired/correlated. C=1.0 remains untuned; performance under other
settings, larger training sets, and untouched test data remains uncertain.

Feature timings retain the pilot's definitions: cached measurements are marked as
reused, with actual current compute and cache I/O separately recorded. Validation
encoding is shared across all six MiniLM fits. Its historical extraction time is
included in each standalone cost but its current compute is counted only once.
Downloads, dataset/model loading, and warmup are recorded separately. Do not turn
cache-hit times or a three-seed score SD into computational-performance claims.

## Verification

Offline tests use synthetic text and a fake encoder, exercising the real sklearn
classifiers. They check fixed validation, nested/balanced samples, reproducibility,
correct sample SD, paired changes, rejection of partial aggregates, analysis of
counterexamples, compatibility checks, reuse of the pilot cache, and preservation
of earlier files. Their temporary outputs are not research results.

From the project directory, use the existing interpreter:

```powershell
& 'D:\Anaconda\envs\icsi436-p\python.exe' -B -m unittest discover -s tests -v
```

After running the real experiments, review `analysis.md` together with individual
rows and error cases. No slides or one-page report are generated by this script.
