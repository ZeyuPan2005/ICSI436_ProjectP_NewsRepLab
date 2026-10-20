# Phase 2 pilot: protocol and interpretation

This is a preliminary, single-seed comparison of representations. Actual pilot
scores are generated only when `src/pilot_experiment.py` is run successfully.
No accuracy or Macro-F1 values are assumed in this document.

## Fixed design

| Item | Setting |
| --- | --- |
| Source | AG News original training split only |
| Seed | 42, NumPy default_rng |
| Validation | 800 rows, 200 per class, reserved before training selection |
| Small training set | 100 rows, 25 per class |
| Larger training set | 400 rows, 100 per class, containing the small set |
| Pipeline A | Training-only TF-IDF + multinomial Logistic Regression |
| Pipeline B | Frozen MiniLM, CUDA inference + multinomial Logistic Regression |
| Common classifier | lbfgs, L2, C=1.0, fit_intercept=True, max_iter=2000, tol=1e-4 |
| Validation metrics | Accuracy, Macro-F1, class-level report, confusion matrix, errors |
| Official test split | Not requested, tuned on, or evaluated |

Both pipelines receive the same ordered training and validation rows. The
`sample_manifest.json` records every selected original row ID and its label.
Sampling reads the 120,000 integer labels; it does not tokenize all articles.
Only 1,200 unique news texts are used. The small training block is encoded
separately to measure its cost, so a fresh run encodes 1,300 article instances
(100 + 400 + 800), plus one fixed warmup sentence. No full-dataset inference runs.

The dataset and model revisions are pinned to the revisions already cached by
the successful feasibility test. Run metadata also records a dataset fingerprint,
package versions, code hash, classifier settings, device, and sampling seed.

## What each component learns

TF-IDF learns a vocabulary and inverse document frequencies using only the
current training set. It uses lowercased word unigrams/bigrams, at most 20,000
features, minimum document frequency 1, logarithmic term frequency, and L2 row
normalization. Other vectorizer defaults follow the recorded sklearn version.
Validation articles are transformed with these fixed statistics. A fresh
vectorizer is fit at each training size.

MiniLM maps each article to 384 floating-point values using existing pretrained
weights. The model is in evaluation mode, its parameters have gradients disabled,
and extraction uses PyTorch inference mode. No Transformer optimizer or update
exists. Embeddings are unit-normalized; batch size is 32, precision is float32.
The tokenizer truncates long articles at the model's maximum sequence length,
which is recorded in run metadata. Labels never enter the encoder.

For either feature matrix, Logistic Regression learns four coefficient vectors
and four intercepts. The input dimension differs, so the number of weights also
differs. Linear scores become softmax probabilities; fitting minimizes standard
multiclass cross-entropy with L2 regularization. sklearn implements this objective
and optimization. No custom loss is used. In the installed sklearn 1.9.1,
`l1_ratio=0.0` specifies L2, and lbfgs uses multinomial loss for four classes.

This common C=1.0 is an untuned starting point. Different feature distributions
and dimensions may require different validation-selected C values in the final
study. A difference in this pilot is not a pure, universal measure of semantic
quality independent of normalization, vocabulary, truncation, and regularization.

## Run in Spyder

1. Confirm `sys.executable` is `D:\Anaconda\envs\icsi436-p\python.exe`.
2. Inspect and open `src/pilot_experiment.py`, then press F5.
3. CUDA is required for this pilot. If unavailable, it fails with a diagnostic
   rather than silently changing the intended feature-extraction device.
4. Look for `PILOT COMPLETE`, then open the printed output folder. Check
   `run_metadata.json` has `"status": "complete"` before presenting the comparison.

The dataset cache uses `D:\hf_cache\datasets` unless `HF_DATASETS_CACHE` specifies
another short absolute path. This preserves the WinError 206 fix. Model files
reuse the existing `.cache/sentence_transformers/` directory. Embedding caches
live under `.cache/pilot_embeddings/`. Directories are created automatically.
Nothing is installed or changed in the Conda environment.

For freshly measured embedding costs, edit `FORCE_RECOMPUTE_EMBEDDINGS = True`
near the top of the script, then run it. Set it back to False for routine reuse.
Each run gets its own UTC-timestamped folder, preserving earlier results.

## Generated files

All outputs are under `results/pilot/<UTC run timestamp>/`:

- `metrics.csv` and `metrics.json`: four experiment rows, with scores, timings,
  error counts, feature dimensions, iteration counts, and cache flags.
- `sample_manifest.json`: exact training/validation rows and labels.
- `run_metadata.json`: completion status, setup timings, revisions, package and
  hardware details, configuration, and per-block timing provenance.
- `performance_comparison.png`: two panels for accuracy and Macro-F1 at 100/400
  training examples, explicitly labeled preliminary/single-seed.
- `<tfidf|minilm>_<100|400>_predictions.csv`: all 800 validation predictions,
  true/predicted class, predicted probability, source row ID, and short preview.
- `<tfidf|minilm>_<100|400>_errors.csv`: only incorrect predictions; an empty
  error set still produces a CSV header.
- `<tfidf|minilm>_<100|400>_diagnostics.json`: per-class precision/recall/F1,
  confusion matrix (true rows, predicted columns), and learned parameter shapes.

Prediction probabilities are not calibrated confidence guarantees. Error CSVs
support qualitative inspection; a few interesting examples are not a statistical
error analysis. Large outputs/caches remain ignored by the existing `.gitignore`.
No model pickle, PowerPoint, or LaTeX report is generated.

## Reading the timing columns correctly

| Field | Meaning |
| --- | --- |
| train_feature_seconds | TF-IDF fit_transform or measured MiniLM training-block extraction |
| validation_feature_seconds | TF-IDF transform or measured MiniLM validation-block extraction |
| feature_extraction_seconds | Sum of these two measured costs for a standalone comparison |
| feature_timing_origin | measured_now or includes_reused_measurement |
| feature_compute_this_run_seconds | Actual feature computation attributed to this experiment in this run |
| feature_cache_read_seconds | Reading/checking cached embedding arrays, kept separate from encoding |
| feature_cache_write_seconds | Writing new embedding cache files, also separate |
| lr_training_seconds | CPU classifier fit only, after features are available |
| prediction_seconds | CPU predict_proba and class selection; excludes feature extraction |

The MiniLM validation block is independent of the training size and computed once.
Its measured cost is included for both standalone comparisons, but only counted
once in current-run computation. On a cache hit, the original measured extraction
seconds and timestamp are retained; they are not a new timing observation.
`validation_cache_hit=True` also includes reuse of validation features in memory.
Do not sum standalone extraction costs to claim total wall-clock runtime.

Model loading (including possible downloads), dataset loading (including possible
downloads/preparation), CUDA warmup, and module import time are recorded separately.
Network time is not isolated from the associated load operation. Feature timing
excludes these startup costs; GPU synchronization brackets actual encoding.
Encoding includes CPU tokenization and conversion back to CPU NumPy arrays.
TF-IDF and sklearn LR are CPU operations; MiniLM inference is on the GPU.
These mixed-device, single-run timings are descriptive, not controlled benchmarks.

Cache identities include ordered row IDs/text hashes, dataset fingerprint, model
revision, encoding settings, GPU/CUDA information, and package versions. Shape,
finite values, and checksums are checked before reuse. Corruption is reported;
set FORCE_RECOMPUTE_EMBEDDINGS=True to regenerate selected feature blocks.
Do not run concurrent pilot scripts writing to the same embedding cache.

## Verification and limitations

The offline unittest suite checks balanced/disjoint/nested sampling, repeatability,
training-only vocabulary fitting, cache reuse/invalidation, corrupt array rejection,
and the four-experiment export flow with real sklearn classifiers on synthetic
features. It uses temporary files and a fake encoder: it is implementation
verification, not evidence about MiniLM or AG News performance.

Run it, if desired, from the project directory with the existing interpreter:

```powershell
& 'D:\Anaconda\envs\icsi436-p\python.exe' -B -m unittest discover -s tests -v
```

The intended real experiment is run manually by the student. One seed and one
validation split cannot quantify sampling variability. There are only two training
sizes, fixed untuned hyperparameters, no held-out test results, and no significance
test. GPU/library differences may affect exact floating-point results and timings
even with a fixed seed. Row sets are disjoint; near-duplicate news stories are not
audited in this first pilot. Pretraining can supply external knowledge; possible
pretraining overlap with news content is not established here.

## What can go into the midterm presentation

Already supported by the student's successful feasibility run: AG News access,
working RTX 4060 inference, and valid 384-dimensional MiniLM features. The pilot
code additionally documents a reproducible comparison protocol, not yet scores.

After a completed real run, include its actual four validation scores, the saved
comparison figure, explicitly qualified timings, and a few real error examples.
Describe any observed advantage as "in this preliminary seed-42 pilot," not proof
of universal superiority or statistical significance. A CPU/GPU speed comparison
must name the devices and exclude startup/cache-read confusion. Report failures
or unexpected results honestly. Future work includes repeated seeds, more sample
sizes, separate validation-based tuning, and eventual untouched-test evaluation.

## API references

- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [SentenceTransformer](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
