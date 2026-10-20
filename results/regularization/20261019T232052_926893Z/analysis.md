# EXPLORATORY regularization sensitivity analysis

All 18 fits use saved samples and fixed features. All six C=1.0 baselines matched the prior repeated run within the recorded tolerance.
The validation set was inspected in earlier rounds: best observed scores may be selection-biased and are not unbiased final estimates.

## 1. Does C change scores relative to C=1? Are changes consistent across seeds?

- tfidf, validation_accuracy, C=0.1: mean change -0.71 pp (sample SD 0.14 pp); 0/3 improvements, 0/3 ties, 3/3 declines. Seeds 42: -0.62 pp, 123: -0.62 pp, 2026: -0.87 pp.
- tfidf, validation_accuracy, C=10: mean change +0.75 pp (sample SD 0.22 pp); 3/3 improvements, 0/3 ties, 0/3 declines. Seeds 42: +0.63 pp, 123: +1.00 pp, 2026: +0.63 pp.
- tfidf, validation_macro_f1, C=0.1: mean change -0.76 pp (sample SD 0.16 pp); 0/3 improvements, 0/3 ties, 3/3 declines. Seeds 42: -0.65 pp, 123: -0.69 pp, 2026: -0.94 pp.
- tfidf, validation_macro_f1, C=10: mean change +0.76 pp (sample SD 0.23 pp); 3/3 improvements, 0/3 ties, 0/3 declines. Seeds 42: +0.64 pp, 123: +1.02 pp, 2026: +0.62 pp.
- minilm, validation_accuracy, C=0.1: mean change -0.17 pp (sample SD 0.76 pp); 1/3 improvements, 1/3 ties, 1/3 declines. Seeds 42: +0.50 pp, 123: +0.00 pp, 2026: -1.00 pp.
- minilm, validation_accuracy, C=10: mean change -0.79 pp (sample SD 0.07 pp); 0/3 improvements, 0/3 ties, 3/3 declines. Seeds 42: -0.75 pp, 123: -0.88 pp, 2026: -0.75 pp.
- minilm, validation_macro_f1, C=0.1: mean change -0.23 pp (sample SD 0.77 pp); 1/3 improvements, 0/3 ties, 2/3 declines. Seeds 42: +0.45 pp, 123: -0.06 pp, 2026: -1.06 pp.
- minilm, validation_macro_f1, C=10: mean change -0.77 pp (sample SD 0.04 pp); 0/3 improvements, 0/3 ties, 3/3 declines. Seeds 42: -0.76 pp, 123: -0.82 pp, 2026: -0.74 pp.

## 2. Do the representations respond differently?

- validation_accuracy, C=0.1: TF-IDF mean change -0.71 pp; MiniLM -0.17 pp. MiniLM-minus-TF-IDF change: +0.54 pp.
- validation_accuracy, C=10: TF-IDF mean change +0.75 pp; MiniLM -0.79 pp. MiniLM-minus-TF-IDF change: -1.54 pp.
- tfidf, validation_accuracy: best observed mean 77.46% at C=[10.0]. Descriptive validation ranking only, not final optimization.
- minilm, validation_accuracy: best observed mean 84.54% at C=[1.0]. Descriptive validation ranking only, not final optimization.
- validation_macro_f1, C=0.1: TF-IDF mean change -0.76 pp; MiniLM -0.23 pp. MiniLM-minus-TF-IDF change: +0.53 pp.
- validation_macro_f1, C=10: TF-IDF mean change +0.76 pp; MiniLM -0.77 pp. MiniLM-minus-TF-IDF change: -1.53 pp.
- tfidf, validation_macro_f1: best observed mean 77.39% at C=[10.0]. Descriptive validation ranking only, not final optimization.
- minilm, validation_macro_f1: best observed mean 84.53% at C=[1.0]. Descriptive validation ranking only, not final optimization.

## 3. Does MiniLM still outperform TF-IDF across tested C values?

- validation_accuracy, C=0.1: 3/3 MiniLM wins, 0/3 ties, 0/3 losses; mean MiniLM advantage +8.38 pp. Advantage holds in all three seeds at this C.
- validation_accuracy, C=1: 3/3 MiniLM wins, 0/3 ties, 0/3 losses; mean MiniLM advantage +7.83 pp. Advantage holds in all three seeds at this C.
- validation_accuracy, C=10: 3/3 MiniLM wins, 0/3 ties, 0/3 losses; mean MiniLM advantage +6.29 pp. Advantage holds in all three seeds at this C.
- validation_macro_f1, C=0.1: 3/3 MiniLM wins, 0/3 ties, 0/3 losses; mean MiniLM advantage +8.43 pp. Advantage holds in all three seeds at this C.
- validation_macro_f1, C=1: 3/3 MiniLM wins, 0/3 ties, 0/3 losses; mean MiniLM advantage +7.90 pp. Advantage holds in all three seeds at this C.
- validation_macro_f1, C=10: 3/3 MiniLM wins, 0/3 ties, 0/3 losses; mean MiniLM advantage +6.37 pp. Advantage holds in all three seeds at this C.

## 4. Limits and final-project planning

- This is an exploratory three-value sensitivity probe. Larger C is not inherently better, and no model is declared fully optimized.
- Three seeds and a shared, previously inspected validation set do not support significance claims. SD is descriptive, not a confidence interval.
- For the final project, predefine a selection metric and use inner validation or stratified cross-validation within training data to choose C separately for each representation. Fit TF-IDF inside each inner training fold to prevent leakage.
- Freeze the chosen protocol before a final evaluation on the untouched official AG News test split. The best score on this explored validation set must not be presented as unbiased final performance.
- Investigate additional labeled-data budgets and seeds, representation-specific C ranges justified by this probe, calibration/error patterns, and variability across inner splits. Keep budgets and preprocessing comparable.
- LR fit times exclude fixed-feature preparation and cache reads. Single-process timing noise and fixed experiment order limit runtime conclusions.
