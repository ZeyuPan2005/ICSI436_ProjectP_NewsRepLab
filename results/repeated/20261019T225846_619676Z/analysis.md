# Repeated baseline analysis

Preliminary results for seeds 42, 123, and 2026 on the same 800 validation rows.
All scores below come from this completed run; the official test split was not evaluated.

## 1. Does MiniLM consistently outperform TF-IDF?

- validation_accuracy, n=100: yes, across these three seeds; 3/3 wins, 0/3 ties, 0/3 losses. MiniLM minus TF-IDF: seed 42: +15.25 pp, seed 123: +17.00 pp, seed 2026: +16.25 pp.
- validation_accuracy, n=400: yes, across these three seeds; 3/3 wins, 0/3 ties, 0/3 losses. MiniLM minus TF-IDF: seed 42: +6.75 pp, seed 123: +8.13 pp, seed 2026: +8.63 pp.
- validation_macro_f1, n=100: yes, across these three seeds; 3/3 wins, 0/3 ties, 0/3 losses. MiniLM minus TF-IDF: seed 42: +15.36 pp, seed 123: +17.82 pp, seed 2026: +16.49 pp.
- validation_macro_f1, n=400: yes, across these three seeds; 3/3 wins, 0/3 ties, 0/3 losses. MiniLM minus TF-IDF: seed 42: +6.76 pp, seed 123: +8.24 pp, seed 2026: +8.69 pp.

## 2. Does TF-IDF consistently improve from 100 to 400?

- validation_accuracy: yes; strictly improves in 3/3 seeds. seed 42: +8.62 pp, seed 123: +10.25 pp, seed 2026: +12.62 pp.
- validation_macro_f1: yes; strictly improves in 3/3 seeds. seed 42: +8.78 pp, seed 123: +11.00 pp, seed 2026: +12.97 pp.

## 3. Is MiniLM's improvement from 100 to 400 limited?

Descriptive convention: a nonnegative gain of at most 1.00 percentage points is called limited. This threshold is not a significance or equivalence test; decreases are reported separately.
- validation_accuracy: limited nonnegative gains in 1/3 seeds, decreases in 0/3, larger gains in 2/3; mean change +2.17 pp. seed 42: +0.12 pp, seed 123: +1.38 pp, seed 2026: +5.00 pp.
- validation_macro_f1: limited nonnegative gains in 1/3 seeds, decreases in 0/3, larger gains in 2/3; mean change +2.26 pp. seed 42: +0.18 pp, seed 123: +1.41 pp, seed 2026: +5.18 pp.

## 4. How much do scores vary across seeds?

Mean scores are percentages; SD and ranges are in percentage points. SD is sample SD (ddof=1).

| Representation | Training n | Metric | Mean (%) | SD (pp) | Min-max (%) |
| --- | ---: | --- | ---: | ---: | ---: |
| tfidf | 100 | validation_accuracy | 66.21 | 2.50 | 63.62-68.62 |
| tfidf | 100 | validation_macro_f1 | 65.72 | 2.64 | 63.22-68.47 |
| tfidf | 400 | validation_accuracy | 76.71 | 0.51 | 76.25-77.25 |
| tfidf | 400 | validation_macro_f1 | 76.63 | 0.55 | 76.20-77.25 |
| minilm | 100 | validation_accuracy | 82.38 | 2.18 | 79.88-83.88 |
| minilm | 100 | validation_macro_f1 | 82.28 | 2.24 | 79.71-83.83 |
| minilm | 400 | validation_accuracy | 84.54 | 0.47 | 84.00-84.88 |
| minilm | 400 | validation_macro_f1 | 84.53 | 0.46 | 84.01-84.89 |

## 5. What remains uncertain?

- Three seeds provide only a small descriptive sample. SD estimates themselves are unstable; no significance claim is justified.
- Validation is fixed, so these results do not quantify variability across validation splits or guarantee performance on unseen test data.
- Training subsets may overlap across seeds, and sizes are nested. Results are paired/correlated, not independent new validation datasets.
- C=1.0 and all preprocessing settings remain untuned. This check does not establish each representation's best attainable performance.
- Two training sizes cannot establish a plateau or a general sample-efficiency curve. Small gains may reflect this validation set or settings.
- Repeated inspection of the same validation set can inform later choices; keep final test evaluation untouched until the protocol is fixed.
- Cached feature times may be historical measurements; cache reads are not new encoding benchmarks. Hardware and numerical effects remain possible.
