# EXPLORATORY regularization sensitivity

400 training examples; fixed validation n=800; seeds 42/123/2026. Mean +/- sample SD (ddof=1).

| Representation | C | Accuracy (%) | Macro-F1 (%) | LR fit (seconds) | Accuracy delta vs C=1 (pp) | F1 delta vs C=1 (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 0.1 | 76.00 +/- 0.62 | 75.88 +/- 0.68 | 0.0150 +/- 0.0003 | -0.71 | -0.76 |
| tfidf | 1 | 76.71 +/- 0.51 | 76.63 +/- 0.55 | 0.0378 +/- 0.0011 | +0.00 | +0.00 |
| tfidf | 10 | 77.46 +/- 0.52 | 77.39 +/- 0.54 | 0.1067 +/- 0.0107 | +0.75 | +0.76 |
| minilm | 0.1 | 84.38 +/- 0.45 | 84.31 +/- 0.42 | 0.0050 +/- 0.0002 | -0.17 | -0.23 |
| minilm | 1 | 84.54 +/- 0.47 | 84.53 +/- 0.46 | 0.0083 +/- 0.0004 | +0.00 | +0.00 |
| minilm | 10 | 83.75 +/- 0.45 | 83.76 +/- 0.46 | 0.0143 +/- 0.0019 | -0.79 | -0.77 |
