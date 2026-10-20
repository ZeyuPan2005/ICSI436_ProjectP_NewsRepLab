# Experiment log

This log summarizes completed artifacts; it does not generate new measurements.
All scores below are validation scores. Original run folders remain unchanged.

## Feasibility — completed

**Purpose:** verify dataset access, actual RTX 4060 CUDA computation, and frozen
MiniLM extraction. AG News loaded with 120,000 training and 7,600 test examples;
five selected training articles produced finite 384-dimensional embeddings.
No supervised metrics were measured. Test split counts were inspected, but its
articles were not used for supervised training, selection, or evaluation.
The short external Datasets cache fixed Windows WinError 206.

## Round 1 — pilot

**Purpose:** establish the complete TF-IDF/LR and frozen-MiniLM/LR workflows.
**Settings:** seed 42, balanced nested 100/400 training rows, fixed disjoint 800
validation rows, common C=1. Four configurations, no custom loss or fine-tuning.
**Artifacts:** [pilot run](../results/pilot/20261019T223707_829532Z/metrics.csv),
plus metadata, exact sample manifest, per-class diagnostics, and comparison figure.
**Observation:** the two pipelines completed successfully; the seed-42 results
are included in the later stability comparison.
**Limits:** one training seed, two budgets, one validation set, untuned settings.
**Protocol:** [pilot guide](pilot_experiment.md).

## Round 2 — repeated baselines

**Purpose:** investigate variability from labeled training samples.
**Settings:** seeds 42, 123, 2026; unchanged models, preprocessing, C=1, budgets,
and ordered validation rows. Seed 42 reuses the original pilot sampling plan.
Twelve classifiers were fitted; these include the pilot settings, not twelve
independent new validation splits.
**Artifacts:** [run summary](../results/repeated/20261019T225846_619676Z/summary.csv),
[individual scores](../results/repeated/20261019T225846_619676Z/individual_results.json),
manifest, metadata, paired differences, per-seed diagnostics, and two figures.

| Features | Labels | Mean validation accuracy ± SD (%) | Mean validation macro-F1 ± SD (%) |
| --- | ---: | ---: | ---: |
| TF-IDF | 100 | 66.21 ± 2.50 | 65.72 ± 2.64 |
| MiniLM | 100 | 82.38 ± 2.18 | 82.28 ± 2.24 |
| TF-IDF | 400 | 76.71 ± 0.51 | 76.63 ± 0.55 |
| MiniLM | 400 | 84.54 ± 0.47 | 84.53 ± 0.46 |

**Observations:** MiniLM wins all six matched seed/budget comparisons on both
metrics. Mean accuracy gains from 100 to 400 are +10.50 points for TF-IDF and
+2.17 for MiniLM. MiniLM's gain varies across seeds; two budgets do not demonstrate
saturation. SD uses the sample denominator (`ddof=1`).
**Limits:** shared validation, correlated/nested subsets, only three seeds, no
significance test or fully optimized model selection.
**Protocol:** [repeated guide](repeated_experiments.md).

## Round 3 — exploratory regularization sensitivity

**Purpose:** examine representation-dependent responses to C, not select a final
optimized model. **Settings:** 400 training rows per saved seed; C=0.1, 1, 10;
all feature and validation choices fixed. Eighteen fits include six C=1 baseline
reproductions. All 12 baseline metric comparisons match exactly.
**Artifacts:** [summary](../results/regularization/20261019T232052_926893Z/summary.csv),
[reproduction checks](../results/regularization/20261019T232052_926893Z/baseline_reproduction.csv),
individual scores, paired differences, metadata, source manifest, analysis, and figures.

| Features | Mean validation accuracy at C=0.1 / 1 / 10 (%) |
| --- | --- |
| TF-IDF | 76.00 / 76.71 / 77.46 |
| MiniLM | 84.38 / 84.54 / 83.75 |

**Observation:** C=1 to C=10 changes mean accuracy by +0.75 points for TF-IDF and
−0.79 for MiniLM. **Interpretation:** representation-specific hyperparameter
selection may help. **Limits:** the validation set has already informed decisions;
the best observed C is not an unbiased final score. C=1 is overlapping evidence.
**Protocol:** [regularization guide](regularization_experiment.md).

## Reviewed midterm deliverables and next phase

[Midterm materials](../reports/midterm/README.md) contain the five editable slides,
one-page report/source, and 510-word script for Group 8. Numerical source data are
recorded in [verified_results.json](../reports/midterm/verified_results.json).

Future work remains distinct from completed work: expanded budgets; stratified
K-fold selection within each permitted labeled budget, fitting TF-IDF inside
folds; systematic representation comparisons; Business/Sci-Tech error analysis;
controlled cost analysis; and official-test evaluation after freezing the protocol.
No new experiments were run for codebase consolidation.
