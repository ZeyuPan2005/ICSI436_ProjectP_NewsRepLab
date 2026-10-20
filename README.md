# NewsRepLab

**Evaluating Pretrained Text Representations for News Classification under Limited Training Data**

UAlbany ICSI436/536, Project P — **Group 8: Zeyu Pan, Jiayu Liao, Yuhan Sun**.

**Research question:** With the same limited task labels, do frozen pretrained
MiniLM representations improve AG News classification and sample efficiency
compared with traditional TF-IDF features?

## Current progress

Feasibility checks, the pilot, three-seed baselines, and an exploratory
regularization sweep are complete. The reviewed five-slide presentation and
one-page report are in [reports/midterm](reports/midterm/README.md). All reported
scores are **validation results**. Official test data have not been used for
training, tuning, or evaluation. The feasibility check loaded split metadata and
counts, including the test split; supervised experiments request only `train`.

| Completed round | Design | Configurations |
| --- | --- | ---: |
| 1 — Pilot | Seed 42; 100 and 400 labels; common C=1 | 4 |
| 2 — Repeated baselines | Seeds 42, 123, 2026; both budgets; C=1 | 12 |
| 3 — Exploratory regularization | Same three seeds; 400 labels; C=0.1, 1, 10 | 18 |

Counts overlap intentionally: Round 2 includes the seed-42 pilot settings and
Round 3 reproduces the C=1 baselines. They are not 34 independent experiments.
See the [experiment log](docs/experiment_log.md) for exact saved runs.

## Pipelines and fair comparison

1. **TF-IDF → multinomial Logistic Regression:** fit vocabulary and inverse
   document frequencies on training text only; transform validation with those
   fixed statistics. Features are sparse word unigram/bigram vectors.
2. **Frozen MiniLM → multinomial Logistic Regression:** extract normalized
   384-dimensional vectors with `all-MiniLM-L6-v2`; do not update its parameters.
   Feature extraction uses CUDA; classifier fitting uses the CPU.

Both classifiers learn class weights and intercepts using standard softmax
probabilities, cross-entropy, and L2 regularization. There is no custom loss or
Transformer fine-tuning. The study compares representations, not new algorithms.

The 100/400 training sets are class-balanced and nested within each seed.
Both pipelines use identical ordered examples. One fixed disjoint validation set
contains 800 articles (200 per class). Sampling seeds change training rows only.
The common classifier uses L-BFGS, C=1, `max_iter=2000`, `tol=1e-4`, an intercept,
and `random_state=42`. Round 3 changes only C. The architecture guide explains
[data shapes, cache validation, and module relationships](docs/architecture.md).

## Verified preliminary findings

Round 2 percentages are mean ± sample standard deviation across three training
seeds (`ddof=1`). SD describes training-sample variability, not a confidence interval.

| Representation | Labels | Validation accuracy (%) | Validation macro-F1 (%) |
| --- | ---: | ---: | ---: |
| TF-IDF | 100 | 66.21 ± 2.50 | 65.72 ± 2.64 |
| Frozen MiniLM | 100 | 82.38 ± 2.18 | 82.28 ± 2.24 |
| TF-IDF | 400 | 76.71 ± 0.51 | 76.63 ± 0.55 |
| Frozen MiniLM | 400 | 84.54 ± 0.47 | 84.53 ± 0.46 |

MiniLM exceeds TF-IDF for every matched seed/budget pair on both metrics. Its
100-label mean accuracy also exceeds TF-IDF's 400-label mean. Increasing labels
from 100 to 400 improves mean accuracy by 10.50 percentage points for TF-IDF and
2.17 for MiniLM. This supports promising label efficiency in these settings;
two budgets do not establish a plateau.

Round 3 is a separate **exploratory** sensitivity analysis at 400 labels:

| Representation | C=0.1 | C=1 | C=10 |
| --- | ---: | ---: | ---: |
| TF-IDF mean validation accuracy (%) | 76.00 | 76.71 | 77.46 |
| MiniLM mean validation accuracy (%) | 84.38 | 84.54 | 83.75 |

C=1 exactly reproduces Round 2. Moving to C=10 changes accuracy by +0.75 and
−0.79 points, respectively. Representation-specific hyperparameter selection
may be useful; neither pipeline is fully optimized.

Sources: [Round 2 summary](results/repeated/20261019T225846_619676Z/summary.csv),
[Round 3 summary](results/regularization/20261019T232052_926893Z/summary.csv), and
[verified midterm data](reports/midterm/verified_results.json).

## Environment and setup

The verified workstation uses Windows, Spyder, Conda environment `icsi436-p`,
Python 3.11.17, PyTorch 2.11.0+cu128, and an NVIDIA GeForce RTX 4060 Laptop GPU.
[requirements.txt](requirements.txt) records the seven runtime packages used or
checked by the experiments. It is an inventory, not a complete environment lock.
**Keep the existing working environment; no reinstall is needed.**

On a separate machine, prepare an isolated Python 3.11 environment and a working
CUDA-enabled PyTorch installation compatible with that machine. Install the
recorded requirements there, then verify imports and CUDA before attempting
reproduction. Do not install this inventory over the established environment.
Exact cross-platform scores, cache identity, and timings are not guaranteed.
See [reproducibility and setup](docs/reproducibility.md) for prerequisites and
the important Round 3 cache restriction.

Paths are derived from each script's location, not the working directory.
The Windows dataset cache defaults to the short external path
`D:\hf_cache\datasets` to avoid WinError 206. To use another location, set
`HF_DATASETS_CACHE` to a short absolute path before running a script, for example
in Spyder's console:

```python
import os
os.environ["HF_DATASETS_CACHE"] = r"C:\hf_cache\datasets"
```

This override is also required on machines without the default D: drive; on
non-Windows systems use an absolute path appropriate to that system. Model files
and embeddings live in ignored project `.cache/` directories. The feasibility,
pilot, and repeated scripts create their cache directories when run. Round 3
requires existing validated embedding blocks. No package is installed by any script.

## Execution

In Spyder, select the existing `icsi436-p` interpreter, verify
`import sys; print(sys.executable)`, open a script, and press **F5**. All four
scripts retain their standalone entry points and `if __name__ == "__main__"` guards.

From an activated environment and the project root, the equivalent commands are:

```powershell
conda activate icsi436-p
python src/feasibility_test.py
python src/pilot_experiment.py
python src/repeated_experiment.py
python src/regularization_experiment.py
```

These commands execute real work; run only the stage you intend to repeat.
No new ML experiment was run during consolidation. Each experiment writes a
new timestamped directory, preserving the original results. Do not run scripts
concurrently against the shared embedding cache.

| Script guide | Prerequisites and scope |
| --- | --- |
| [Feasibility/data notes](data/README.md) | Downloads if needed; CUDA probe with explicit CPU fallback; five embeddings only |
| [Pilot](docs/pilot_experiment.md) | CUDA required; four validation configurations |
| [Repeated baselines](docs/repeated_experiments.md) | Saved pilot metadata/manifest; CUDA required; twelve fits |
| [Regularization](docs/regularization_experiment.md) | Saved repeated reference plus its four matching embedding blocks; cache-only MiniLM; eighteen CPU fits |

The repeated and regularization scripts pin named historical run directories.
They reject incompatible references instead of silently choosing the newest run.
Restoring these small reference artifacts is necessary on a fresh checkout.

## Offline verification

```powershell
python -B -m unittest discover -s tests -v
```

Tests use synthetic text, a fake encoder, temporary outputs, and real CPU
Logistic Regression. They do not download AG News or run CUDA inference.
Formatting tools are optional development tools, not required to run experiments.
See [maintenance verification](docs/maintenance_review.md) for the executed checks
and limits of the behavioral-equivalence claim.

## Project map

- `src/`: four readable experimental entry points with existing shared helpers.
- `tests/`: offline sampling, leakage, cache, aggregation, and workflow checks.
- `docs/`: architecture, experiment log, protocols, reproducibility, course notes,
  maintenance review, and publication preparation.
- `results/`: selected small historical summaries, manifests, metadata, diagnostics,
  and figures; raw prediction/error CSVs remain local and ignored.
- `reports/midterm/`: [editable slides](reports/midterm/NewsRepLab_Midterm.pptx),
  [one-page PDF](reports/midterm/midterm_report.pdf),
  [LaTeX source](reports/midterm/midterm_report.tex), and
  [510-word speaking script](reports/midterm/presentation_script.md).
- `data/README.md`: data attribution and cache policy; downloaded data are ignored.

## Limits and future work

Three training seeds share one repeatedly inspected validation set. Results are
paired/correlated, not independent validation replications. Pretraining supplies
external information; this is task-label efficiency, not equal total training
effort. Timing mixes CPU/GPU work and reused caches, so it is not a controlled
speed benchmark. There is no significance claim, final test result, universal
superiority claim, or completed hyperparameter optimization.

Planned work: expand labeled budgets; select hyperparameters separately using
stratified K-fold CV **within each labeled budget**, fitting TF-IDF inside each
fold; evaluate representations more systematically; analyze Business versus
Sci/Tech errors; measure costs under controlled conditions; then fix the protocol
and evaluate on the untouched official test set. K-fold selection is not yet implemented.

## Attribution and publication

- [AG News dataset card](https://huggingface.co/datasets/fancyzhx/ag_news): corpus
  collected by Antonio Gulli; classification benchmark from Zhang, Zhao, and
  LeCun (2015), [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626).
- [Sentence Transformers all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- [Course requirement notes](docs/course_requirements.md); the instructor's
  original PDF is provided separately, not bundled in this repository.
- Code and documentation were developed with Codex assistance. Group 8 is
  responsible for final review and the course's authorship/attribution requirements.

GitHub publication is pending confirmation of account/organization, visibility,
and the exact tracked file list. See [Git preparation](docs/git_preparation.md).
No remote repository has been created and nothing has been pushed.
