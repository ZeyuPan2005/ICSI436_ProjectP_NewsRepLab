# NewsRepLab

**Evaluating Pretrained Text Representations for News Classification under Limited Training Data**

UAlbany ICSI436/536, Fall 2026. Course topic: **Project P - Learning with
Pretrained Representations**.

Research question: **Do pretrained text representations improve news
classification performance compared with traditional TF-IDF features when
labeled training data is limited?**

## Current status

The feasibility test has passed in the student's Spyder environment: AG News
loaded, RTX 4060 CUDA computation worked, and five articles produced valid
384-dimensional MiniLM embeddings. `src/feasibility_test.py` is preserved.

Phase 2 adds `src/pilot_experiment.py`: TF-IDF versus frozen MiniLM, each followed
by multinomial Logistic Regression, using nested balanced training sets of 100
and 400 and a fixed disjoint validation set of 800 (seed 42). The official test
split is not used. The seed-42 pilot has completed successfully, with saved results
under `results/pilot/20261019T223707_829532Z`. Final test evaluation has not been performed.
No PowerPoint or LaTeX report has been generated.

## Run the repeated baselines

Open `src/repeated_experiment.py` in the same Spyder environment and press F5.
It evaluates seeds **42, 123, and 2026** using the completed pilot's exact fixed
800-example validation set. Each seed has balanced nested 100/400 training sets,
identical for both representations. Model/preprocessing settings and C=1.0 stay
unchanged; no hyperparameter tuning or official test evaluation is performed.

The original scripts/results are preserved. Matching embeddings are reused.
Each new run saves 12 individual scores, mean/sample SD summaries, paired
differences, two comparison figures, and a data-derived `analysis.md` under
`results/repeated/<UTC timestamp>/`. The repeated real experiments have not yet
been run during implementation. See [the repeated-experiment guide](docs/repeated_experiments.md)
for the fixed reference, seed interpretation, cache behavior, and limitations.

## Run the Phase 2 pilot

Open `src/pilot_experiment.py` in Spyder with the existing `icsi436-p` interpreter,
inspect the code, and press F5. CUDA is required. Successful runs print
`PILOT COMPLETE` and save CSV/JSON metrics, predictions, errors, configuration,
sample IDs, and a performance figure under `results/pilot/<UTC timestamp>/`.

Read [the pilot protocol and timing guide](docs/pilot_experiment.md) before
interpreting results. Both classifiers use L2 regularization with C=1.0, which is
an untuned common starting point. Embeddings are cached; cache reads and reused
encoding measurements are explicitly distinguished from new computation.
All findings must be labeled **preliminary single-seed validation results**.

## Research design and later study

| Component | Planned approach |
| --- | --- |
| Dataset | AG News, four news categories |
| Baseline | TF-IDF features + Logistic Regression |
| Pretrained approach | Frozen all-MiniLM-L6-v2 embeddings + Logistic Regression |
| Main variable | Number of labeled training examples |
| Evaluation | Accuracy, Macro-F1, computational cost |
| Main analysis | Learning curves and when pretrained features improve sample efficiency |

Use matching stratified subsets/seeds for both approaches, and reserve a
validation set from training data for decisions. Fit TF-IDF on training data
only. Keep the official test split out of tuning. Record seeds, sample sizes,
model/dataset revisions, hyperparameters, dependency versions, and device details
to reproduce results. Separate downloads/model startup, feature extraction,
classifier fitting, and prediction costs; five-article startup timings are not
experimental benchmarks. The first pilot implements 100/400 training examples
and an 800-example validation set with seed 42. The repeated-baseline script adds
three training-subset seeds; broader sample sizes, separate hyperparameter tuning,
and final test evaluation remain future work.

## Environment

Use the existing Conda environment; no environment recreation is necessary.

- OS: Windows; IDE: Spyder; environment: `icsi436-p`.
- Interpreter: `D:\Anaconda\envs\icsi436-p\python.exe`.
- Python: 3.11.17; PyTorch: 2.11.0+cu128.
- NumPy: 2.4.6; Datasets: 5.1.0; Sentence Transformers: 6.1.0.
- scikit-learn: 1.9.1; Matplotlib: 3.11.2 (existing installations).
- User-provided hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, CUDA 12.8.
- Initialization checks confirmed imports of all four direct dependencies and
  `torch.cuda.is_available() == True`. The full feasibility script has not been
  executed during initialization; no dataset/model downloads were initiated.

`requirements.txt` records only the direct dependencies used now, with observed
versions. It is not a full Conda export or a command to reinstall packages.
Preserve the working CUDA-enabled PyTorch installation. scikit-learn and
Matplotlib are now included in the inventory for the pilot; no packages were installed.

## Directory structure

```text
ICSI436_ProjectP_NewsRepLab/
|-- src/
|   |-- feasibility_test.py
|   |-- pilot_experiment.py
|   `-- repeated_experiment.py
|-- data/
|   `-- README.md
|-- results/
|   |-- pilot/
|   `-- repeated/                 (created when repeated baselines run)
|-- reports/
|   |-- midterm/
|   `-- final/
|-- docs/
|   |-- course_requirements.md
|   |-- pilot_experiment.md
|   `-- repeated_experiments.md
|-- tests/
|   |-- test_pilot_experiment.py
|   `-- test_repeated_experiment.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```

Small `.gitkeep` files preserve empty output folders if Git is initialized later.
The Datasets cache defaults to `D:\hf_cache\datasets`, outside the project;
the script creates it automatically when Part B runs. Model files still go under
the Git-ignored `.cache/sentence_transformers/` directory. Hugging Face Hub may
also use its separate shared download cache. No cache, environment, weight file,
or large generated artifact should be committed. The dataset cache is absolute;
project paths use the script's location, so Spyder's working directory does not
matter. The existing project directory structure is unchanged.

### Configurable dataset cache

No configuration is needed for the default `D:\hf_cache\datasets`. To choose a
different location, set `HF_DATASETS_CACHE` to a short absolute path before
running the file. For example, in the Spyder IPython console:

```python
import os
os.environ["HF_DATASETS_CACHE"] = r"D:\hf_cache\datasets"
```

Then run the file with F5. The script reads this variable on each execution and
passes the selected directory explicitly to `load_dataset(..., cache_dir=...)`,
even if Datasets was already imported in that kernel. This console setting lasts
for the current kernel session. An unset or empty variable uses the default;
relative paths are rejected. Part B prints the actual directory in use.

This short path addresses **WinError 206**: Datasets generates long filenames
for cache locks, including a flattened copy of the cache path. Under a deeply
nested directory, the resulting filename or full path can exceed Windows limits.
Shortening the cache root reduces both lengths without moving the project.
Existing files in `data/hf_datasets/` are left untouched and are no longer used
by this loader. The prepared dataset may need to be rebuilt in the new location.
This setting relocates the Datasets cache, not the separate Hugging Face Hub or
MiniLM model cache. See [Hugging Face cache management](https://huggingface.co/docs/datasets/en/cache)
and [Windows path limits](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation).

## Run manually in Spyder

1. Use your existing Spyder configuration for `icsi436-p`. In the IPython console,
   run `import sys; print(sys.executable)` and verify the interpreter path above.
2. If it is different, select that existing interpreter in Spyder's Python
   interpreter preferences and restart the IPython kernel. Do not recreate the
   environment or reinstall PyTorch.
3. Open `src/feasibility_test.py` from this project and use **Run File / F5**.
4. Allow time for the first dataset/model downloads. Keep the complete output,
   including any traceback, for review before moving to classification.

Optional PowerShell command (an alternative to Spyder):

```powershell
& 'D:\Anaconda\envs\icsi436-p\python.exe' 'D:\大学\UAlbany\Courses\ICSI436 Machine Learning\Project\ICSI436_ProjectP_NewsRepLab\src\feasibility_test.py'
```

## What the script teaches and expected output

- **Part A:** interpreter/version, PyTorch/CUDA availability, GPU name and memory,
  and a tiny GPU matrix multiplication. If CUDA is unavailable or its probe fails,
  an explicit warning selects CPU. A CPU pass does not validate GPU execution.
- **Part B:** split names, expected 120,000 training and 7,600 test rows, label
  mapping, and five short article previews with their known labels. Raw text is
  input X; the category is target y. Preview shortening does not alter model input.
- **Part C:** MiniLM load time and five-article encoding time measured separately.
  Each original text is encoded without its label. The tokenizer can truncate
  long inputs; the model's maximum token length is printed. Output is a CPU NumPy
  array even when neural inference uses CUDA.
- **Part D:** floating-point values must all be finite, row counts must agree,
  and the embedding matrix must have shape **(5, 384)**. The first eight values
  of the first row are printed. These values are features, not class probabilities.

Successful CUDA output ends with:

```text
FEASIBILITY TEST PASSED on cuda:0: five articles produced valid 384-D embeddings.
No classifier training, fine-tuning, accuracy, or Macro-F1 evaluation was performed.
```

CPU mode prints `cpu` instead and clarifies that GPU feasibility is unverified.
Exact vector values and timings can vary. Loading includes download time on a
cold cache; extraction includes CPU tokenization, inference, and CPU array
conversion. GPU synchronization makes timings include completed GPU work.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Import fails | Confirm `sys.executable`, the named dependency, and the original traceback; restart the Spyder kernel after selecting the existing interpreter. |
| Download/network/model loading fails | Check access to Hugging Face, internet/proxy settings, and the original exception. Rerun after connectivity recovers; cached downloads are reused. |
| WinError 206: filename or extension too long | Use the default `D:\hf_cache\datasets` or set `HF_DATASETS_CACHE` to another short absolute path; verify the path printed in Part B. |
| Cache permission error or full disk | Check available space and write access to the printed cache directories, including the external dataset cache. |
| CUDA unavailable | The script continues on CPU. Check the interpreter and NVIDIA driver before changing installed packages. |
| CUDA probe fails | The exact error is printed and CPU is selected. This is not a GPU pass. |
| CUDA error during MiniLM loading/encoding | The run fails with a traceback; close GPU-heavy applications and restart the kernel before rerunning. It does not silently retry a failed model operation on CPU. |
| Wrong shape or NaN/infinity | Validation fails; retain the traceback and verify the requested model identifier and library versions. |
| First run is slow | Downloads/model startup and initial inference add overhead; do not interpret this as a controlled benchmark. |
| Windows distributed redirect notice | Dependency imports may print a notice that distributed process redirects are unsupported on Windows. This script does not use distributed training; inspect any actual subsequent exception separately. |

## Future development

1. Run the three-seed baselines in Spyder and inspect stability, metrics, and errors.
2. Use only verified, qualified findings in the midterm presentation.
3. Later extend training sizes/seeds and tune each representation on validation.
4. Evaluate final selected methods on the untouched official test split.
5. Analyze limitations and reproducibility, then prepare the course presentations
   and LaTeX reports after reviewing the real results.

See [course requirement notes](docs/course_requirements.md) for the provided
instruction PDF's deliverables and Project P alignment. No slides or reports have
been generated at this stage.

## Sources and attribution

- [AG News dataset card and hosted dataset](https://huggingface.co/datasets/fancyzhx/ag_news).
  Original corpus collected by Antonio Gulli; classification benchmark associated
  with Zhang, Zhao, and LeCun (2015), *Character-level Convolutional Networks for
  Text Classification*.
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
  Public pretrained encoder used for 384-dimensional sentence/paragraph features;
  model-card default truncation length is 256 word pieces.
- [SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html).
- Course instructions: `../project.pdf`, supplied by the user and left unchanged.
- Initial scaffold and educational code were prepared with Codex assistance.
  The group should understand, review, and document its own final implementation
  and follow course attribution requirements.
