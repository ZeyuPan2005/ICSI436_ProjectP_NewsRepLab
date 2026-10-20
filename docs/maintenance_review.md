# Codebase consolidation review

## Scope and architectural decision

This is maintenance after the verified midterm, not a fourth research round.
The four standalone Spyder scripts and existing script-to-script helper reuse
remain intact. A new package, configuration framework, or training abstraction
would add complexity without improving this small study. Small duplicated model
setup and cache-key preflight code remain explicit. No sampling calls, model
settings, feature definitions, reference-run constants, or cache schemas changed.

The initial local Git snapshot is `6d582d3` on `main`. It was committed before
these edits and preserves the pre-maintenance sources and selected artifacts.
Existing result folders, raw local predictions, and reviewed midterm files remain
unchanged. The environment and installed packages were not modified.

## Source and test changes

All seven Python files use the installed Black formatter's 88-column style.
Existing detailed educational comments were retained; docstrings now explain
shapes and responsibilities where helpful. New annotations describe boundaries
without imposing a new data model or changing runtime validation.

| File | Specific maintenance |
| --- | --- |
| `src/feasibility_test.py` | Update the project-status description; annotate text/label/embedding and device outputs; replace a workstation-specific interpreter error message with the existing environment name |
| `src/pilot_experiment.py` | Annotate sampling, serialization, row selection, validation, and plotting; explain sparse/dense shapes, vocabulary alignment, frozen features, and historical cache timing; make the interpreter error message portable |
| `src/repeated_experiment.py` | Annotate sampling and aggregation boundaries; explain guarded helper imports and sample SD versus confidence intervals |
| `src/regularization_experiment.py` | Annotate classifier settings, cache retrieval, aggregation, and reports; explain why the small independent cache identity check prevents unintended encoding |
| `tests/test_pilot_experiment.py` | Formatting and explicit UTF-8 fixture reads |
| `tests/test_repeated_experiment.py` | Formatting and explicit UTF-8 fixture reads |
| `tests/test_regularization_experiment.py` | Formatting and explicit UTF-8 fixture reads |

The existing explanations of training-only TF-IDF, softmax/cross-entropy, learned
classifier weights/intercepts, frozen MiniLM, random seeds, regularization,
accuracy, macro-F1, and validation limitations remain in the source.

## Documentation and configuration

- Root README is the entry point: Group 8, research question, completed rounds,
  verified validation scores, execution, limitations, future work, and attribution.
- New architecture, experiment log, reproducibility, maintenance, and Git guides
  separate completed work from future experiments and describe reference/cache
  dependencies that matter on a fresh checkout.
- Existing pilot/repeated/regularization/course/data guides have current status,
  consistent terminology, and links rather than duplicated numerical narratives.
- `requirements.txt` includes the already installed Transformers version because
  cache identity records it. This is documentation, not an installation or a
  complete transitive lockfile.
- `pyproject.toml` records optional formatter/linter settings; `.gitignore` excludes
  generated/private files and permits reviewed historical results explicitly.
- `.gitattributes` preserves historical artifact bytes. The SHA-256
  [artifact manifest](artifact_manifest.json) covers 62 selected existing files.

## Verification record

The checks below were executed before committing. Tests use synthetic
inputs and temporary files; their classifier fits are software checks, not new
AG News measurements. The working environment is the existing `icsi436-p`.

| Check | Result |
| --- | --- |
| Existing offline suite before maintenance | 18 tests passed |
| Offline suite after maintenance | 18 tests passed using unchanged installed dependencies |
| Syntax compilation | All four source scripts and three test modules passed |
| Guarded imports from another working directory | All four scripts imported; dataset loading, model construction, classifier fitting, and CUDA probes were forbidden |
| Executable AST comparison | Seven files matched the baseline after removing docstrings/annotations and normalizing the two interpreter messages and explicit UTF-8 test reads |
| Black / Ruff | Seven files passed the Python 3.11, 88-column formatter check; selected E4/E7/E9/F lint checks passed |
| Saved settings and sampling | Original hyperparameters/revisions, class balance, nesting, disjointness, common validation, and pilot seed-42 row identity checked |
| Saved summary arithmetic | Four repeated groups and six sweep groups recomputed from saved individual scores and matched their JSON summaries; mean/sample SD independently checked for repeated groups |
| Historical C=1 reproduction | All 12 saved metric comparisons have zero difference; no new C sweep was run |
| Historical source hashes | The three supervised scripts in the initial snapshot exactly match the source/helper hashes recorded by Round 3; pilot also matches its own run metadata |
| Existing artifact preservation | SHA-256 unchanged for all 112 pre-existing result/report files, including untracked raw prediction/error files |
| Selected artifact preservation | 62 selected historical files match the committed artifact manifest |
| Documentation | Relative links in maintained README/docs/data notes resolve |

The initial sandboxed baseline test attempt encountered Windows temporary-file
permission errors. The same suite passed outside that sandbox; no ML logic or
dependencies were changed to accommodate it. The formatter likewise needed the
normal Windows process context. These are execution-runner limitations.

Re-run the maintained software checks from the project root when needed:

```powershell
python -B -m unittest discover -s tests -v
python -B -m black --workers 1 --check src tests
python -B -m ruff check --no-cache src tests
```

Black and Ruff are optional tools already installed on this workstation; the
experiment scripts do not depend on them. The one-time source/provenance audit
compared the baseline snapshot with the maintained files and re-read saved JSON;
it did not execute any experimental `main()`.


## Limits and publication review

No fresh-cache AG News experiment, full CUDA pipeline, interactive Spyder F5
session, or cross-machine reproduction was run during maintenance. Offline
checks and normalized AST comparison support preservation of the reviewed logic;
they are not proof of identical real-world timings or universal behavioral
equivalence. Type annotations and the two diagnostic messages intentionally
change source metadata/output, and explicit UTF-8 test reads change decoding.

Historic metadata retain the original source hashes and workstation paths rather
than being rewritten to match the new code. Exact saved sources can be checked
against the initial commit. Hardware/package-dependent cache identity limits
portability; see [reproducibility](reproducibility.md).

The publication review and exact candidate file list are in
[Git preparation](git_preparation.md). No remote has been created or pushed.
