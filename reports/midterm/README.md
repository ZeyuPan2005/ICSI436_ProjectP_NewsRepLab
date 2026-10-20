# NewsRepLab — Midterm submission materials

**Group 8 — Zeyu Pan, Jiayu Liao, Yuhan Sun**

This folder contains materials prepared from the completed experiment artifacts. No new classification experiments were run.

## Main files

- `NewsRepLab_Midterm.pptx`: exactly five editable slides, including one native learning-curve chart with an embedded Excel data snapshot, three editable tables, and English speaker notes with source attribution.
- `midterm_report.pdf`: compiled academic report, exactly one US Letter page.
- `midterm_report.tex`: self-contained LaTeX source, including the bibliography. Compile twice with `pdflatex -interaction=nonstopmode -halt-on-error midterm_report.tex` after source edits so references resolve.
- `presentation_script.md`: 510 spoken words, with a 4:30 target and timings for each slide. Timing is estimated and requires rehearsal.
- `repeated_chart_data.csv`, `regularization_chart_data.csv`, and `verified_results.json`: recomputed descriptive summaries and artifact provenance for the displayed results. Values in these source files are proportions; slides and report display percentages. Standard deviations use `ddof=1`.

## Central narrative

Changing the feature representation while retaining multiclass Logistic Regression gives a practical comparison of label efficiency. Frozen MiniLM has higher validation accuracy and macro-F1 at both tested training budgets for each matched seed. At 100 labels, its mean accuracy is 82.38%, compared with TF-IDF's 66.21%; its 100-label mean also exceeds TF-IDF's 400-label mean of 76.71%. The advantage is promising within these settings. Three sampling seeds and one fixed validation set do not establish statistical significance or universal superiority.

The regularization sweep is a separate exploratory result: at 400 labels, moving from C=1 to C=10 improves TF-IDF mean accuracy by 0.75 percentage points and reduces MiniLM by 0.79 points. Representation-specific tuning remains future work. C=1 repeats the earlier baseline exactly and is not independent evidence.

## Source-of-truth runs

Paths below are relative to the project root:

- Pilot: `results/pilot/20261019T223707_829532Z/`
- Repeated baselines: `results/repeated/20261019T225846_619676Z/`
- Exploratory regularization: `results/regularization/20261019T232052_926893Z/`

Summary means and sample standard deviations were independently recomputed from the saved individual JSON records and checked against the saved summary CSV files. The sample manifest confirms balanced labels, nested subsets, and disjoint training/validation indices. No test data were loaded during preparation.

The chart workbook is a static snapshot of the saved summary values, not a live link to future experiment files. Chart percentages are stored to six decimal places for Excel precision compatibility and displayed to two decimal places. The accompanying JSON and CSV files retain the recomputed floating-point values.

## Checks completed and manual review

- All five final slides were re-imported and visually inspected. Package, slide geometry, font, chart-data, and editable-table checks passed. No layout findings or warnings remained.
- The PDF was compiled twice, inspected visually, and confirmed to contain one page. No unresolved citations or overfull/underfull box warnings remained.
- Existing Python code and experiment results were preserved. Their file hashes were checked after preparation.
- Group 8 and all three members are visible on slide 1, appear beneath the report title, and are included in the script introduction and slide 1 notes. No member placeholders remain.
- Open the PPTX in your installed PowerPoint and check slideshow mode once. The generated-file previews were inspected, but a native PowerPoint edit/save/reopen test was not performed.
- Assign speaking roles and rehearse with a timer. Aim for 4:20–4:40, leaving a margin below the five-minute limit.
- Keep every claim labeled as validation evidence. Final test evaluation, expanded budgets, stratified K-fold selection inside the labeled budget, systematic representation comparisons, error analysis, and controlled cost measurement remain future work.

## Final targeted revision

- Preserved the academic title, slide order, existing visual style, learning-curve chart, regularization table, and all quantitative results.
- Added a slide 3 footnote and speaker-note clarification: Round 2 includes seed-42 pilot configurations, and Round 3 includes C=1 reproduction. Counts of 4, 12, and 18 are not 34 independent experiments.
- Added a visible exploratory/nonconclusive label on slide 4. Speaker notes and the script distinguish observed validation performance and seed variability from the interpretation that representation-specific hyperparameter selection may be useful.
- Rewrote the script in simpler English. Spoken priorities are expanded budgets, future stratified K-fold selection within each labeled budget, and deeper error analysis followed by final evaluation on the untouched official test set. Cost analysis remains included briefly; K-fold is planned and is not claimed to improve accuracy directly.
- Updated all slide speaker notes to match the revised script. Preserved source attribution and methodological notes.
- Rechecked all accuracy and macro-F1 means and standard deviations against both verified_results.json and the original individual-results JSON / summary CSV artifacts. The three existing provenance files were left unchanged.
- Re-rendered and visually inspected all five final slides, checked the editable chart and tables, recompiled and inspected the one-page US Letter PDF, and checked for remaining placeholders. Font sizes in the report are unchanged.

Timing remains an estimate, not a recorded rehearsal. Use the slide-by-slide timings and practice at a comfortable pace to finish in 4:20–4:40. Check the deck once in the installed PowerPoint application before presenting.
