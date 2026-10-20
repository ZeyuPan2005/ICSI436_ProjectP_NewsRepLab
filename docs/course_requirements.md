# Course requirements and scope

Source: the user-provided `../project.pdf`, UAlbany CSI 436/536 Machine Learning,
Fall 2026, Course Project, instructor Prof. Chong Liu. The PDF remains unchanged
in the parent directory. These notes summarize it; they do not replace it.

## Why this project fits Project P

Project P asks groups to extract representations from publicly available
pretrained image or text models, train standard downstream machine learning
models on those features, and compare against simpler or raw features,
particularly with limited training data. Analyze when the pretrained features
help and their effect on sample efficiency, generalization, or robustness.
Other representations or downstream algorithms are optional novelty directions.

Our implemented comparison is TF-IDF + Logistic Regression versus frozen MiniLM +
Logistic Regression on AG News. Varying labeled training size addresses sample
efficiency. The feasibility script establishes prerequisites only. Three supervised
rounds and reviewed midterm deliverables are now complete; the final study remains
planned. See [experiment log](experiment_log.md) for current evidence.

## Course deliverables

- Midterm presentation: 5 points; soundness, organization, clarity, progress,
  and challenges.
- Midterm one-pager: 2 points; one page covering problem, progress, and challenges.
- Final presentation: 10 points; soundness, organization, clarity, performance,
  and novelty.
- Final report: 4 points; up to four pages, with introduction, problem statement,
  challenges, methods, results, conclusions, and optional references.
- Final Python code must reproduce every reported result. The instructions warn
  that non-reproducible results or substantial unattributed copied code can cost
  all 21 project points.
- Reports must be written in LaTeX and submitted as PDFs to Gradescope. Final
  code is emailed to the TA with the instructor copied, when the group submits.
- Work is completed in study groups; submitted work must be the group's own.
- No late submissions for the one-pager, final report, or code. This PDF does not
  list concrete deadlines; check course announcements/syllabus for dates/updates.

The reviewed midterm presentation/report are in `reports/midterm/`. Their
preparation does not mean they were submitted. Final materials, submission, and
any email remain separate user actions. Preserve source attribution and report
only actual measurements; feasibility alone is not classification evidence.
