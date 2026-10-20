# NewsRepLab — Group 8 midterm speaking script

**Group 8: Zeyu Pan, Jiayu Liao, Yuhan Sun**

**Target: 4 minutes 30 seconds; acceptable rehearsal range: 4:20–4:40.** The script contains 510 spoken words. Speak at about 120 words per minute, with brief pauses and slide transitions. The spoken paragraphs below are the script. Headings, timings, stage directions, and review notes are not spoken. Assign speakers before rehearsal and avoid repeating introductions at each handoff.

## Slide 1 — Background and Research Motivation
**0:00–0:40 · approximately 40 seconds**

Hello, we are Group Eight: Zeyu Pan, Jiayu Liao, and Yuhan Sun. Our project is NewsRepLab.

We study news classification with limited labeled data. Labeling articles takes human effort, so we ask whether pretrained features can help us learn from fewer examples.

We compare TF-IDF, which represents word and phrase usage, with MiniLM embeddings, which use information learned during pretraining. Our focus is the quality of these representations and how efficiently they use task labels.

## Slide 2 — Research Question and Problem Setup
**0:40–1:20 · approximately 40 seconds**

Our research question is: with the same limited labels, do frozen MiniLM features improve news classification compared with TF-IDF?

We use AG News, with four categories: World, Sports, Business, and Science and Technology. Our training budgets are one hundred and four hundred articles.

Both pipelines use multiclass Logistic Regression. This keeps the classifier type consistent while we change the features. We train the classifier's weights and biases, but keep MiniLM frozen. We do not train the Transformer on our task.

## Slide 3 — Methodology and Experimental Design
**1:20–2:10 · approximately 50 seconds**

We completed a pilot, repeated baselines with three sampling seeds, and an exploratory regularization sweep. The counts overlap: the repeated baselines include the pilot, and the sweep reproduces the original baseline setting. These are not independent experiments throughout.

Both pipelines use identical examples. Training sets are balanced across classes, and each smaller set is contained in its larger set. We use one fixed validation set of eight hundred articles, separate from training. TF-IDF is fitted only on training text.

We measure accuracy and macro-F1. The official test set remains untouched. Different seeds change the training samples, while validation stays fixed.

## Slide 4 — Current Progress and Key Findings
**2:10–3:25 · approximately 75 seconds**

[Point to the learning curve.]

The chart shows mean validation accuracy across three seeds. At one hundred labels, TF-IDF reaches about sixty-six percent and MiniLM eighty-two percent. At four hundred labels, they reach about seventy-seven and eighty-five percent.

MiniLM performs better for every matched seed and budget in these runs. With one hundred labels, MiniLM also exceeds TF-IDF's four-hundred-label mean in these runs. Both improve on average with more labels, but MiniLM's gain is smaller. Scores also vary with the sampled training examples. These observations suggest useful sample efficiency, but two budgets do not prove a plateau.

[Point to the regularization table.]

The separate sweep changes C, the inverse regularization strength. Increasing C from one to ten slightly improves TF-IDF but lowers MiniLM's mean validation accuracy. This suggests that selecting hyperparameters separately for each representation may be useful. That is an interpretation, not a completed tuning result. These findings remain exploratory and are not statistically conclusive.

## Slide 5 — Challenges and Future Work
**3:25–4:30 · approximately 65 seconds**

Our main challenges are small training budgets, only three seeds, and repeated use of one validation set. Pretraining also brings outside information.

We have three priorities. First, expand training budgets to study sample efficiency more systematically. Second, use stratified K-fold cross-validation within each permitted labeled budget to select hyperparameters, fitting TF-IDF separately inside each fold. This is future work.

Third, examine errors more closely, especially Business versus Science and Technology. After fixing our protocol, we will evaluate on the untouched official test set. We will also compare representations more systematically and measure computational costs under controlled conditions.

Our current conclusion is promising label efficiency, with further evaluation still needed.

---

## Presenter review notes — not spoken

- Group 8: Zeyu Pan, Jiayu Liao, Yuhan Sun. Assign the slides among members before rehearsal.
- Timing is estimated, not a recording. Practice with a timer, including handoffs. Target 4:20–4:40 and stay below five minutes. Exact two-decimal values remain on slide 4; spoken values are deliberately rounded for clarity.
- Round counts are 4 pilot configurations, 12 repeated baseline configurations, and 18 exploratory sweep configurations. Round 2 includes the seed-42 pilot; Round 3 includes C=1 baseline reproduction. Do not describe these as 34 independent experiments.
- All performance numbers are **validation** results. Means use three training-sampling seeds; sample standard deviations use `ddof=1`. They are not confidence intervals or K-fold results. K-fold selection is future work, not an implemented method or a promise of higher accuracy.
- Round 2: `results/repeated/20261019T225846_619676Z/individual_results.json` and `summary.csv`.
- Round 3: `results/regularization/20261019T232052_926893Z/individual_results.json` and `summary.csv`. The C=1 scores exactly reproduce Round 2 at n=400.
- Source attribution: Zhang, Zhao, and LeCun (2015), *Character-level Convolutional Networks for Text Classification*, https://arxiv.org/abs/1509.01626 (AG News benchmark); Sentence Transformers, *all-MiniLM-L6-v2* model card, https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2.
