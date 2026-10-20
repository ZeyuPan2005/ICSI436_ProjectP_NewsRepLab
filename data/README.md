# Data directory

The feasibility script loads [AG News on Hugging Face](https://huggingface.co/datasets/fancyzhx/ag_news)
with `load_dataset("fancyzhx/ag_news", cache_dir=...)`.

The expected splits are 120,000 training examples and 7,600 test examples.
Each example contains `text` (the news input) and `label` (the target category).
The four labels are World, Sports, Business, and Sci/Tech; the script prints the
actual integer-to-name mapping from dataset metadata.

The Datasets cache defaults to the short external path `D:\hf_cache\datasets`
to avoid Windows filename/path length errors (WinError 206). Set the environment
variable `HF_DATASETS_CACHE` to another short absolute path to override it; see
the project README for a Spyder example. Part B creates the directory if needed
and passes that same path explicitly to `load_dataset` as `cache_dir`.
Hugging Face may also use its separate shared Hub download cache. Existing files
under `data/hf_datasets/` are left untouched but are no longer used by this loader.
Downloaded data under this project directory remain excluded from Git. Do not
commit copies of the dataset.
The script loads both splits but encodes only five selected training articles;
it does not preprocess the full dataset or inspect test article contents.

Model files are stored separately under `.cache/sentence_transformers/` at the
project root. Neither cache is created by importing the script alone.

AG News originates from the AG news corpus collected by Antonio Gulli. The
classification benchmark is associated with Xiang Zhang, Junbo Zhao, and Yann
LeCun, *Character-level Convolutional Networks for Text Classification* (2015).
See the linked dataset card for source details and the hosted dataset version.

## Supervised experiment boundary

The three supervised scripts request only the pinned original training split.
They use matching balanced training rows and a disjoint validation set; no
official test examples enter training, hyperparameter decisions, or evaluation.
The feasibility script above is the only entry point that loads split metadata
and counts for both splits. See [architecture](../docs/architecture.md) for the
full workflow and [reproducibility](../docs/reproducibility.md) for cache recovery.
