# Reproducibility and environment

## Established workstation

Use the existing `icsi436-p` Conda environment and Spyder configuration. No package,
CUDA, or PyTorch change is required for this project. `requirements.txt` lists
the seven runtime packages imported or version-checked by the scripts. Historical
run metadata record Python, package versions, hardware, revisions, and settings.
The inventory is not a transitive lockfile or a portable Conda export.

For a **separate fresh environment**, use Python 3.11, select an appropriate
CUDA-enabled PyTorch build for the machine, and then install the recorded package
versions from `requirements.txt`. For example, after activating that separate
environment, `python -m pip install -r requirements.txt` installs the inventory;
it does not select or verify a correct GPU build on its own. This command is not
needed on the verified workstation and was not executed during consolidation.
Check `sys.executable`, package imports, `torch.__version__`, and CUDA availability
before running the feasibility script. Version availability and GPU compatibility
on another machine must be checked there; exact reproduction is not guaranteed.

The recorded PyTorch build is 2.11.0+cu128; `torch==2.11.0` accepts that local
version. Classifier code uses the recorded scikit-learn 1.9.1 API. Arbitrarily
substituting older packages is not a supported reproduction route. Optional
formatting tools are already available on the development machine, but are not
runtime dependencies and do not belong in the ML requirements inventory.

## Paths and cached data

All project paths derive from `__file__`. Set `HF_DATASETS_CACHE` to a short
absolute writable path if `D:\hf_cache\datasets` is unavailable. On Unix-like
systems the Windows default is not absolute and must be overridden before import.
The default remains unchanged for the verified Windows environment.

Datasets may use a separate shared Hub download cache. Project model downloads
use `.cache/sentence_transformers/`; numeric features and paired metadata use
`.cache/pilot_embeddings/`. These are deliberately excluded from Git. The
feasibility test loads both splits to check names/counts; supervised scripts
request only `train`. Neither loading nor importing is test evaluation.

## Reference chain and a fresh checkout

The selected small historical artifacts in Git are necessary inputs, not only
presentation attachments:

1. The pilot has no historical run dependency. It creates a new timestamped run.
2. Repeated baselines pin `results/pilot/20261019T223707_829532Z`, checking its
   metadata and exact sample manifest. They retain the original seed-42 rows and
   validation order. They can populate missing embeddings using CUDA.
3. The sweep pins `results/repeated/20261019T225846_619676Z`, checking metadata,
   sample manifest, individual scores, and encoding context. It requires four
   existing MiniLM blocks: shared validation and a 400-row training block per seed.
   It deliberately does not encode missing blocks.

On the original environment/hardware, rerunning repeated baselines can recreate
the matching cache blocks while preserving all historical result folders. On
another GPU or software stack, device/version-dependent cache keys can differ.
The old sweep may therefore reject the new cache or reference. Do not weaken its
checks or edit historical metadata. For a separately reviewed new study, point
`REFERENCE_RUN` to the new completed repeated run and treat it as new provenance;
this is a protocol choice, not a maintenance change made here. Never overwrite
the old reference run to make a check pass.

## Data and source provenance

Supervised dataset revision: `eb185aade064a813bc0b7f42de02595523103ca4`.
MiniLM revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
The saved dataset fingerprint is additionally checked. A matching revision alone
does not guarantee a matching fingerprint after library changes.

Sample manifests contain row IDs and labels, not article text. Metadata record
the source byte hashes that were current when the experiment ran. Maintenance
changes those source hashes, but original metadata are historical records and
must not be rewritten. The initial Git commit preserves the pre-maintenance
source snapshot; compare a saved hash before claiming it is the exact execution
version. Formatting is not evidence that a full real experiment was rerun.

Git attributes preserve the **exact bytes** of historical `results/` and
`reports/` files. Automatic line-ending conversion would otherwise invalidate
manifest/reference-file hashes. New source/docs use consistent LF line endings.

## Offline checks and practical boundaries

From the project root in the existing environment:

```powershell
python -B -m unittest discover -s tests -v
```

The suite uses synthetic text/features, a fake encoder, temporary output folders,
and real scikit-learn fits. It checks class balance, nesting/disjointness,
training-only TF-IDF, cache identity/checksums, aggregate arithmetic, reuse, and
the C=1 failure gate. These are software checks, not new scientific measurements.
Import checks can set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` to prohibit
Hub/dataset access. These process-local settings do not alter the environment.

If Windows sandbox permissions block temporary-file creation/replacement,
rerun the same checks in a normal local terminal. Do not change ML logic or
remove cache checks to accommodate that runner restriction.

Three seeds do not make results statistically conclusive. Timing observations
exclude some startup costs but still depend on caching, hardware, and execution
order. No full fresh-cache GPU or cross-machine rerun was performed during this
maintenance task. See [maintenance review](maintenance_review.md) for evidence.
