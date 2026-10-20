"""NewsRepLab Phase 2: a preliminary, single-seed representation comparison.

Question: with little labeled news data, do frozen MiniLM features help a
standard classifier more than traditional TF-IDF features? We compare two
representations, not two custom learning algorithms. Both use scikit-learn's
multinomial Logistic Regression, with the same L2 regularization and solver.

The original AG News training split supplies everything: 800 fixed validation
rows (200/class), plus nested training subsets of 100 and 400 rows. The official
test split is never requested or evaluated. Seed 42 fixes the row selection.
Only selected article texts are processed; scanning the label column is enough
to find balanced subsets. No full-dataset Transformer inference is performed.

Feature extraction changes text into numbers. Classifier training then learns
how these numbers relate to known news categories. TF-IDF learns vocabulary
and inverse document frequencies ONLY from the current training subset. MiniLM
uses previously learned weights without any updates on this news dataset.

Run this file with F5 in Spyder's existing icsi436-p environment. CUDA is required
for new MiniLM embeddings; Logistic Regression runs on the CPU. Downloads and
model startup are recorded separately from feature/classifier timings. Cached
embeddings retain their original measured encoding time, explicitly marked as
reused; a fast cache read is not reported as a new encoding measurement.

All scores describe ONE seed, ONE validation set, and untuned settings. They do
not establish statistical significance or universal representation superiority.
"""

import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

IMPORT_START = time.perf_counter()
try:
    import numpy as np
    import torch
    from datasets import load_dataset
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sentence_transformers import SentenceTransformer
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
except Exception:
    print("Pilot dependency import failed. Active Python:", sys.executable)
    print(r"Use the existing D:\Anaconda\envs\icsi436-p\python.exe interpreter.")
    print("Keep the original traceback; do not reinstall PyTorch or change CUDA.")
    raise
IMPORT_SECONDS = time.perf_counter() - IMPORT_START


# Paths never depend on Spyder's current working directory. Keep the short
# external Datasets cache that fixed WinError 206 in the feasibility stage.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CACHE = Path(os.environ.get("HF_DATASETS_CACHE") or r"D:\hf_cache\datasets").expanduser()
MODEL_CACHE = PROJECT_ROOT / ".cache" / "sentence_transformers"
EMBEDDING_CACHE = PROJECT_ROOT / ".cache" / "pilot_embeddings"
RESULTS_ROOT = PROJECT_ROOT / "results" / "pilot"

DATASET_NAME = "fancyzhx/ag_news"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# These immutable Hub revisions were read from the successful feasibility run's
# local caches. Pinning them prevents a future remote update from changing inputs.
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SEED = 42
TRAIN_SIZES = (100, 400)
VALIDATION_PER_CLASS = 200
CLASS_IDS = (0, 1, 2, 3)
BATCH_SIZE = 32
EMBEDDING_DIMENSION = 384
DEVICE = "cuda:0"

# For a new measured encoding run, set this to True before running the file.
# It recomputes selected blocks and refreshes their cache; it never encodes the
# entire dataset. A normal rerun reuses matching cached feature arrays.
FORCE_RECOMPUTE_EMBEDDINGS = False

# This common setting is a controlled starting point, not a claim that C=1 is
# optimal for both feature spaces. Smaller C means stronger regularization.
# In the installed sklearn 1.9, l1_ratio=0 specifies L2 without the deprecated
# penalty argument. lbfgs optimizes the joint multinomial objective for 4 classes;
# no removed multi_class argument or one-versus-rest wrapper is needed.
LR_SETTINGS = dict(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=2000,
                   tol=1e-4, fit_intercept=True, random_state=SEED)
TFIDF_SETTINGS = dict(lowercase=True, ngram_range=(1, 2), min_df=1,
                      max_features=20000, sublinear_tf=True, norm="l2")


def utc_now():
    """Use an explicit UTC timestamp so saved runs can be identified unambiguously."""
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    """Write readable machine-readable metadata; reject non-finite JSON numbers."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def save_csv(path, rows, columns):
    """Always write a header, even when an experiment has zero prediction errors."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sample_indices(labels):
    """Reserve balanced validation rows, then select nested training rows."""
    labels = np.asarray(labels)
    if set(np.unique(labels).tolist()) != set(CLASS_IDS):
        raise ValueError("Expected the four AG News class IDs 0, 1, 2, 3.")
    rng = np.random.default_rng(SEED)
    validation = []
    training = {size: [] for size in TRAIN_SIZES}
    for class_id in CLASS_IDS:
        candidates = rng.permutation(np.flatnonzero(labels == class_id))
        if len(candidates) < VALIDATION_PER_CLASS + max(TRAIN_SIZES) // 4:
            raise ValueError(f"Not enough rows for class {class_id}.")
        # Reserve validation FIRST. No row in this prefix may train either model.
        validation.extend(candidates[:VALIDATION_PER_CLASS].tolist())
        remaining = candidates[VALIDATION_PER_CLASS:]
        for size in TRAIN_SIZES:
            # The first 25/class are contained in the first 100/class, so the
            # larger dataset adds examples rather than replacing the small set.
            training[size].extend(remaining[:size // 4].tolist())

    validation = rng.permutation(validation).tolist()
    for size in TRAIN_SIZES:
        training[size] = rng.permutation(training[size]).tolist()
    # Check the experimental design before feature extraction or classifier fit.
    if len(set(validation)) != 800:
        raise ValueError("Validation must contain 800 distinct rows.")
    for size, indices in training.items():
        if len(set(indices)) != size or set(indices) & set(validation):
            raise ValueError("Training rows must be unique and disjoint from validation.")
        if not np.all(np.bincount(labels[indices], minlength=4) == size // 4):
            raise ValueError("Training class counts are not balanced.")
    if not set(training[100]).issubset(training[400]):
        raise ValueError("The 100-example set must be contained in the 400-example set.")
    return training, validation


def select_rows(dataset, indices):
    """Materialize text only for requested rows, preserving row/label alignment."""
    selected = dataset.select(indices)
    texts = list(selected["text"])
    labels = np.asarray(selected["label"], dtype=np.int64)
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Selected rows contain empty or invalid text.")
    return texts, labels


def tfidf_features(train_texts, validation_texts):
    """Learn a training-only vocabulary/IDF and transform held-out validation."""
    vectorizer = TfidfVectorizer(**TFIDF_SETTINGS)
    # Sparse rows mostly contain zeros: each article uses only a small fraction
    # of the vocabulary. L2 normalization keeps article feature norms comparable.
    # fit_transform learns vocabulary + IDF AND extracts training features.
    started = time.perf_counter()
    train_features = vectorizer.fit_transform(train_texts)
    train_seconds = time.perf_counter() - started
    # transform uses those existing statistics. Fitting on validation would leak
    # information about unseen articles into the representation being evaluated.
    started = time.perf_counter()
    validation_features = vectorizer.transform(validation_texts)
    validation_seconds = time.perf_counter() - started
    return train_features, validation_features, vectorizer, train_seconds, validation_seconds


def validate_embeddings(array, count):
    """Reject stale, malformed, or non-finite feature arrays before fitting LR."""
    if array.shape != (count, EMBEDDING_DIMENSION):
        raise ValueError(f"Expected embeddings of shape ({count}, 384), got {array.shape}.")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("Embedding values must be finite floating-point numbers.")


def embedding_block(model, texts, indices, context, force=False):
    """Encode or reuse one ordered text block, retaining honest timing provenance."""
    # The cache key covers text content/order, source row IDs, dataset fingerprint,
    # pinned model revision, encoding settings, device, and package versions.
    # A different subset or configuration cannot accidentally reuse wrong rows.
    text_digest = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).hexdigest()
    identity = dict(context, indices=list(indices), text_sha256=text_digest)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    EMBEDDING_CACHE.mkdir(parents=True, exist_ok=True)
    array_path = EMBEDDING_CACHE / f"{key}.npy"
    metadata_path = EMBEDDING_CACHE / f"{key}.json"
    if not force and array_path.exists() and metadata_path.exists():
        started = time.perf_counter()
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        # allow_pickle=False reads plain numeric data, not arbitrary Python objects.
        features = np.load(array_path, allow_pickle=False)
        validate_embeddings(features, len(texts))
        if saved["identity"] != identity:
            raise ValueError(f"Cache identity mismatch: {metadata_path}")
        if hashlib.sha256(features.tobytes()).hexdigest() != saved["array_sha256"]:
            raise ValueError(f"Embedding cache checksum mismatch: {array_path}")
        seconds = float(saved["extraction_seconds"])
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid cached extraction timing.")
        return features, dict(key=key, cache_hit=True, measured_extraction_seconds=seconds,
                              compute_this_call_seconds=0.0,
                              cache_read_seconds=time.perf_counter() - started,
                              cache_write_seconds=0.0, measured_at=saved["measured_at"])

    # GPU operations are asynchronous. Synchronize on both timing boundaries so
    # elapsed seconds include completed computation, not just queued CUDA work.
    torch.cuda.synchronize(DEVICE)
    started = time.perf_counter()
    with torch.inference_mode():
        features = model.encode(texts, batch_size=BATCH_SIZE, device=DEVICE,
                                show_progress_bar=False, convert_to_numpy=True,
                                normalize_embeddings=True, precision="float32")
    torch.cuda.synchronize(DEVICE)
    seconds = time.perf_counter() - started
    validate_embeddings(features, len(texts))
    # This measurement includes CPU tokenization, GPU forward computation, and
    # transfer to a CPU NumPy array. It excludes model load, cache I/O, and LR fit.
    saved = dict(identity=identity, extraction_seconds=seconds, measured_at=utc_now(),
                 array_sha256=hashlib.sha256(features.tobytes()).hexdigest())
    started = time.perf_counter()
    # Replace each file only after its temporary write finishes. A missing half
    # of a cache pair is recomputed; inconsistent existing pairs fail visibly.
    temporary_array = array_path.with_suffix(".tmp")
    with temporary_array.open("wb") as handle:
        np.save(handle, features, allow_pickle=False)
    temporary_array.replace(array_path)
    temporary_metadata = metadata_path.with_suffix(".tmp.json")
    save_json(temporary_metadata, saved)
    temporary_metadata.replace(metadata_path)
    return features, dict(key=key, cache_hit=False, measured_extraction_seconds=seconds,
                          compute_this_call_seconds=seconds, cache_read_seconds=0.0,
                          cache_write_seconds=time.perf_counter() - started,
                          measured_at=saved["measured_at"])


def fit_and_evaluate(name, size, train_features, train_labels, validation_features,
                     validation_labels, validation_texts, validation_indices, class_names, run_dir):
    """Fit standard multiclass LR and save held-out predictions/error analysis."""
    # Both sparse TF-IDF and dense embeddings are numeric matrices (n samples,
    # d features), which is all Logistic Regression needs. Their d values differ,
    # so these classifiers share a functional form, not the same parameter count.
    # LR learns W (4 x d coefficients) and b (4 intercepts). Each article gets
    # four scores W*x+b; softmax converts them into probabilities summing to 1.
    # sklearn minimizes multiclass cross-entropy (-log probability of the true
    # class), with L2 regularization on weights. We implement no custom loss.
    classifier = LogisticRegression(**LR_SETTINGS)
    started = time.perf_counter()
    # Treat failed convergence as an explicit failed run, not a polished result.
    # A higher iteration limit can be reviewed later if the pilot needs it.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(train_features, train_labels)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = classifier.predict_proba(validation_features)
    predictions = classifier.classes_[np.argmax(probabilities, axis=1)]
    prediction_seconds = time.perf_counter() - started

    # Accuracy is the fraction of correct labels. Macro-F1 averages the four
    # per-class F1 scores equally; it exposes weak class performance that a single
    # overall accuracy can hide. Validation labels are used here, never in fit().
    result = dict(representation=name, train_size=size, validation_size=len(validation_labels),
                  seed=SEED, preliminary=True,
                  validation_accuracy=float(accuracy_score(validation_labels, predictions)),
                  validation_macro_f1=float(f1_score(validation_labels, predictions,
                                                     labels=CLASS_IDS, average="macro", zero_division=0)),
                  lr_training_seconds=training_seconds, prediction_seconds=prediction_seconds,
                  feature_dimension=int(train_features.shape[1]),
                  lr_iterations=int(np.max(classifier.n_iter_)),
                  error_count=int(np.sum(predictions != validation_labels)))
    prefix = f"{name}_{size}"
    prediction_rows = []
    for position, row_id in enumerate(validation_indices):
        true_label = int(validation_labels[position])
        predicted = int(predictions[position])
        prediction_rows.append(dict(dataset_row=row_id, true_label=true_label,
                                    true_class=class_names[true_label], predicted_label=predicted,
                                    predicted_class=class_names[predicted], correct=(true_label == predicted),
                                    predicted_probability=float(np.max(probabilities[position])),
                                    text_preview=" ".join(validation_texts[position].split())[:240]))
    columns = list(prediction_rows[0])
    save_csv(run_dir / f"{prefix}_predictions.csv", prediction_rows, columns)
    errors = [row for row in prediction_rows if not row["correct"]]
    save_csv(run_dir / f"{prefix}_errors.csv", errors, columns)
    save_json(run_dir / f"{prefix}_diagnostics.json", dict(
        class_names=class_names, confusion_matrix_rows="true", confusion_matrix_columns="predicted",
        confusion_matrix=confusion_matrix(validation_labels, predictions, labels=CLASS_IDS).tolist(),
        classification_report=classification_report(validation_labels, predictions,
            labels=CLASS_IDS, target_names=class_names, output_dict=True, zero_division=0),
        learned_coefficient_shape=list(classifier.coef_.shape),
        learned_intercept_shape=list(classifier.intercept_.shape)))
    print(f"{name}, n={size}: accuracy={result['validation_accuracy']:.4f}, "
          f"macro-F1={result['validation_macro_f1']:.4f}, errors={len(errors)}")
    for error in errors[:3]:
        print(f"  Row {error['dataset_row']}: {error['true_class']} -> "
              f"{error['predicted_class']} | {error['text_preview'][:120]}")
    return result


def save_comparison_figure(rows, path):
    """Plot actual validation scores only; a single seed has no error bars."""
    # A Figure with an Agg canvas saves a PNG without changing Spyder's GUI backend.
    figure = Figure(figsize=(10, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, metric, title in zip(axes, ["validation_accuracy", "validation_macro_f1"],
                                  ["Validation accuracy", "Validation macro-F1"]):
        for name, color in [("tfidf", "#2563eb"), ("minilm", "#d97706")]:
            selected = sorted([row for row in rows if row["representation"] == name],
                              key=lambda row: row["train_size"])
            axis.plot([row["train_size"] for row in selected], [row[metric] for row in selected],
                      marker="o", color=color, label=f"{name.upper()} + LR")
        axis.set(title=title, xlabel="Labeled training examples", ylabel="Score", ylim=(0, 1))
        axis.set_xticks(TRAIN_SIZES)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Preliminary pilot: seed 42, fixed validation n=800; no test evaluation")
    figure.savefig(path, dpi=160)


def main():
    """Run four controlled experiments and write a separate folder per execution."""
    run_dir = RESULTS_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = run_dir / "run_metadata.json"
    packages = {name: version(name) for name in ["numpy", "torch", "datasets", "sentence-transformers",
                                                "transformers", "scikit-learn", "matplotlib"]}
    metadata = dict(status="running", started_at=utc_now(), preliminary_single_seed=True,
                    seed=SEED, python=sys.version, executable=sys.executable, packages=packages,
                    platform=platform.platform(), cpu=platform.processor(), cpu_count=os.cpu_count(),
                    dataset=DATASET_NAME, dataset_revision=DATASET_REVISION, split="train",
                    model=MODEL_NAME, model_revision=MODEL_REVISION,
                    dataset_cache=str(DATASET_CACHE), model_cache=str(MODEL_CACHE),
                    embedding_cache=str(EMBEDDING_CACHE), classifier_settings=LR_SETTINGS,
                    tfidf_settings=TFIDF_SETTINGS, import_seconds=IMPORT_SECONDS,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    setup_timings={}, feature_events=[])
    save_json(metadata_path, metadata)
    stage = "CUDA verification"
    try:
        print("PRELIMINARY PILOT - single seed 42; no official test evaluation")
        print("Python:", sys.executable)
        print("Results folder:", run_dir)
        if not DATASET_CACHE.is_absolute():
            raise ValueError("HF_DATASETS_CACHE must be a short absolute path.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this pilot. Check the Spyder interpreter/driver.")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.synchronize(DEVICE)
        metadata["gpu"] = torch.cuda.get_device_name(DEVICE)
        metadata["cuda_build"] = torch.version.cuda
        metadata["torch_cpu_threads"] = torch.get_num_threads()

        stage = "training split loading and sampling"
        DATASET_CACHE.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        # Request only train. No test labels/text are used for any pilot decision.
        dataset = load_dataset(DATASET_NAME, split="train", revision=DATASET_REVISION,
                               cache_dir=str(DATASET_CACHE))
        metadata["setup_timings"]["dataset_load_seconds_including_possible_download"] = time.perf_counter() - started
        metadata["dataset_fingerprint"] = dataset._fingerprint
        class_names = list(dataset.features["label"].names)
        # Reading this one small integer column is not full-text preprocessing.
        labels = np.asarray(dataset["label"], dtype=np.int64)
        training_indices, validation_indices = sample_indices(labels)
        validation_texts, validation_labels = select_rows(dataset, validation_indices)
        training_data = {size: select_rows(dataset, training_indices[size]) for size in TRAIN_SIZES}
        manifest = dict(seed=SEED, dataset_revision=DATASET_REVISION,
                        dataset_fingerprint=dataset._fingerprint, source_split="train",
                        class_names=class_names, validation_indices=validation_indices,
                        validation_labels=validation_labels.tolist(),
                        training_indices=training_indices,
                        training_labels={size: training_data[size][1].tolist() for size in TRAIN_SIZES})
        save_json(run_dir / "sample_manifest.json", manifest)
        print("Balanced, disjoint validation: 800 rows; nested training: 100 and 400 rows.")

        # The same saved row lists and validation texts feed BOTH representations.
        # Any difference in scores therefore cannot be attributed to different
        # sampled articles, although representation settings still matter.
        results = []
        for size in TRAIN_SIZES:
            stage = f"TF-IDF experiment with {size} training rows"
            texts, targets = training_data[size]
            train_x, validation_x, vectorizer, train_time, val_time = tfidf_features(texts, validation_texts)
            result = fit_and_evaluate("tfidf", size, train_x, targets, validation_x, validation_labels,
                                      validation_texts, validation_indices, class_names, run_dir)
            result.update(train_feature_seconds=train_time, validation_feature_seconds=val_time,
                          feature_extraction_seconds=train_time + val_time,
                          feature_compute_this_run_seconds=train_time + val_time,
                          feature_cache_read_seconds=0.0, feature_cache_write_seconds=0.0,
                          train_cache_hit=False, validation_cache_hit=False,
                          feature_timing_origin="measured_now")
            results.append(result)

        stage = "frozen MiniLM loading and warmup"
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        torch.cuda.synchronize(DEVICE)
        started = time.perf_counter()
        model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION,
                                    cache_folder=str(MODEL_CACHE), device=DEVICE)
        model.eval()
        # Disabling gradients plus inference_mode means no Transformer parameters
        # are learned. Only the later Logistic Regression coefficients/intercepts
        # are fit from AG News labels. eval() alone would not freeze parameters.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        torch.cuda.synchronize(DEVICE)
        metadata["setup_timings"]["model_load_seconds_including_possible_download"] = time.perf_counter() - started
        metadata["max_sequence_length"] = model.max_seq_length
        if model.get_sentence_embedding_dimension() != EMBEDDING_DIMENSION:
            raise ValueError("The selected MiniLM model must output 384-dimensional vectors.")
        started = time.perf_counter()
        with torch.inference_mode():
            # Warmup initializes CUDA inference paths without reading held-out data.
            model.encode(["News classification pilot warmup."], device=DEVICE,
                         convert_to_numpy=True, show_progress_bar=False)
        torch.cuda.synchronize(DEVICE)
        metadata["setup_timings"]["model_warmup_seconds"] = time.perf_counter() - started
        context = dict(cache_schema=1, model=MODEL_NAME, revision=MODEL_REVISION,
                       dataset_fingerprint=dataset._fingerprint, source_split="train",
                       normalize_embeddings=True, precision="float32", batch_size=BATCH_SIZE,
                       max_sequence_length=model.max_seq_length, device=DEVICE, gpu=metadata["gpu"],
                       cuda_build=metadata["cuda_build"], packages=packages)

        # Validation vectors do not depend on training labels/subset size, so one
        # extraction serves both MiniLM experiments. Its measured time is included
        # in each standalone feature-cost comparison but computed only once here.
        # The two nested training blocks are measured separately (100 + 400 texts)
        # to avoid inventing a 100-row extraction time by prorating the 400-row time.
        stage = "MiniLM validation feature extraction"
        validation_x, validation_event = embedding_block(
            model, validation_texts, validation_indices, context, FORCE_RECOMPUTE_EMBEDDINGS)
        for size in TRAIN_SIZES:
            stage = f"MiniLM experiment with {size} training rows"
            texts, targets = training_data[size]
            train_x, train_event = embedding_block(model, texts, training_indices[size], context,
                                                    FORCE_RECOMPUTE_EMBEDDINGS)
            # On the second experiment, validation is reused directly from memory.
            # Mark this reuse explicitly; do not count its compute twice this run.
            val_event = dict(validation_event)
            if size != TRAIN_SIZES[0]:
                val_event.update(cache_hit=True, compute_this_call_seconds=0.0,
                                 cache_read_seconds=0.0, cache_write_seconds=0.0)
            metadata["feature_events"].append(dict(train_size=size, train=train_event, validation=val_event))
            result = fit_and_evaluate("minilm", size, train_x, targets, validation_x, validation_labels,
                                      validation_texts, validation_indices, class_names, run_dir)
            result.update(train_feature_seconds=train_event["measured_extraction_seconds"],
                          validation_feature_seconds=val_event["measured_extraction_seconds"],
                          feature_extraction_seconds=train_event["measured_extraction_seconds"] + val_event["measured_extraction_seconds"],
                          feature_compute_this_run_seconds=train_event["compute_this_call_seconds"] + val_event["compute_this_call_seconds"],
                          feature_cache_read_seconds=train_event["cache_read_seconds"] + val_event["cache_read_seconds"],
                          feature_cache_write_seconds=train_event["cache_write_seconds"] + val_event["cache_write_seconds"],
                          train_cache_hit=train_event["cache_hit"], validation_cache_hit=val_event["cache_hit"],
                          feature_timing_origin="includes_reused_measurement" if train_event["cache_hit"] or val_event["cache_hit"] else "measured_now")
            results.append(result)

        stage = "saving completed pilot results"
        save_csv(run_dir / "metrics.csv", results, list(results[0]))
        save_json(run_dir / "metrics.json", results)
        save_comparison_figure(results, run_dir / "performance_comparison.png")
        metadata.update(status="complete", finished_at=utc_now(), experiment_count=len(results))
        save_json(metadata_path, metadata)
        print("\nPILOT COMPLETE. Results saved to:", run_dir)
        print("Preliminary single-seed validation results only; no statistical superiority claim.")
        print("Review errors, timing provenance, and regularization before interpreting differences.")
    except Exception as error:
        metadata.update(status="failed", failed_stage=stage,
                        error=f"{type(error).__name__}: {error}", finished_at=utc_now())
        save_json(metadata_path, metadata)
        print(f"\nPILOT FAILED during {stage}: {type(error).__name__}: {error}")
        print("Any partial files belong to a failed run; do not present them as a completed comparison.")
        print("For download/path errors, check connectivity, the short dataset cache, and disk space.")
        print("For CUDA errors, check the interpreter, GPU memory, and driver; keep existing packages.")
        print("For a cache checksum error, set FORCE_RECOMPUTE_EMBEDDINGS=True and rerun.")
        raise


# Safe to import for verification: main() alone downloads, encodes, fits, or writes results.
if __name__ == "__main__":
    main()
