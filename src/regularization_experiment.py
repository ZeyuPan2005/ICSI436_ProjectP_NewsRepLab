"""Round 3: EXPLORATORY Logistic Regression regularization sensitivity.

Regularization discourages excessively large learned coefficients. With few
labeled examples and many features, this can reduce fitting of accidental
patterns in the training data. It trades training fit against model complexity;
too much regularization can also underfit. sklearn implements the standard
multinomial cross-entropy objective with an L2 penalty; we write no custom loss.

C controls inverse regularization strength: C=0.1 is stronger regularization,
C=10 is weaker, and C=1 reproduces our baseline. These factor-of-ten steps are a
small controlled sensitivity probe, not a broad optimization search. Larger C
is not assumed to be better. Different feature dimensions, correlations, and
geometries can make TF-IDF and normalized MiniLM respond differently to C.

Only C changes. We reuse the completed repeated experiment's exact 400-row
training sets for seeds 42/123/2026 and its fixed 800-row validation set. We do
not sample new rows, change preprocessing, update MiniLM, or inspect the test
split. Each feature matrix is built/read once and reused at all three C values.

MiniLM embeddings are read from the established validated cache, without model
loading or GPU computation. Missing or invalid caches cause an explicit failure,
not silent re-encoding. TF-IDF must be refit once per seed because the previous
experiment did not persist its sparse matrices or fitted vectorizers.

The same validation set has already influenced our research decisions. Scores
and any best observed C are therefore EXPLORATORY and potentially selection-biased,
not unbiased final performance. Three seeds support descriptive mean/sample SD
and paired differences, not significance tests. Final-project tuning should use
inner validation/cross-validation within training data, with the official test
split reserved for one final evaluation after choices are fixed.
"""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Import by absolute file path for Spyder, without executing either previous
# experiment's main(). Existing helper source files are never edited.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPEATED_PATH = PROJECT_ROOT / "src" / "repeated_experiment.py"
spec = importlib.util.spec_from_file_location("newsreplab_repeated_helpers", REPEATED_PATH)
repeated = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repeated)
pilot = repeated.pilot

REFERENCE_RUN = PROJECT_ROOT / "results" / "repeated" / "20261019T225846_619676Z"
RESULTS_ROOT = PROJECT_ROOT / "results" / "regularization"
SEEDS = (42, 123, 2026)
REPRESENTATIONS = ("tfidf", "minilm")
C_VALUES = (0.1, 1.0, 10.0)
TRAIN_SIZE = 400
SCORE_METRICS = ("validation_accuracy", "validation_macro_f1")
BASELINE_ATOL = 1e-10  # Absolute score tolerance; rtol=0. One changed label is NOT tolerated.


def classifier_settings(c_value):
    """Return a fresh copy so changing C never mutates the original pilot settings."""
    if not np.isfinite(c_value) or c_value <= 0 or c_value not in C_VALUES:
        raise ValueError("This controlled sweep permits only C=0.1, 1.0, or 10.0.")
    settings = dict(pilot.LR_SETTINGS)
    settings["C"] = float(c_value)
    return settings


def read_reference(dataset, packages):
    """Read the saved samples/results and reject incompatible or incomplete sources."""
    metadata = json.loads((REFERENCE_RUN / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((REFERENCE_RUN / "sample_manifest.json").read_text(encoding="utf-8"))
    baseline_rows = json.loads((REFERENCE_RUN / "individual_results.json").read_text(encoding="utf-8"))
    expected = dict(status="complete", experiment_count=12, seeds=list(SEEDS), fixed_validation=True,
                    dataset=pilot.DATASET_NAME, dataset_revision=pilot.DATASET_REVISION,
                    dataset_fingerprint=dataset._fingerprint, split="train", model=pilot.MODEL_NAME,
                    model_revision=pilot.MODEL_REVISION, classifier_settings=pilot.LR_SETTINGS,
                    tfidf_settings=pilot.TFIDF_SETTINGS, packages=packages)
    for key, value in expected.items():
        if metadata.get(key) != json.loads(json.dumps(value)):
            raise ValueError(f"Repeated reference mismatch: {key}. Keep the established configuration.")
    for key, value in dict(seeds=list(SEEDS), source_split="train", dataset_revision=pilot.DATASET_REVISION,
                           dataset_fingerprint=dataset._fingerprint,
                           class_names=list(dataset.features["label"].names)).items():
        if manifest.get(key) != value:
            raise ValueError(f"Saved sample manifest mismatch: {key}.")
    # The previous aggregation helper already checks all twelve unique cells and
    # score validity. We take only its six 400-example baselines for comparison.
    repeated.summarize_results(baseline_rows)
    baselines = {(row["seed"], row["representation"]): row for row in baseline_rows if row["train_size"] == TRAIN_SIZE}
    if any(row["validation_size"] != 800 for row in baselines.values()):
        raise ValueError("Reference results must use 800 validation examples.")
    labels = np.asarray(dataset["label"], dtype=np.int64)
    validation = manifest["validation_indices"]
    plans = {}
    for seed in SEEDS:
        saved_plan = {size: manifest["training_indices"][str(seed)][str(size)] for size in pilot.TRAIN_SIZES}
        # Reuse the existing row/balance/disjointness checker, but never either
        # sampling function. The saved 100-row set is checked only for provenance.
        repeated.validate_indices(labels, saved_plan, validation)
        for size, indices in saved_plan.items():
            if labels[indices].tolist() != manifest["training_labels"][str(seed)][str(size)]:
                raise ValueError("Saved training labels differ from the pinned dataset.")
        plans[seed] = saved_plan[TRAIN_SIZE]
    if labels[validation].tolist() != manifest["validation_labels"]:
        raise ValueError("Saved validation labels differ from the pinned dataset.")

    expected_context = dict(cache_schema=1, model=pilot.MODEL_NAME, revision=pilot.MODEL_REVISION,
                            dataset_fingerprint=dataset._fingerprint, source_split="train",
                            normalize_embeddings=True, precision="float32", batch_size=pilot.BATCH_SIZE,
                            max_sequence_length=256, device=pilot.DEVICE, gpu=metadata["gpu"],
                            cuda_build=metadata["cuda_build"], packages=packages)
    if metadata["encoding_context"] != expected_context:
        raise ValueError("Reference embedding settings differ from the established MiniLM configuration.")
    events = [event for event in metadata["feature_events"] if event["train_size"] == TRAIN_SIZE]
    if len(events) != 3 or {event["seed"] for event in events} != set(SEEDS):
        raise ValueError("Missing or duplicate 400-example feature-cache records.")
    validation_keys = {event["validation"]["key"] for event in events}
    if len(validation_keys) != 1:
        raise ValueError("All seeds must refer to the same validation embedding cache.")
    keys = {event["seed"]: event["train"]["key"] for event in events}
    return metadata, manifest, baselines, plans, validation, keys, validation_keys.pop()


def cached_embeddings(texts, indices, context, recorded_key):
    """Use the pilot's integrity checks while forbidding any cache-miss encoding."""
    text_hash = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).hexdigest()
    identity = dict(context, indices=list(indices), text_sha256=text_hash)
    expected_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    if recorded_key != expected_key:
        raise ValueError("Recorded embedding key does not match these exact ordered texts/settings.")
    for suffix in (".npy", ".json"):
        path = pilot.EMBEDDING_CACHE / (expected_key + suffix)
        if not path.is_file():
            raise FileNotFoundError(f"Required existing embedding cache is missing: {path}. Restore it before this sweep.")
    # With an existing pair, the helper validates identity, dimensions, numeric
    # values, checksum, and timing metadata. It needs no model on its cache-hit
    # path. Passing None makes our deliberate read-only feature reuse explicit.
    features, event = pilot.embedding_block(None, texts, indices, context, force=False)
    if not event["cache_hit"]:
        raise RuntimeError("This sensitivity experiment must use existing embeddings.")
    return features, event


def fit_one(train_x, targets, validation_x, validation_labels, seed, name, c_value):
    """Fit a new standard LR instance for each C, with identical fixed features."""
    # Each fresh estimator learns its own coefficient matrix and intercepts.
    # No warm start carries weights from another C, and C is the ONLY changed
    # classifier setting. All three sampling seeds retain the pilot's lbfgs
    # random_state=42; lbfgs does not use this option to randomize its fitting.
    classifier = pilot.LogisticRegression(**classifier_settings(c_value))
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", pilot.ConvergenceWarning)
        classifier.fit(train_x, targets)
    training_seconds = time.perf_counter() - started
    # Follow the same probability/argmax decision rule as the prior helper, so
    # our C=1 comparison also checks identical prediction behavior.
    probabilities = classifier.predict_proba(validation_x)
    predictions = classifier.classes_[np.argmax(probabilities, axis=1)]
    result = dict(seed=seed, representation=name, C=float(c_value), train_size=TRAIN_SIZE,
                  validation_size=len(validation_labels), exploratory=True,
                  validation_accuracy=float(pilot.accuracy_score(validation_labels, predictions)),
                  validation_macro_f1=float(pilot.f1_score(validation_labels, predictions,
                                              labels=pilot.CLASS_IDS, average="macro", zero_division=0)),
                  lr_training_seconds=training_seconds, lr_iterations=int(np.max(classifier.n_iter_)),
                  feature_dimension=int(train_x.shape[1]))
    return result, predictions


def baseline_checks(rows, baselines):
    """Compare all six C=1 scores with the saved repeated run, never its timings."""
    selected = [row for row in rows if row["C"] == 1.0]
    expected = {(seed, name) for seed in SEEDS for name in REPRESENTATIONS}
    actual = [(row["seed"], row["representation"]) for row in selected]
    if len(actual) != 6 or set(actual) != expected:
        raise ValueError("Baseline verification requires all six unique C=1.0 fits.")
    checks = []
    for row in selected:
        previous = baselines[row["seed"], row["representation"]]
        for metric in SCORE_METRICS:
            difference = row[metric] - previous[metric]
            checks.append(dict(seed=row["seed"], representation=row["representation"], metric=metric,
                               repeated_baseline=previous[metric], current=row[metric], difference=difference,
                               absolute_tolerance=BASELINE_ATOL,
                               matches=bool(np.isclose(row[metric], previous[metric], atol=BASELINE_ATOL, rtol=0))))
    return checks


def aggregate_results(rows):
    """Calculate sample SD and paired deltas from all 18 complete experiment cells."""
    expected = {(seed, name, c) for seed in SEEDS for name in REPRESENTATIONS for c in C_VALUES}
    actual = [(row["seed"], row["representation"], row["C"]) for row in rows]
    if len(actual) != 18 or set(actual) != expected:
        raise ValueError("Aggregation requires all 18 unique seed/representation/C cells.")
    for row in rows:
        for metric in SCORE_METRICS:
            if not np.isfinite(row[metric]) or not 0 <= row[metric] <= 1:
                raise ValueError("Validation scores must be finite and between zero and one.")
        if not np.isfinite(row["lr_training_seconds"]) or row["lr_training_seconds"] < 0:
            raise ValueError("Training time must be finite and nonnegative.")
    baseline = {(row["seed"], row["representation"]): row for row in rows if row["C"] == 1.0}
    paired = []
    for row in rows:
        delta = dict(seed=row["seed"], representation=row["representation"], C=row["C"])
        for metric in SCORE_METRICS:
            delta[metric + "_difference"] = row[metric] - baseline[row["seed"], row["representation"]][metric]
        paired.append(delta)
    summaries = []
    for name in REPRESENTATIONS:
        for c_value in C_VALUES:
            selected = [row for row in rows if row["representation"] == name and row["C"] == c_value]
            changes = [row for row in paired if row["representation"] == name and row["C"] == c_value]
            summary = dict(representation=name, C=c_value, n_seeds=3, std_ddof=1)
            for metric in (*SCORE_METRICS, "lr_training_seconds"):
                values = [row[metric] for row in selected]
                summary[metric + "_mean"] = float(np.mean(values))
                summary[metric + "_std"] = float(np.std(values, ddof=1))
            for metric in SCORE_METRICS:
                # Pair the same seed against itself before calculating spread.
                # Subtracting two unpaired SDs would not describe paired changes.
                values = [row[metric + "_difference"] for row in changes]
                summary[metric + "_difference_mean"] = float(np.mean(values))
                summary[metric + "_difference_std"] = float(np.std(values, ddof=1))
            summaries.append(summary)
    return summaries, paired


def export_reports(rows, summaries, paired, run_dir):
    """Create descriptive figures, a compact table, and evidence-driven analysis."""
    table = ["# EXPLORATORY regularization sensitivity", "",
             "400 training examples; fixed validation n=800; seeds 42/123/2026. Mean +/- sample SD (ddof=1).", "",
             "| Representation | C | Accuracy (%) | Macro-F1 (%) | LR fit (seconds) | Accuracy delta vs C=1 (pp) | F1 delta vs C=1 (pp) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        table.append(f"| {row['representation']} | {row['C']:g} | "
                     f"{100*row['validation_accuracy_mean']:.2f} +/- {100*row['validation_accuracy_std']:.2f} | "
                     f"{100*row['validation_macro_f1_mean']:.2f} +/- {100*row['validation_macro_f1_std']:.2f} | "
                     f"{row['lr_training_seconds_mean']:.4f} +/- {row['lr_training_seconds_std']:.4f} | "
                     f"{100*row['validation_accuracy_difference_mean']:+.2f} | {100*row['validation_macro_f1_difference_mean']:+.2f} |")
    (run_dir / "results_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    for metric, title, filename in [("validation_accuracy", "Validation accuracy", "accuracy_vs_C.png"),
                                     ("validation_macro_f1", "Validation Macro-F1", "macro_f1_vs_C.png")]:
        figure = Figure(figsize=(8.5, 5), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for name, color in [("tfidf", "#2563eb"), ("minilm", "#d97706")]:
            group = sorted([row for row in summaries if row["representation"] == name], key=lambda row: row["C"])
            axis.errorbar([row["C"] for row in group], [row[metric + "_mean"] for row in group],
                          yerr=[row[metric + "_std"] for row in group], marker="o", capsize=5,
                          color=color, label=name.upper() + " + LR: mean +/- SD")
            for seed in SEEDS:
                individual = sorted([row for row in rows if row["representation"] == name and row["seed"] == seed], key=lambda row: row["C"])
                axis.plot([row["C"] for row in individual], [row[metric] for row in individual],
                          color=color, alpha=0.25, linewidth=1)
        # Log spacing makes 0.1 -> 1 and 1 -> 10 equally spaced factor-of-ten steps.
        axis.set(xscale="log", xlabel="C (inverse regularization strength)", ylabel=title,
                 ylim=(0, 1), title=title)
        axis.set_xticks(C_VALUES, labels=["0.1", "1", "10"])
        axis.grid(alpha=0.25)
        axis.legend()
        # A suptitle participates in constrained layout, keeping explanatory text
        # away from the x-axis label when the saved figure is resized or exported.
        figure.suptitle("EXPLORATORY: n=400, fixed validation n=800, three seeds\n"
                       "Thin lines: individual seeds. SD bars are not confidence intervals.", fontsize=11)
        figure.savefig(run_dir / filename, dpi=160)

    lines = ["# EXPLORATORY regularization sensitivity analysis", "",
             "All 18 fits use saved samples and fixed features. All six C=1.0 baselines matched the prior repeated run within the recorded tolerance.",
             "The validation set was inspected in earlier rounds: best observed scores may be selection-biased and are not unbiased final estimates.", "",
             "## 1. Does C change scores relative to C=1? Are changes consistent across seeds?", ""]
    for name in REPRESENTATIONS:
        for metric in SCORE_METRICS:
            for c_value in (0.1, 10.0):
                group = [row for row in paired if row["representation"] == name and row["C"] == c_value]
                values = np.asarray([row[metric + "_difference"] for row in group])
                wins, ties, losses = int(np.sum(values > 1e-12)), int(np.sum(np.abs(values) <= 1e-12)), int(np.sum(values < -1e-12))
                detail = ", ".join(f"{row['seed']}: {100*row[metric+'_difference']:+.2f} pp" for row in group)
                lines.append(f"- {name}, {metric}, C={c_value:g}: mean change {100*np.mean(values):+.2f} pp "
                             f"(sample SD {100*np.std(values, ddof=1):.2f} pp); {wins}/3 improvements, {ties}/3 ties, {losses}/3 declines. Seeds {detail}.")
    lines.extend(["", "## 2. Do the representations respond differently?", ""])
    for metric in SCORE_METRICS:
        for c_value in (0.1, 10.0):
            response = {}
            for name in REPRESENTATIONS:
                response[name] = next(row[metric + "_difference_mean"] for row in summaries if row["representation"] == name and row["C"] == c_value)
            lines.append(f"- {metric}, C={c_value:g}: TF-IDF mean change {100*response['tfidf']:+.2f} pp; "
                         f"MiniLM {100*response['minilm']:+.2f} pp. MiniLM-minus-TF-IDF change: {100*(response['minilm']-response['tfidf']):+.2f} pp.")
        for name in REPRESENTATIONS:
            group = [row for row in summaries if row["representation"] == name]
            maximum = max(row[metric + "_mean"] for row in group)
            best = [row["C"] for row in group if abs(row[metric + "_mean"] - maximum) <= 1e-12]
            lines.append(f"- {name}, {metric}: best observed mean {100*maximum:.2f}% at C={best}. Descriptive validation ranking only, not final optimization.")
    lines.extend(["", "## 3. Does MiniLM still outperform TF-IDF across tested C values?", ""])
    index = {(row["seed"], row["representation"], row["C"]): row for row in rows}
    for metric in SCORE_METRICS:
        for c_value in C_VALUES:
            changes = [index[seed, "minilm", c_value][metric] - index[seed, "tfidf", c_value][metric] for seed in SEEDS]
            wins = sum(value > 1e-12 for value in changes)
            ties = sum(abs(value) <= 1e-12 for value in changes)
            lines.append(f"- {metric}, C={c_value:g}: {wins}/3 MiniLM wins, {ties}/3 ties, {3-wins-ties}/3 losses; "
                         f"mean MiniLM advantage {100*np.mean(changes):+.2f} pp. "
                         f"{'Advantage holds in all three seeds at this C.' if wins == 3 else 'Advantage is not uniform at this C.'}")
    lines.extend(["", "## 4. Limits and final-project planning", "",
                  "- This is an exploratory three-value sensitivity probe. Larger C is not inherently better, and no model is declared fully optimized.",
                  "- Three seeds and a shared, previously inspected validation set do not support significance claims. SD is descriptive, not a confidence interval.",
                  "- For the final project, predefine a selection metric and use inner validation or stratified cross-validation within training data to choose C separately for each representation. Fit TF-IDF inside each inner training fold to prevent leakage.",
                  "- Freeze the chosen protocol before a final evaluation on the untouched official AG News test split. The best score on this explored validation set must not be presented as unbiased final performance.",
                  "- Investigate additional labeled-data budgets and seeds, representation-specific C ranges justified by this probe, calibration/error patterns, and variability across inner splits. Keep budgets and preprocessing comparable.",
                  "- LR fit times exclude fixed-feature preparation and cache reads. Single-process timing noise and fixed experiment order limit runtime conclusions.", ""])
    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    """Prepare fixed features once, verify C=1 first, then complete the 18-cell sweep."""
    run_dir = RESULTS_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = run_dir / "run_metadata.json"
    packages = {name: pilot.version(name) for name in ["numpy", "torch", "datasets", "sentence-transformers",
                                                       "transformers", "scikit-learn", "matplotlib"]}
    metadata = dict(status="running", exploratory=True, started_at=pilot.utc_now(), reference_run=str(REFERENCE_RUN),
                    seeds=list(SEEDS), C_values=list(C_VALUES), train_size=TRAIN_SIZE, validation_size=800,
                    classifier_settings_by_C={str(c): classifier_settings(c) for c in C_VALUES},
                    baseline_absolute_tolerance=BASELINE_ATOL, baseline_relative_tolerance=0,
                    packages=packages, python=sys.version, executable=sys.executable, platform=platform.platform(),
                    dataset=pilot.DATASET_NAME, dataset_revision=pilot.DATASET_REVISION, source_split="train",
                    model=pilot.MODEL_NAME, model_revision=pilot.MODEL_REVISION,
                    embedding_mode="existing cache only; no model loading or inference",
                    feature_preparation=[], setup_timings={}, std_ddof=1,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    repeated_helpers_sha256=hashlib.sha256(REPEATED_PATH.read_bytes()).hexdigest(),
                    pilot_helpers_sha256=hashlib.sha256(repeated.PILOT_PATH.read_bytes()).hexdigest())
    pilot.save_json(metadata_path, metadata)
    stage = "reading the pinned repeated experiment"
    try:
        print("EXPLORATORY C SENSITIVITY: C=0.1, 1, 10; n=400; three existing training subsets.")
        print("Output folder:", run_dir)
        if not pilot.DATASET_CACHE.is_absolute():
            raise ValueError("HF_DATASETS_CACHE must be a short absolute path.")
        started = time.perf_counter()
        dataset = pilot.load_dataset(pilot.DATASET_NAME, split="train", revision=pilot.DATASET_REVISION,
                                     cache_dir=str(pilot.DATASET_CACHE))
        metadata["setup_timings"]["dataset_load_seconds"] = time.perf_counter() - started
        reference, manifest, baselines, plans, validation, keys, val_key = read_reference(dataset, packages)
        metadata["reference_file_hashes"] = {name: hashlib.sha256((REFERENCE_RUN / name).read_bytes()).hexdigest()
                                            for name in ["run_metadata.json", "sample_manifest.json", "individual_results.json"]}
        metadata["dataset_fingerprint"] = dataset._fingerprint
        metadata["tfidf_settings"] = pilot.TFIDF_SETTINGS
        metadata["encoding_context"] = reference["encoding_context"]
        # Preserve the source manifest verbatim as evidence. We use only its 400
        # training rows per seed; its 100-row entries are not new experiment cells.
        (run_dir / "source_sample_manifest.json").write_bytes((REFERENCE_RUN / "sample_manifest.json").read_bytes())
        validation_texts, validation_labels = pilot.select_rows(dataset, validation)
        stage = "validating existing embeddings and fixed TF-IDF features"
        val_embeddings, event = cached_embeddings(validation_texts, validation, reference["encoding_context"], val_key)
        metadata["feature_preparation"].append(dict(role="shared_validation_embeddings", **event))
        features = {}
        targets_by_seed = {}
        for seed in SEEDS:
            texts, targets = pilot.select_rows(dataset, plans[seed])
            targets_by_seed[seed] = targets
            train_x, val_x, _, train_time, val_time = pilot.tfidf_features(texts, validation_texts)
            features[seed, "tfidf"] = (train_x, val_x)
            metadata["feature_preparation"].append(dict(seed=seed, role="tfidf", train_seconds=train_time, validation_seconds=val_time))
            embeddings, event = cached_embeddings(texts, plans[seed], reference["encoding_context"], keys[seed])
            features[seed, "minilm"] = (embeddings, val_embeddings)
            metadata["feature_preparation"].append(dict(seed=seed, role="training_embeddings", **event))
        results = []
        # Run C=1 for ALL representations/seeds before exploring anything else.
        # If reproduction fails, stop and retain a discrepancy report for review.
        for c_value in (1.0, 0.1, 10.0):
            for seed in SEEDS:
                for name in REPRESENTATIONS:
                    stage = f"fitting seed {seed}, {name}, C={c_value:g}"
                    train_x, val_x = features[seed, name]
                    result, predictions = fit_one(train_x, targets_by_seed[seed], val_x, validation_labels, seed, name, c_value)
                    results.append(result)
                    prediction_rows = [dict(dataset_row=row_id, true_label=int(validation_labels[i]), predicted_label=int(predictions[i]))
                                       for i, row_id in enumerate(validation)]
                    tag = str(c_value).replace(".", "p")
                    pilot.save_csv(run_dir / f"predictions_{name}_seed{seed}_C{tag}.csv", prediction_rows, list(prediction_rows[0]))
                    print(f"{name}, seed={seed}, C={c_value:g}: accuracy={result['validation_accuracy']:.4f}, macro-F1={result['validation_macro_f1']:.4f}")
            pilot.save_json(run_dir / "individual_results.json", results)
            if c_value == 1.0:
                stage = "verifying C=1 reproduces the repeated baseline"
                checks = baseline_checks(results, baselines)
                pilot.save_csv(run_dir / "baseline_reproduction.csv", checks, list(checks[0]))
                pilot.save_json(run_dir / "baseline_reproduction.json", checks)
                if not all(check["matches"] for check in checks):
                    raise ValueError("C=1 baseline mismatch. See baseline_reproduction.csv; other C values were not run.")
                metadata["baseline_reproduction_passed"] = True
                pilot.save_json(metadata_path, metadata)
                print("All six C=1 baseline fits match the repeated results within tolerance.")
        stage = "aggregating and exporting exploratory results"
        summaries, paired = aggregate_results(results)
        pilot.save_csv(run_dir / "individual_results.csv", results, list(results[0]))
        pilot.save_csv(run_dir / "summary.csv", summaries, list(summaries[0]))
        pilot.save_json(run_dir / "summary.json", summaries)
        pilot.save_csv(run_dir / "paired_differences.csv", paired, list(paired[0]))
        pilot.save_json(run_dir / "paired_differences.json", paired)
        export_reports(results, summaries, paired, run_dir)
        metadata.update(status="complete", finished_at=pilot.utc_now(), experiment_count=len(results))
        pilot.save_json(metadata_path, metadata)
        print("\nEXPLORATORY REGULARIZATION SWEEP COMPLETE:", run_dir)
        print("Read analysis.md and baseline_reproduction.csv before interpreting the sweep.")
    except Exception as error:
        metadata.update(status="failed", failed_stage=stage, error=f"{type(error).__name__}: {error}", finished_at=pilot.utc_now())
        pilot.save_json(metadata_path, metadata)
        print(f"REGULARIZATION SWEEP FAILED during {stage}: {error}")
        print("Keep the traceback. Review reference files/caches and baseline reproduction; no package changes are made.")
        print("Partial outputs are not a completed sensitivity study.")
        raise


if __name__ == "__main__":
    main()
