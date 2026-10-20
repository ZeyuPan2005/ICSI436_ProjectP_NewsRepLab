"""NewsRepLab: three training-subset seeds on ONE fixed validation set.

The completed seed-42 pilot motivates a stability check, not a new model search.
We keep its dataset, preprocessing, frozen MiniLM, Logistic Regression settings,
and exact ordered validation rows. Seeds 123 and 2026 change only the choice and
order of training articles. Seed 42 reuses the pilot's original training rows.

A seed initializes a pseudorandom number generator. Reusing it with the same
data and sampling procedure reproduces the chosen rows. Different seeds can
select easier or harder examples, causing different learned classifiers and
validation scores. Holding validation fixed helps isolate this training-sample
variability; it does NOT measure uncertainty from choosing a new validation set.

Each seed uses balanced nested training sets (100 within 400), identical across
representations, with no validation overlap. The original test split is not used.
The shared lbfgs classifier's random_state remains 42, as in the pilot; lbfgs
does not use that option. The experimental seeds describe training-row sampling.
MiniLM remains in frozen evaluation mode, with no seed-dependent training.

Import the existing pilot helpers without running its main(), so vectorization,
encoding, cache validation, classifier fitting, and diagnostics stay identical.
New results go to results/repeated/<timestamp>/; previous runs are never edited.
Analysis is generated only after all twelve real experiments finish. Three seeds
provide a descriptive stability check, not a statistical superiority proof.
"""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# An absolute file import works in Spyder regardless of its working directory,
# and reloads the helpers if this script is rerun in an already-open kernel.
# Keep this small script-to-script dependency explicit: importing the module
# defines helpers, while its __main__ guard prevents an accidental pilot run.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = PROJECT_ROOT / "src" / "pilot_experiment.py"
spec = importlib.util.spec_from_file_location("newsreplab_pilot_helpers", PILOT_PATH)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

SEEDS = (42, 123, 2026)
REPRESENTATIONS = ("tfidf", "minilm")
METRICS = ("validation_accuracy", "validation_macro_f1")
RESULTS_ROOT = PROJECT_ROOT / "results" / "repeated"
# Pin the completed pilot explicitly: selecting the newest folder automatically
# could silently change the validation set after a later unrelated experiment.
REFERENCE_PILOT_RUN = PROJECT_ROOT / "results" / "pilot" / "20261019T223707_829532Z"
FORCE_RECOMPUTE_EMBEDDINGS = False
# This is a descriptive reporting convention, NOT a tuned model parameter,
# equivalence margin established by a test, or proof of performance saturation.
LIMITED_GAIN_THRESHOLD = 0.01  # 0.01 on a 0..1 score scale = one percentage point.


def validate_indices(
    labels: np.ndarray, training: dict[int, list[int]], validation: list[int]
) -> None:
    """Check row identity, class counts, nesting, and leakage before fitting."""

    def check_group(indices, count, per_class):
        array = np.asarray(indices)
        if array.shape != (count,) or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("Sample row IDs must be a correctly sized integer list.")
        if (
            len(set(indices)) != count
            or np.any(array < 0)
            or np.any(array >= len(labels))
        ):
            raise ValueError(
                "Sample row IDs are duplicated or outside the training split."
            )
        if not np.array_equal(np.bincount(labels[array], minlength=4), [per_class] * 4):
            raise ValueError("Selected rows do not have the required class balance.")

    check_group(validation, 800, 200)
    for size in pilot.TRAIN_SIZES:
        check_group(training[size], size, size // 4)
        if set(training[size]) & set(validation):
            raise ValueError("Training and validation rows overlap.")
    if not set(training[100]).issubset(training[400]):
        raise ValueError("Training subsets are not nested.")


def load_reference(labels, fingerprint, class_names):
    """Require a completed, compatible pilot and reuse its exact saved row order."""
    metadata = json.loads(
        (REFERENCE_PILOT_RUN / "run_metadata.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (REFERENCE_PILOT_RUN / "sample_manifest.json").read_text(encoding="utf-8")
    )
    expected = dict(
        status="complete",
        seed=42,
        dataset=pilot.DATASET_NAME,
        dataset_revision=pilot.DATASET_REVISION,
        dataset_fingerprint=fingerprint,
        split="train",
        model=pilot.MODEL_NAME,
        model_revision=pilot.MODEL_REVISION,
        classifier_settings=pilot.LR_SETTINGS,
        tfidf_settings=pilot.TFIDF_SETTINGS,
    )
    for key, value in expected.items():
        # JSON normalization makes an in-code tuple and its saved JSON list equal.
        if metadata.get(key) != json.loads(json.dumps(value)):
            raise ValueError(f"Reference pilot is incompatible: {key}.")
    for key, value in dict(
        seed=42,
        dataset_revision=pilot.DATASET_REVISION,
        dataset_fingerprint=fingerprint,
        source_split="train",
        class_names=class_names,
    ).items():
        if manifest.get(key) != value:
            raise ValueError(f"Reference sample manifest is incompatible: {key}.")
    training = {
        size: manifest["training_indices"][str(size)] for size in pilot.TRAIN_SIZES
    }
    validation = manifest["validation_indices"]
    validate_indices(labels, training, validation)
    if manifest["validation_labels"] != labels[validation].tolist():
        raise ValueError("Reference validation labels differ from the current dataset.")
    for size in pilot.TRAIN_SIZES:
        if manifest["training_labels"][str(size)] != labels[training[size]].tolist():
            raise ValueError(
                "Reference training labels differ from the current dataset."
            )
    return training, validation


def sample_training(
    labels: np.ndarray, validation: list[int], seed: int
) -> dict[int, list[int]]:
    """Draw new nested class-balanced training sets from outside fixed validation."""
    # Do not call the pilot's sampling function with a different seed: that would
    # redraw validation too. Remove the fixed validation rows BEFORE sampling.
    rng = np.random.default_rng(seed)
    eligible = np.ones(len(labels), dtype=bool)
    eligible[validation] = False
    training = {size: [] for size in pilot.TRAIN_SIZES}
    for class_id in pilot.CLASS_IDS:
        candidates = rng.permutation(np.flatnonzero(eligible & (labels == class_id)))
        if len(candidates) < 100:
            raise ValueError(f"Too few eligible training rows for class {class_id}.")
        for size in pilot.TRAIN_SIZES:
            training[size].extend(candidates[: size // 4].tolist())
    for size in pilot.TRAIN_SIZES:
        training[size] = rng.permutation(training[size]).tolist()
    validate_indices(labels, training, validation)
    return training


def summarize_results(rows: list[dict]) -> list[dict]:
    """Reduce 12 individual result records to four mean/sample-SD records.

    Pairing is preserved in the separate paired_differences helper. A standard
    deviation across training seeds is descriptive spread, not uncertainty
    estimated from independent validation datasets or a significance test.
    """
    expected = {
        (seed, name, size)
        for seed in SEEDS
        for name in REPRESENTATIONS
        for size in pilot.TRAIN_SIZES
    }
    actual = [(row["seed"], row["representation"], row["train_size"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(
            "Summary requires all 12 unique experiments; partial/duplicate results are invalid."
        )
    summaries = []
    for name in REPRESENTATIONS:
        for size in pilot.TRAIN_SIZES:
            selected = [
                row
                for row in rows
                if row["representation"] == name and row["train_size"] == size
            ]
            summary = dict(
                representation=name, train_size=size, seeds=len(SEEDS), std_ddof=1
            )
            for metric in METRICS:
                values = np.asarray([row[metric] for row in selected], dtype=float)
                if (
                    not np.isfinite(values).all()
                    or np.any(values < 0)
                    or np.any(values > 1)
                ):
                    raise ValueError(
                        "Accuracy and Macro-F1 must be finite values between 0 and 1."
                    )
                # ddof=1 gives the sample standard deviation (denominator n-1).
                # It describes observed spread across three training subsets,
                # not standard error, confidence intervals, or test-set uncertainty.
                summary[metric + "_mean"] = float(np.mean(values))
                summary[metric + "_std"] = float(np.std(values, ddof=1))
                summary[metric + "_min"] = float(np.min(values))
                summary[metric + "_max"] = float(np.max(values))
            summaries.append(summary)
    return summaries


def paired_differences(rows: list[dict]) -> list[dict]:
    """Compare matched seeds instead of comparing unrelated best/worst runs."""
    scores = {
        (row["seed"], row["representation"], row["train_size"]): row for row in rows
    }
    differences = []
    for seed in SEEDS:
        for metric in METRICS:
            for size in pilot.TRAIN_SIZES:
                differences.append(
                    dict(
                        seed=seed,
                        metric=metric,
                        contrast="minilm_minus_tfidf",
                        train_size=size,
                        representation="both",
                        difference=scores[seed, "minilm", size][metric]
                        - scores[seed, "tfidf", size][metric],
                    )
                )
            for name in REPRESENTATIONS:
                differences.append(
                    dict(
                        seed=seed,
                        metric=metric,
                        contrast="400_minus_100",
                        train_size="",
                        representation=name,
                        difference=scores[seed, name, 400][metric]
                        - scores[seed, name, 100][metric],
                    )
                )
    return differences


def write_analysis(rows, summaries, path):
    """Answer the five research questions from actual results, including counterexamples."""
    differences = paired_differences(rows)
    lines = [
        "# Repeated baseline analysis",
        "",
        "Preliminary results for seeds 42, 123, and 2026 on the same 800 validation rows.",
        "All scores below come from this completed run; the official test split was not evaluated.",
        "",
        "## 1. Does MiniLM consistently outperform TF-IDF?",
        "",
    ]
    for metric in METRICS:
        for size in pilot.TRAIN_SIZES:
            selected = [
                row
                for row in differences
                if row["metric"] == metric
                and row["contrast"] == "minilm_minus_tfidf"
                and row["train_size"] == size
            ]
            gains = np.asarray([row["difference"] for row in selected])
            wins = int(np.sum(gains > 1e-12))
            ties = int(np.sum(np.abs(gains) <= 1e-12))
            details = ", ".join(
                f"seed {row['seed']}: {100 * row['difference']:+.2f} pp"
                for row in selected
            )
            lines.append(
                f"- {metric}, n={size}: {'yes, across these three seeds' if wins == 3 else 'not consistently across these three seeds'}; "
                f"{wins}/3 wins, {ties}/3 ties, {3-wins-ties}/3 losses. MiniLM minus TF-IDF: {details}."
            )
    lines.extend(["", "## 2. Does TF-IDF consistently improve from 100 to 400?", ""])
    for metric in METRICS:
        selected = [
            row
            for row in differences
            if row["metric"] == metric
            and row["contrast"] == "400_minus_100"
            and row["representation"] == "tfidf"
        ]
        improved = sum(row["difference"] > 1e-12 for row in selected)
        details = ", ".join(
            f"seed {row['seed']}: {100 * row['difference']:+.2f} pp" for row in selected
        )
        lines.append(
            f"- {metric}: {'yes' if improved == 3 else 'not consistently'}; strictly improves in {improved}/3 seeds. {details}."
        )
    threshold_pp = 100 * LIMITED_GAIN_THRESHOLD
    lines.extend(
        [
            "",
            "## 3. Is MiniLM's improvement from 100 to 400 limited?",
            "",
            f"Descriptive convention: a nonnegative gain of at most {threshold_pp:.2f} percentage points is called limited. "
            "This threshold is not a significance or equivalence test; decreases are reported separately.",
        ]
    )
    for metric in METRICS:
        selected = [
            row
            for row in differences
            if row["metric"] == metric
            and row["contrast"] == "400_minus_100"
            and row["representation"] == "minilm"
        ]
        gains = np.asarray([row["difference"] for row in selected])
        limited = int(
            np.sum((gains >= -1e-12) & (gains <= LIMITED_GAIN_THRESHOLD + 1e-12))
        )
        decreases = int(np.sum(gains < -1e-12))
        details = ", ".join(
            f"seed {row['seed']}: {100 * row['difference']:+.2f} pp" for row in selected
        )
        lines.append(
            f"- {metric}: limited nonnegative gains in {limited}/3 seeds, decreases in {decreases}/3, "
            f"larger gains in {3-limited-decreases}/3; mean change {100 * np.mean(gains):+.2f} pp. {details}."
        )
    lines.extend(
        [
            "",
            "## 4. How much do scores vary across seeds?",
            "",
            "Mean scores are percentages; SD and ranges are in percentage points. SD is sample SD (ddof=1).",
            "",
            "| Representation | Training n | Metric | Mean (%) | SD (pp) | Min-max (%) |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for summary in summaries:
        for metric in METRICS:
            lines.append(
                f"| {summary['representation']} | {summary['train_size']} | {metric} | "
                f"{100 * summary[metric+'_mean']:.2f} | {100 * summary[metric+'_std']:.2f} | "
                f"{100 * summary[metric+'_min']:.2f}-{100 * summary[metric+'_max']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## 5. What remains uncertain?",
            "",
            "- Three seeds provide only a small descriptive sample. SD estimates themselves are unstable; no significance claim is justified.",
            "- Validation is fixed, so these results do not quantify variability across validation splits or guarantee performance on unseen test data.",
            "- Training subsets may overlap across seeds, and sizes are nested. Results are paired/correlated, not independent new validation datasets.",
            "- C=1.0 and all preprocessing settings remain untuned. This check does not establish each representation's best attainable performance.",
            "- Two training sizes cannot establish a plateau or a general sample-efficiency curve. Small gains may reflect this validation set or settings.",
            "- Repeated inspection of the same validation set can inform later choices; keep final test evaluation untouched until the protocol is fixed.",
            "- Cached feature times may be historical measurements; cache reads are not new encoding benchmarks. Hardware and numerical effects remain possible.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return differences


def save_figures(rows: list[dict], summaries: list[dict], directory: Path) -> None:
    """Show both mean +/- sample SD and individual seed trajectories."""
    colors = {"tfidf": "#2563eb", "minilm": "#d97706"}
    figure = Figure(figsize=(10, 4.6), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, metric in zip(figure.subplots(1, 2), METRICS):
        for name in REPRESENTATIONS:
            selected = sorted(
                [row for row in summaries if row["representation"] == name],
                key=lambda row: row["train_size"],
            )
            axis.errorbar(
                [row["train_size"] for row in selected],
                [row[metric + "_mean"] for row in selected],
                yerr=[row[metric + "_std"] for row in selected],
                marker="o",
                capsize=5,
                color=colors[name],
                label=name.upper() + " + LR",
            )
        axis.set(
            title=metric.replace("validation_", "").replace("_", " "),
            xlabel="Training examples",
            ylabel="Validation score",
            ylim=(0, 1),
        )
        axis.set_xticks(pilot.TRAIN_SIZES)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(
        "Three seeds: mean +/- sample SD (not a confidence interval); fixed validation n=800"
    )
    figure.savefig(Path(directory) / "mean_std_comparison.png", dpi=160)

    figure = Figure(figsize=(12, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    for column, seed in enumerate(SEEDS):
        for row_number, metric in enumerate(METRICS):
            axis = axes[row_number, column]
            for name in REPRESENTATIONS:
                selected = sorted(
                    [
                        row
                        for row in rows
                        if row["seed"] == seed and row["representation"] == name
                    ],
                    key=lambda row: row["train_size"],
                )
                axis.plot(
                    [row["train_size"] for row in selected],
                    [row[metric] for row in selected],
                    marker="o",
                    color=colors[name],
                    label=name.upper() + " + LR",
                )
            axis.set(
                title=f"Seed {seed}: {metric.replace('validation_', '')}",
                xlabel="Training examples",
                ylabel="Validation score",
                ylim=(0, 1),
            )
            axis.set_xticks(pilot.TRAIN_SIZES)
            axis.grid(alpha=0.25)
            axis.legend()
    figure.suptitle(
        "Individual training-subset seeds; identical fixed validation rows; preliminary"
    )
    figure.savefig(Path(directory) / "individual_seed_comparison.png", dpi=160)


def main() -> None:
    """Run twelve classifiers without changing the pilot or any prior result files."""
    run_dir = RESULTS_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = run_dir / "run_metadata.json"
    packages = {
        name: pilot.version(name)
        for name in [
            "numpy",
            "torch",
            "datasets",
            "sentence-transformers",
            "transformers",
            "scikit-learn",
            "matplotlib",
        ]
    }
    metadata = dict(
        status="running",
        started_at=pilot.utc_now(),
        seeds=list(SEEDS),
        reference_pilot_run=str(REFERENCE_PILOT_RUN),
        fixed_validation=True,
        classifier_settings=pilot.LR_SETTINGS,
        tfidf_settings=pilot.TFIDF_SETTINGS,
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        packages=packages,
        dataset=pilot.DATASET_NAME,
        dataset_revision=pilot.DATASET_REVISION,
        split="train",
        model=pilot.MODEL_NAME,
        model_revision=pilot.MODEL_REVISION,
        dataset_cache=str(pilot.DATASET_CACHE),
        embedding_cache=str(pilot.EMBEDDING_CACHE),
        setup_timings={},
        feature_events=[],
        std_ddof=1,
        limited_gain_threshold=LIMITED_GAIN_THRESHOLD,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pilot_helpers_sha256=hashlib.sha256(PILOT_PATH.read_bytes()).hexdigest(),
    )
    pilot.save_json(metadata_path, metadata)
    stage = "loading and validating the fixed sampling plan"
    try:
        if not pilot.DATASET_CACHE.is_absolute():
            raise ValueError("HF_DATASETS_CACHE must be a short absolute path.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for new MiniLM features; check the existing Spyder interpreter."
            )
        print(
            "REPEATED BASELINES: seeds 42, 123, 2026; fixed pilot validation; no tuning/test evaluation."
        )
        print("Output folder:", run_dir)
        pilot.DATASET_CACHE.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        dataset = pilot.load_dataset(
            pilot.DATASET_NAME,
            split="train",
            revision=pilot.DATASET_REVISION,
            cache_dir=str(pilot.DATASET_CACHE),
        )
        metadata["setup_timings"][
            "dataset_load_seconds_including_possible_download"
        ] = (time.perf_counter() - started)
        labels = np.asarray(dataset["label"], dtype=np.int64)
        class_names = list(dataset.features["label"].names)
        original_training, validation_indices = load_reference(
            labels, dataset._fingerprint, class_names
        )
        plans = {42: original_training}
        for seed in SEEDS[1:]:
            plans[seed] = sample_training(labels, validation_indices, seed)
        validation_texts, validation_labels = pilot.select_rows(
            dataset, validation_indices
        )
        metadata["dataset_fingerprint"] = dataset._fingerprint
        metadata["reference_manifest_sha256"] = hashlib.sha256(
            (REFERENCE_PILOT_RUN / "sample_manifest.json").read_bytes()
        ).hexdigest()
        pilot.save_json(
            run_dir / "sample_manifest.json",
            dict(
                seeds=list(SEEDS),
                reference_pilot_run=str(REFERENCE_PILOT_RUN),
                source_split="train",
                dataset_revision=pilot.DATASET_REVISION,
                dataset_fingerprint=dataset._fingerprint,
                class_names=class_names,
                validation_indices=validation_indices,
                validation_labels=validation_labels.tolist(),
                training_indices=plans,
                training_labels={
                    seed: {
                        size: labels[indices].tolist() for size, indices in plan.items()
                    }
                    for seed, plan in plans.items()
                },
            ),
        )

        stage = "loading the unchanged frozen MiniLM model"
        # Keep the frozen encoder setup identical to the pilot. The seed variation
        # is in sampled labels/texts, not in pretrained model weights or dropout.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        metadata["gpu"] = torch.cuda.get_device_name(pilot.DEVICE)
        metadata["cuda_build"] = torch.version.cuda
        pilot.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        torch.cuda.synchronize(pilot.DEVICE)
        started = time.perf_counter()
        model = pilot.SentenceTransformer(
            pilot.MODEL_NAME,
            revision=pilot.MODEL_REVISION,
            cache_folder=str(pilot.MODEL_CACHE),
            device=pilot.DEVICE,
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        torch.cuda.synchronize(pilot.DEVICE)
        metadata["setup_timings"]["model_load_seconds_including_possible_download"] = (
            time.perf_counter() - started
        )
        if model.get_sentence_embedding_dimension() != pilot.EMBEDDING_DIMENSION:
            raise ValueError("MiniLM must produce 384-dimensional embeddings.")
        started = time.perf_counter()
        with torch.inference_mode():
            model.encode(
                ["News classification pilot warmup."],
                device=pilot.DEVICE,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        torch.cuda.synchronize(pilot.DEVICE)
        metadata["setup_timings"]["model_warmup_seconds"] = (
            time.perf_counter() - started
        )
        # Match the pilot's cache identity exactly. Do not add the sampling seed:
        # frozen features depend on ordered texts and encoder settings, not labels.
        context = dict(
            cache_schema=1,
            model=pilot.MODEL_NAME,
            revision=pilot.MODEL_REVISION,
            dataset_fingerprint=dataset._fingerprint,
            source_split="train",
            normalize_embeddings=True,
            precision="float32",
            batch_size=pilot.BATCH_SIZE,
            max_sequence_length=model.max_seq_length,
            device=pilot.DEVICE,
            gpu=metadata["gpu"],
            cuda_build=metadata["cuda_build"],
            packages=packages,
        )
        metadata["encoding_context"] = context
        stage = "fixed validation embedding retrieval"
        validation_x, validation_event = pilot.embedding_block(
            model,
            validation_texts,
            validation_indices,
            context,
            FORCE_RECOMPUTE_EMBEDDINGS,
        )
        results = []
        for seed in SEEDS:
            seed_dir = run_dir / f"seed_{seed}"
            seed_dir.mkdir()
            print(f"\nTraining-subset seed {seed}")
            seed_rows = []
            for size in pilot.TRAIN_SIZES:
                texts, targets = pilot.select_rows(dataset, plans[seed][size])
                for name in REPRESENTATIONS:
                    stage = f"seed {seed}, {name}, n={size}"
                    if name == "tfidf":
                        train_x, val_x, _, train_time, val_time = pilot.tfidf_features(
                            texts, validation_texts
                        )
                        timing = dict(
                            train_feature_seconds=train_time,
                            validation_feature_seconds=val_time,
                            feature_extraction_seconds=train_time + val_time,
                            feature_compute_this_run_seconds=train_time + val_time,
                            feature_cache_read_seconds=0.0,
                            feature_cache_write_seconds=0.0,
                            train_cache_hit=False,
                            validation_cache_hit=False,
                            feature_timing_origin="measured_now",
                        )
                    else:
                        train_x, event = pilot.embedding_block(
                            model,
                            texts,
                            plans[seed][size],
                            context,
                            FORCE_RECOMPUTE_EMBEDDINGS,
                        )
                        val_x = validation_x
                        val_event = dict(validation_event)
                        if (seed, size) != (SEEDS[0], pilot.TRAIN_SIZES[0]):
                            val_event.update(
                                cache_hit=True,
                                compute_this_call_seconds=0.0,
                                cache_read_seconds=0.0,
                                cache_write_seconds=0.0,
                            )
                        metadata["feature_events"].append(
                            dict(
                                seed=seed,
                                train_size=size,
                                train=event,
                                validation=val_event,
                            )
                        )
                        timing = dict(
                            train_feature_seconds=event["measured_extraction_seconds"],
                            validation_feature_seconds=val_event[
                                "measured_extraction_seconds"
                            ],
                            feature_extraction_seconds=event[
                                "measured_extraction_seconds"
                            ]
                            + val_event["measured_extraction_seconds"],
                            feature_compute_this_run_seconds=event[
                                "compute_this_call_seconds"
                            ]
                            + val_event["compute_this_call_seconds"],
                            feature_cache_read_seconds=event["cache_read_seconds"]
                            + val_event["cache_read_seconds"],
                            feature_cache_write_seconds=event["cache_write_seconds"]
                            + val_event["cache_write_seconds"],
                            train_cache_hit=event["cache_hit"],
                            validation_cache_hit=val_event["cache_hit"],
                            feature_timing_origin=(
                                "includes_reused_measurement"
                                if event["cache_hit"] or val_event["cache_hit"]
                                else "measured_now"
                            ),
                        )
                    result = pilot.fit_and_evaluate(
                        name,
                        size,
                        train_x,
                        targets,
                        val_x,
                        validation_labels,
                        validation_texts,
                        validation_indices,
                        class_names,
                        seed_dir,
                    )
                    # The helper uses the original classifier unchanged. Replace
                    # only its output's sampling-seed annotation for this run.
                    result["seed"] = seed
                    result.update(timing)
                    seed_rows.append(result)
                    results.append(result)
            pilot.save_csv(seed_dir / "metrics.csv", seed_rows, list(seed_rows[0]))
            pilot.save_json(seed_dir / "metrics.json", seed_rows)
            metadata.setdefault("completed_seeds", []).append(seed)
            pilot.save_json(metadata_path, metadata)

        stage = "aggregating all twelve experiments"
        summaries = summarize_results(results)
        pilot.save_csv(run_dir / "individual_results.csv", results, list(results[0]))
        pilot.save_json(run_dir / "individual_results.json", results)
        pilot.save_csv(run_dir / "summary.csv", summaries, list(summaries[0]))
        pilot.save_json(run_dir / "summary.json", summaries)
        differences = write_analysis(results, summaries, run_dir / "analysis.md")
        pilot.save_csv(
            run_dir / "paired_differences.csv", differences, list(differences[0])
        )
        save_figures(results, summaries, run_dir)
        metadata.update(
            status="complete",
            finished_at=pilot.utc_now(),
            experiment_count=len(results),
        )
        pilot.save_json(metadata_path, metadata)
        print(
            "\nREPEATED BASELINES COMPLETE. Read analysis.md and summary.csv in:",
            run_dir,
        )
        print(
            "Three-seed descriptive stability only; SD bars are not confidence intervals."
        )
    except Exception as error:
        metadata.update(
            status="failed",
            failed_stage=stage,
            error=f"{type(error).__name__}: {error}",
            finished_at=pilot.utc_now(),
        )
        pilot.save_json(metadata_path, metadata)
        print(f"REPEATED BASELINES FAILED during {stage}: {error}")
        print(
            "Keep the traceback. Partial files are not a completed three-seed comparison."
        )
        print(
            "Check the reference pilot files, cache permissions, connectivity, or existing CUDA environment as appropriate."
        )
        raise


if __name__ == "__main__":
    main()
