"""Lightweight offline checks; synthetic values are never project research results."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from datasets import ClassLabel, Dataset, Features, Value
from test_pilot_experiment import FakeEncoder

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "regularization_experiment.py"
SPEC = importlib.util.spec_from_file_location("regularization_checks", SCRIPT)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)
pilot = sweep.pilot
repeated = sweep.repeated


def synthetic_rows():
    """Known numbers for arithmetic/interpretation tests, not AG News metrics."""
    rows = []
    for i, seed in enumerate(sweep.SEEDS):
        for name in sweep.REPRESENTATIONS:
            for c in sweep.C_VALUES:
                effects = (
                    {0.1: -0.02, 1.0: 0.0, 10.0: 0.01}
                    if name == "tfidf"
                    else {0.1: 0.005, 1.0: 0.0, 10.0: -0.01}
                )
                score = 0.6 + i * 0.01 + (0.1 if name == "minilm" else 0) + effects[c]
                rows.append(
                    dict(
                        seed=seed,
                        representation=name,
                        C=c,
                        validation_accuracy=score,
                        validation_macro_f1=score,
                        lr_training_seconds=0.01 + i * 0.001,
                    )
                )
    return rows


class RegularizationChecks(unittest.TestCase):
    def test_only_C_changes_and_original_settings_are_not_mutated(self):
        original = dict(pilot.LR_SETTINGS)
        for c in sweep.C_VALUES:
            settings = sweep.classifier_settings(c)
            self.assertEqual(settings["C"], c)
            self.assertEqual(
                {key: value for key, value in settings.items() if key != "C"},
                {key: value for key, value in original.items() if key != "C"},
            )
            self.assertEqual(settings["l1_ratio"], 0.0)
        self.assertEqual(pilot.LR_SETTINGS, original)
        for invalid in [0, -1, 3, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                sweep.classifier_settings(invalid)

    def test_aggregation_and_paired_sample_standard_deviation(self):
        summaries, paired = sweep.aggregate_results(synthetic_rows())
        self.assertEqual(len(summaries), 6)
        self.assertEqual(len(paired), 18)
        baseline = next(
            row
            for row in summaries
            if row["representation"] == "tfidf" and row["C"] == 1
        )
        self.assertAlmostEqual(baseline["validation_accuracy_mean"], 0.61)
        self.assertAlmostEqual(baseline["validation_accuracy_std"], 0.01)
        for row in paired:
            if row["C"] == 1:
                self.assertEqual(row["validation_accuracy_difference"], 0.0)
        larger = next(
            row
            for row in summaries
            if row["representation"] == "minilm" and row["C"] == 10
        )
        self.assertAlmostEqual(larger["validation_accuracy_difference_mean"], -0.01)
        self.assertAlmostEqual(larger["validation_accuracy_difference_std"], 0.0)

    def test_incomplete_duplicate_and_invalid_scores_rejected(self):
        rows = synthetic_rows()
        for invalid in [rows[:-1], rows[:-1] + [rows[0]]]:
            with self.assertRaises(ValueError):
                sweep.aggregate_results(invalid)
        rows[0]["validation_macro_f1"] = float("nan")
        with self.assertRaises(ValueError):
            sweep.aggregate_results(rows)

    def test_baseline_verification_tolerance(self):
        rows = synthetic_rows()
        baselines = {
            (row["seed"], row["representation"]): dict(row)
            for row in rows
            if row["C"] == 1
        }
        self.assertTrue(
            all(row["matches"] for row in sweep.baseline_checks(rows, baselines))
        )
        baselines[42, "tfidf"]["validation_accuracy"] += 1 / 800
        self.assertFalse(
            all(row["matches"] for row in sweep.baseline_checks(rows, baselines))
        )

    def test_reports_do_not_assume_larger_C_is_better(self):
        rows = synthetic_rows()
        summaries, paired = sweep.aggregate_results(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep.export_reports(rows, summaries, paired, root)
            analysis = (root / "analysis.md").read_text(encoding="utf-8")
            self.assertIn("C=[0.1]", analysis)
            self.assertIn("3/3 declines", analysis)
            self.assertIn("selection-biased", analysis)
            self.assertIn("inner validation or stratified cross-validation", analysis)
            self.assertTrue((root / "results_table.md").is_file())
            for name in ["accuracy_vs_C.png", "macro_f1_vs_C.png"]:
                self.assertTrue((root / name).read_bytes().startswith(b"\x89PNG"))

    def test_existing_samples_caches_exports_and_failure_gate(self):
        labels = np.repeat(np.arange(4), 400).tolist()
        texts = [
            f"synthetic category{label} article number{index}"
            for index, label in enumerate(labels)
        ]
        dataset = Dataset.from_dict(
            dict(text=texts, label=labels),
            features=Features(
                dict(text=Value("string"), label=ClassLabel(names=["a", "b", "c", "d"]))
            ),
        )
        encoder = FakeEncoder()
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.ExitStack() as stack,
        ):
            root = Path(directory)
            for name, path in [
                ("RESULTS_ROOT", root / "pilot"),
                ("EMBEDDING_CACHE", root / "embeddings"),
                ("MODEL_CACHE", root / "models"),
                ("DATASET_CACHE", root / "datasets"),
            ]:
                stack.enter_context(patch.object(pilot, name, path))
            stack.enter_context(
                patch.object(repeated, "RESULTS_ROOT", root / "repeated")
            )
            stack.enter_context(
                patch.object(sweep, "RESULTS_ROOT", root / "regularization")
            )
            loader = stack.enter_context(
                patch.object(pilot, "load_dataset", return_value=dataset)
            )
            stack.enter_context(
                patch.object(pilot, "SentenceTransformer", return_value=encoder)
            )
            stack.enter_context(
                patch.object(pilot.torch.cuda, "is_available", return_value=True)
            )
            stack.enter_context(
                patch.object(
                    pilot.torch.cuda,
                    "get_device_name",
                    return_value="SYNTHETIC TEST DEVICE",
                )
            )
            stack.enter_context(patch.object(pilot.torch.cuda, "synchronize"))
            stack.enter_context(patch.object(pilot.torch.cuda, "manual_seed_all"))
            stack.enter_context(patch.object(pilot.torch, "manual_seed"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            # Build compatible historical fixtures with the unchanged real helpers.
            # All content is synthetic, and all output directories are temporary.
            pilot.main()
            pilot_reference = next((root / "pilot").iterdir())
            stack.enter_context(
                patch.object(repeated, "REFERENCE_PILOT_RUN", pilot_reference)
            )
            repeated.main()
            reference = next((root / "repeated").iterdir())
            stack.enter_context(patch.object(sweep, "REFERENCE_RUN", reference))
            previous_files = {
                path: path.read_bytes()
                for folder in [root / "pilot", root / "repeated", root / "embeddings"]
                for path in folder.rglob("*")
                if path.is_file()
            }
            calls_before = encoder.calls
            # The sweep must not request a model, call CUDA, or sample any rows.
            stack.enter_context(
                patch.object(
                    pilot,
                    "SentenceTransformer",
                    side_effect=AssertionError("Unexpected model load"),
                )
            )
            stack.enter_context(
                patch.object(
                    pilot.torch.cuda,
                    "synchronize",
                    side_effect=AssertionError("Unexpected CUDA work"),
                )
            )
            stack.enter_context(
                patch.object(
                    repeated,
                    "sample_training",
                    side_effect=AssertionError("Unexpected resampling"),
                )
            )
            stack.enter_context(
                patch.object(
                    pilot,
                    "sample_indices",
                    side_effect=AssertionError("Unexpected resampling"),
                )
            )
            vectorizer = stack.enter_context(
                patch.object(pilot, "tfidf_features", wraps=pilot.tfidf_features)
            )
            fitter = stack.enter_context(
                patch.object(sweep, "fit_one", wraps=sweep.fit_one)
            )
            sweep.main()
            self.assertEqual(encoder.calls, calls_before)
            self.assertEqual(vectorizer.call_count, 3)
            self.assertEqual(fitter.call_count, 18)
            self.assertTrue(
                all(call.args[-1] == 1.0 for call in fitter.call_args_list[:6])
            )
            output = next((root / "regularization").iterdir())
            metadata = json.loads(
                (output / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "complete")
            self.assertTrue(metadata["baseline_reproduction_passed"])
            self.assertEqual(
                len(
                    json.loads(
                        (output / "individual_results.json").read_text(encoding="utf-8")
                    )
                ),
                18,
            )
            self.assertEqual(
                len(json.loads((output / "summary.json").read_text(encoding="utf-8"))),
                6,
            )
            source_manifest = (reference / "sample_manifest.json").read_bytes()
            self.assertEqual(
                (output / "source_sample_manifest.json").read_bytes(), source_manifest
            )
            validation_ids = json.loads(source_manifest)["validation_indices"]
            for path in output.glob("predictions_*.csv"):
                with path.open(encoding="utf-8", newline="") as handle:
                    predictions = list(pilot.csv.DictReader(handle))
                self.assertEqual(
                    [int(row["dataset_row"]) for row in predictions], validation_ids
                )
            for call in loader.call_args_list:
                self.assertEqual(call.kwargs["split"], "train")
            for path, contents in previous_files.items():
                self.assertEqual(path.read_bytes(), contents)

            # A missing cache must not fall through to encode(None) or GPU work.
            context = json.loads(
                (reference / "run_metadata.json").read_text(encoding="utf-8")
            )["encoding_context"]
            with patch.object(pilot, "EMBEDDING_CACHE", root / "missing_cache"):
                manifest = json.loads(source_manifest)
                indices = manifest["validation_indices"]
                selected = [texts[index] for index in indices]
                events = json.loads(
                    (reference / "run_metadata.json").read_text(encoding="utf-8")
                )["feature_events"]
                with self.assertRaises(FileNotFoundError):
                    sweep.cached_embeddings(
                        selected, indices, context, events[0]["validation"]["key"]
                    )

            # Deliberately damage a temporary baseline score: verification must
            # stop after six C=1 fits and never publish a completed sweep/analysis.
            baseline_file = reference / "individual_results.json"
            changed = json.loads(baseline_file.read_text(encoding="utf-8"))
            row = next(row for row in changed if row["train_size"] == 400)
            row["validation_accuracy"] -= 0.01
            baseline_file.write_text(json.dumps(changed), encoding="utf-8")
            fitter.reset_mock()
            with self.assertRaisesRegex(ValueError, "baseline mismatch"):
                sweep.main()
            self.assertEqual(fitter.call_count, 6)
            failed = sorted((root / "regularization").iterdir())[-1]
            self.assertEqual(
                json.loads((failed / "run_metadata.json").read_text(encoding="utf-8"))[
                    "status"
                ],
                "failed",
            )
            self.assertTrue((failed / "baseline_reproduction.csv").exists())
            self.assertFalse((failed / "analysis.md").exists())


if __name__ == "__main__":
    unittest.main()
