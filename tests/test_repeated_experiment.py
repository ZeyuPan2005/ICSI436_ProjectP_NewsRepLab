"""Offline tests only: synthetic data and temporary outputs, no research scores.

These tests verify fixed validation, exact seed-42 reuse, aggregation arithmetic,
honest analysis of counterexamples, and use of the prior pilot's embedding cache.
They never load AG News or run GPU inference.
"""

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

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "repeated_experiment.py"
SPEC = importlib.util.spec_from_file_location("repeated_checks", SCRIPT)
repeated = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repeated)
pilot = repeated.pilot


def synthetic_scores():
    """Controlled numbers for checking arithmetic, never written to project results."""
    rows = []
    for index, seed in enumerate(repeated.SEEDS):
        for name in repeated.REPRESENTATIONS:
            for size in pilot.TRAIN_SIZES:
                value = (
                    0.5
                    + 0.1 * index
                    + (0.05 if name == "minilm" else 0)
                    + (0.02 if size == 400 else 0)
                )
                rows.append(
                    dict(
                        seed=seed,
                        representation=name,
                        train_size=size,
                        validation_accuracy=value,
                        validation_macro_f1=value,
                    )
                )
    return rows


class RepeatedChecks(unittest.TestCase):
    def test_fixed_validation_and_nested_balanced_seeds(self):
        labels = np.repeat(np.arange(4), 400)
        original, validation = pilot.sample_indices(labels)
        saved_validation = validation.copy()
        plans = {42: original}
        for seed in repeated.SEEDS[1:]:
            plans[seed] = repeated.sample_training(labels, validation, seed)
            self.assertEqual(
                plans[seed], repeated.sample_training(labels, validation, seed)
            )
        for plan in plans.values():
            repeated.validate_indices(labels, plan, validation)
            self.assertTrue(set(plan[100]) < set(plan[400]))
            self.assertFalse(set(plan[400]) & set(validation))
            np.testing.assert_array_equal(np.bincount(labels[plan[100]]), [25] * 4)
            np.testing.assert_array_equal(np.bincount(labels[plan[400]]), [100] * 4)
        self.assertEqual(validation, saved_validation)
        self.assertNotEqual(plans[123], plans[2026])

    def test_overlap_is_rejected(self):
        labels = np.repeat(np.arange(4), 400)
        training, validation = pilot.sample_indices(labels)
        replacement = next(
            row for row in validation if labels[row] == labels[training[400][0]]
        )
        training[400][0] = replacement
        with self.assertRaisesRegex(ValueError, "overlap"):
            repeated.validate_indices(labels, training, validation)

    def test_mean_sample_std_and_paired_differences(self):
        rows = synthetic_scores()
        summary = repeated.summarize_results(rows)
        small_tfidf = next(
            row
            for row in summary
            if row["representation"] == "tfidf" and row["train_size"] == 100
        )
        self.assertAlmostEqual(small_tfidf["validation_accuracy_mean"], 0.6)
        self.assertAlmostEqual(small_tfidf["validation_accuracy_std"], 0.1)
        self.assertEqual(small_tfidf["std_ddof"], 1)
        differences = repeated.paired_differences(rows)
        self.assertEqual(len(differences), 24)
        for row in differences:
            self.assertAlmostEqual(
                row["difference"],
                0.05 if row["contrast"] == "minilm_minus_tfidf" else 0.02,
            )

    def test_incomplete_duplicate_or_invalid_results_rejected(self):
        rows = synthetic_scores()
        for invalid in [rows[:-1], rows[:-1] + [rows[0]]]:
            with self.assertRaises(ValueError):
                repeated.summarize_results(invalid)
        rows[0]["validation_accuracy"] = float("nan")
        with self.assertRaises(ValueError):
            repeated.summarize_results(rows)

    def test_analysis_reports_losses_and_does_not_assume_plateau(self):
        rows = synthetic_scores()
        target = next(
            row
            for row in rows
            if row["seed"] == 123
            and row["representation"] == "minilm"
            and row["train_size"] == 400
        )
        target.update(validation_accuracy=0.55, validation_macro_f1=0.55)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.md"
            repeated.write_analysis(rows, repeated.summarize_results(rows), path)
            content = path.read_text(encoding="utf-8")
            self.assertIn("not consistently across these three seeds", content)
            self.assertIn("decreases in 1/3", content)
            self.assertIn("larger gains in 2/3", content)
            self.assertIn("SD is sample SD (ddof=1)", content)
            self.assertIn("not a significance or equivalence test", content)

    def test_full_workflow_reuses_pilot_cache_and_preserves_results(self):
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
            # First populate the cache using the unmodified pilot implementation.
            pilot.main()
            reference = next((root / "pilot").iterdir())
            original_files = {
                path: path.read_bytes()
                for path in reference.rglob("*")
                if path.is_file()
            }
            stack.enter_context(
                patch.object(repeated, "REFERENCE_PILOT_RUN", reference)
            )
            self.assertEqual(encoder.calls, 4)
            repeated.main()
            # Only four NEW training blocks (123/2026 x 100/400) plus warmup.
            self.assertEqual(encoder.calls, 9)
            output = next((root / "repeated").iterdir())
            rows = json.loads(
                (output / "individual_results.json").read_text(encoding="utf-8")
            )
            summaries = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            metadata = json.loads(
                (output / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(summaries), 4)
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["completed_seeds"], [42, 123, 2026])
            self.assertTrue(
                all(
                    row["validation_cache_hit"]
                    for row in rows
                    if row["representation"] == "minilm"
                )
            )
            self.assertTrue(
                all(
                    row["train_cache_hit"]
                    for row in rows
                    if row["representation"] == "minilm" and row["seed"] == 42
                )
            )
            manifest = json.loads(
                (output / "sample_manifest.json").read_text(encoding="utf-8")
            )
            reference_manifest = json.loads(
                (reference / "sample_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["validation_indices"], reference_manifest["validation_indices"]
            )
            self.assertEqual(
                manifest["training_indices"]["42"],
                reference_manifest["training_indices"],
            )
            for path, original in original_files.items():
                self.assertEqual(path.read_bytes(), original)
            for call in loader.call_args_list:
                self.assertEqual(call.kwargs["split"], "train")
            for seed in repeated.SEEDS:
                seed_dir = output / f"seed_{seed}"
                self.assertEqual(
                    len(
                        json.loads(
                            (seed_dir / "metrics.json").read_text(encoding="utf-8")
                        )
                    ),
                    4,
                )
                for name in repeated.REPRESENTATIONS:
                    for size in pilot.TRAIN_SIZES:
                        with (seed_dir / f"{name}_{size}_predictions.csv").open(
                            encoding="utf-8", newline=""
                        ) as handle:
                            prediction_rows = list(pilot.csv.DictReader(handle))
                        self.assertEqual(
                            [int(row["dataset_row"]) for row in prediction_rows],
                            manifest["validation_indices"],
                        )
            for filename in [
                "mean_std_comparison.png",
                "individual_seed_comparison.png",
            ]:
                self.assertTrue((output / filename).read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((output / "analysis.md").exists())
            # A second run must preserve the first and encode only the warmup.
            repeated.main()
            self.assertEqual(encoder.calls, 10)
            self.assertEqual(len(list((root / "repeated").iterdir())), 2)
            # A changed dataset fingerprint must not silently replace validation.
            with self.assertRaisesRegex(ValueError, "dataset_fingerprint"):
                repeated.load_reference(
                    np.asarray(labels), "wrong-fingerprint", ["a", "b", "c", "d"]
                )


if __name__ == "__main__":
    unittest.main()
