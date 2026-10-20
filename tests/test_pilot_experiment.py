"""Offline implementation checks, NOT AG News experiments or research results.

Run with the existing interpreter: python -B -m unittest discover -s tests -v.
Synthetic texts and a fake encoder exercise the experiment plumbing. Every
generated file lives in a temporary directory and is removed after the check.
No Hugging Face downloads or CUDA computations are performed by these tests.
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

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "pilot_experiment.py"
SPEC = importlib.util.spec_from_file_location("pilot", MODULE_PATH)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class FakeEncoder:
    """Predictable numeric features for software tests, not a pretrained model."""

    max_seq_length = 256

    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def parameters(self):
        return []

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        self.calls += 1
        result = np.zeros((len(texts), 384), dtype=np.float32)
        for index, text in enumerate(texts):
            class_id = next(
                (label for label in range(4) if f"category{label}" in text), 0
            )
            result[index, class_id] = 1.0
        return result


class PilotChecks(unittest.TestCase):
    def test_balanced_disjoint_nested_repeatable_sampling(self):
        labels = np.repeat(np.arange(4), 310)
        training, validation = pilot.sample_indices(labels)
        self.assertEqual((training, validation), pilot.sample_indices(labels))
        self.assertEqual(len(set(validation)), 800)
        np.testing.assert_array_equal(np.bincount(labels[validation]), [200] * 4)
        for size in (100, 400):
            self.assertEqual(len(set(training[size])), size)
            self.assertFalse(set(training[size]) & set(validation))
            np.testing.assert_array_equal(
                np.bincount(labels[training[size]]), [size // 4] * 4
            )
        self.assertTrue(set(training[100]) < set(training[400]))

    def test_sampling_rejects_insufficient_rows(self):
        with self.assertRaises(ValueError):
            pilot.sample_indices(np.repeat(np.arange(4), 299))

    def test_validation_words_do_not_enter_tfidf_vocabulary(self):
        train_x, val_x, vectorizer, _, _ = pilot.tfidf_features(
            ["common trainingword", "common secondword"], ["validationonlytoken common"]
        )
        self.assertNotIn("validationonlytoken", vectorizer.vocabulary_)
        self.assertEqual(train_x.shape[1], val_x.shape[1])

    def test_embedding_cache_reuse_invalidation_and_corruption(self):
        encoder = FakeEncoder()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(pilot, "EMBEDDING_CACHE", Path(directory)),
                patch.object(pilot.torch.cuda, "synchronize"),
            ):
                first, event = pilot.embedding_block(
                    encoder, ["category0 text"], [10], {"revision": "a"}
                )
                again, hit = pilot.embedding_block(
                    encoder, ["category0 text"], [10], {"revision": "a"}
                )
                self.assertEqual(encoder.calls, 1)
                self.assertTrue(hit["cache_hit"])
                self.assertEqual(hit["compute_this_call_seconds"], 0)
                self.assertEqual(
                    hit["measured_extraction_seconds"],
                    event["measured_extraction_seconds"],
                )
                np.testing.assert_array_equal(first, again)
                for texts, ids, context in [
                    (["category1 changed"], [10], {"revision": "a"}),
                    (["category0 text"], [11], {"revision": "a"}),
                    (["category0 text"], [10], {"revision": "b"}),
                ]:
                    _, new = pilot.embedding_block(encoder, texts, ids, context)
                    self.assertFalse(new["cache_hit"])
                self.assertEqual(encoder.calls, 4)
                np.save(
                    Path(directory) / f"{event['key']}.npy",
                    np.zeros((1, 384), dtype=np.float32),
                )
                with self.assertRaisesRegex(ValueError, "checksum"):
                    pilot.embedding_block(
                        encoder, ["category0 text"], [10], {"revision": "a"}
                    )
                _, refreshed = pilot.embedding_block(
                    encoder, ["category0 text"], [10], {"revision": "a"}, force=True
                )
                self.assertFalse(refreshed["cache_hit"])

    def test_invalid_embedding_values_rejected(self):
        for bad in [
            np.zeros((2, 383)),
            np.full((2, 384), np.nan),
            np.ones((2, 384), dtype=int),
        ]:
            with self.assertRaises(ValueError):
                pilot.validate_embeddings(bad, 2)

    def test_offline_four_experiment_workflow_and_exports(self):
        # Real sklearn classifiers run on synthetic text/features; no scores are
        # presented as evidence about AG News. This tests both sparse/dense paths.
        labels = np.repeat(np.arange(4), 310).tolist()
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
                ("RESULTS_ROOT", root / "outputs"),
                ("EMBEDDING_CACHE", root / "embeddings"),
                ("MODEL_CACHE", root / "models"),
                ("DATASET_CACHE", root / "datasets"),
            ]:
                stack.enter_context(patch.object(pilot, name, path))
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
            pilot.main()
            run_dir = next((root / "outputs").iterdir())
            loader.assert_called_once_with(
                pilot.DATASET_NAME,
                split="train",
                revision=pilot.DATASET_REVISION,
                cache_dir=str(root / "datasets"),
            )
            metadata = json.loads(
                (run_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "complete")
            rows = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(
                {(row["representation"], row["train_size"]) for row in rows},
                {("tfidf", 100), ("tfidf", 400), ("minilm", 100), ("minilm", 400)},
            )
            self.assertEqual(encoder.calls, 4)  # one warmup plus 800, 100, and 400
            self.assertTrue(rows[-1]["validation_cache_hit"])
            self.assertEqual(
                rows[-1]["feature_timing_origin"], "includes_reused_measurement"
            )
            self.assertTrue(
                (run_dir / "performance_comparison.png")
                .read_bytes()
                .startswith(b"\x89PNG")
            )
            for row in rows:
                prefix = f"{row['representation']}_{row['train_size']}"
                with (run_dir / f"{prefix}_predictions.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    predictions = list(pilot.csv.DictReader(handle))
                self.assertEqual(len(predictions), 800)
                with (run_dir / f"{prefix}_errors.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    self.assertEqual(
                        len(list(pilot.csv.DictReader(handle))), row["error_count"]
                    )
                diagnostics = json.loads(
                    (run_dir / f"{prefix}_diagnostics.json").read_text(encoding="utf-8")
                )
                self.assertEqual(int(np.sum(diagnostics["confusion_matrix"])), 800)
                self.assertEqual(diagnostics["learned_coefficient_shape"][0], 4)
            pilot.main()
            self.assertEqual(encoder.calls, 5)  # second warmup, then all cache hits
            latest = sorted((root / "outputs").iterdir())[-1]
            rerun_rows = json.loads(
                (latest / "metrics.json").read_text(encoding="utf-8")
            )
            for row in rerun_rows:
                if row["representation"] == "minilm":
                    self.assertEqual(row["feature_compute_this_run_seconds"], 0.0)
                    self.assertTrue(row["train_cache_hit"])


if __name__ == "__main__":
    unittest.main()
