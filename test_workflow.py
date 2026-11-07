"""Verhaltens- und Integrationstests ohne zusätzliche Testbibliothek."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from workflow import LABELS, ROOT, load_data, predict, train


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.workspace.name)
        cls.metrics = train(output_dir=cls.directory / "artifacts")
        cls.model_path = cls.directory / "artifacts/model.joblib"

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def write_csv(self, name, rows):
        path = self.directory / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)
        return path

    def test_training_produces_metrics_and_saved_model(self):
        self.assertTrue(self.model_path.is_file())
        report = json.loads((self.model_path.parent / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(report["train_samples"], 72)
        self.assertEqual(report["test_samples"], 18)
        self.assertGreater(report["accuracy"], report["baseline_accuracy"])
        self.assertTrue(0 <= report["macro_f1"] <= 1)
        self.assertEqual(sum(map(sum, report["confusion_matrix"])), report["test_samples"])

    def test_saved_model_classifies_new_texts(self):
        examples = {
            "Meine App stürzt ab und der Player startet nicht": "Technik",
            "Die Rechnung ist falsch und die Zahlung wurde doppelt abgebucht": "Abrechnung",
            "Ich möchte meinen Vertrag kündigen und den Tarif wechseln": "Vertrag",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(predict(text, self.model_path, None)["category"], expected)

    def test_empty_input_is_rejected(self):
        for text in ["", "   ", None, 42]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                predict(text, self.model_path, None)

    def test_missing_model_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "train"):
            predict("Die App startet nicht", self.directory / "missing.joblib", None)

    def test_log_appends_metadata_without_raw_text(self):
        path = self.directory / "logs/test.jsonl"
        text = "Meine Rechnung ist falsch"
        first = predict(text, self.model_path, path)
        second = predict("Der Player startet nicht", self.model_path, path)
        content = path.read_text(encoding="utf-8")
        entries = [json.loads(line) for line in content.splitlines()]
        self.assertEqual(entries, [first, second])
        self.assertNotIn(text, content)
        self.assertEqual(first["text_length"], len(text))
        self.assertGreaterEqual(first["runtime_ms"], 0)
        self.assertTrue(first["timestamp_utc"].endswith("+00:00"))
        self.assertEqual(first["model_version"], self.metrics["model_version"])

    def test_wrong_csv_columns_are_rejected(self):
        path = self.write_csv("wrong.csv", [["message", "category"], ["Hallo", "Technik"]])
        with self.assertRaisesRegex(ValueError, "Spalten"):
            load_data(path)

    def test_duplicate_texts_are_rejected_even_with_different_capitalization(self):
        path = self.write_csv("duplicate.csv", [
            ["text", "label"], ["App startet nicht", "Technik"],
            ["  APP   startet nicht  ", "Technik"],
        ])
        with self.assertRaisesRegex(ValueError, "Doppelter Text"):
            load_data(path)

    def test_unknown_categories_are_rejected(self):
        path = self.write_csv("unknown.csv", [["text", "label"], ["Hallo", "Sonstiges"]])
        with self.assertRaisesRegex(ValueError, "Kategorie"):
            load_data(path)

    def test_too_few_training_examples_are_rejected(self):
        path = self.write_csv("small.csv", [["text", "label"], ["App startet nicht", "Technik"]])
        with self.assertRaisesRegex(ValueError, "mindestens fünf"):
            load_data(path)

    def test_cli_prediction_returns_valid_json(self):
        run = subprocess.run([
            sys.executable, str(ROOT / "workflow.py"), "predict",
            "--text", "Meine Rechnung ist falsch", "--model", str(self.model_path),
            "--log-file", str(self.directory / "cli.jsonl"),
        ], capture_output=True, text=True, encoding="utf-8", check=True)
        self.assertEqual(json.loads(run.stdout)["category"], "Abrechnung")

    def test_cli_invalid_input_exits_with_error(self):
        run = subprocess.run([
            sys.executable, str(ROOT / "workflow.py"), "predict", "--text", "   ",
        ], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(run.returncode, 2)
        self.assertIn("nicht leeren Text", run.stderr)


if __name__ == "__main__":
    unittest.main()
