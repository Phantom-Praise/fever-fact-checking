"""Migration checks without downloading weights, training, or GPU inference."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "colab"))
import climate_fever as climate
import fever_colab_runtime as core


def response(verdicts, label, ids=None):
    return json.dumps({"label": label, "parts": [
        {"part": f"assertion {i}", "verdict": v, "evidence_ids":
         (ids if ids is not None else [] if v == "NOT ENOUGH INFO" else ["E1"]),
         "justification": "Evidence comparison."} for i, v in enumerate(verdicts)]})


class ClimateTests(unittest.TestCase):
    def test_pinned_release_and_feature_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            manifest = climate.prepare(ROOT / "data/climate_fever/source/climate-fever.jsonl", out)
            rows, verified = climate.verify_data(out)
            self.assertEqual(verified, manifest)
            self.assertEqual(len(rows), 1535)
            self.assertEqual(manifest["label_counts"]["DISPUTED"], 154)
            self.assertTrue(all(set(row) == {"id", "claim"} for row in rows))
            labels = json.loads((out / "labels.json").read_text())
            self.assertEqual(sum(v != "DISPUTED" for v in labels.values()), 1381)
            # Re-preparation is idempotent, never replaces differing source data.
            climate.prepare(ROOT / "data/climate_fever/source/climate-fever.jsonl", out)
            (out / "inputs.json").write_text("[]")
            with self.assertRaises(ValueError):
                climate.verify_data(out)

    def test_disputed_aggregation_and_reported_disagreement(self):
        value = climate.parse_response(response(["SUPPORTS", "REFUTES"], "REFUTES"), {"E1"})
        self.assertEqual(value["label"], "DISPUTED")
        self.assertFalse(value["consistent"])
        self.assertEqual(climate.parse_response(response(["DISPUTED"], "DISPUTED"), {"E1"})["label"], "DISPUTED")

    def test_missing_information_does_not_refute(self):
        value = climate.parse_response(response(["SUPPORTS", "NOT ENOUGH INFO"], "NOT ENOUGH INFO"), {"E1"})
        self.assertEqual(value["label"], "NOT ENOUGH INFO")

    def test_invented_citation_and_incomplete_json_rejected(self):
        for raw in (response(["REFUTES"], "REFUTES", ["invented"]),
                    response(["SUPPORTS"], "SUPPORTS", []), '{"parts":['):
            with self.assertRaises(ValueError):
                climate.parse_response(raw, {"E1"})

    def test_four_class_report_counts_invalid_as_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [{"id": "climate-1", "claim": "One"}, {"id": "climate-2", "claim": "Two"}]
            protocol = {"test": "pinned"}
            identity = core.freeze(output / "roberta", protocol)
            for row, label in zip(rows, ["INVALID", "REFUTES"]):
                core.write_once(output / "roberta/predictions" / (row["id"] + ".json"),
                                {**row, "identity": identity, "label": label, "wall_seconds": 1})
            report = climate.report(rows, {"climate-1": "DISPUTED", "climate-2": "REFUTES"},
                                    output, ["roberta", "qwen"])
            self.assertEqual(report[0]["accuracy_percent"], 50)
            self.assertEqual(report[0]["invalid"], 1)
            self.assertEqual(report[0]["confusion_matrix"][3][4], 1)
            self.assertEqual(report[0]["non_disputed_secondary_accuracy_percent"], 100)
            self.assertFalse(report[0]["native_four_class"])
            self.assertEqual(report[1]["status"], "not_started")

    def test_ground_truth_features_rejected_before_loading_models(self):
        with self.assertRaises(ValueError):
            climate.run([{"id": "climate-1", "claim": "One", "label": "SUPPORTS"}],
                        ".", ".", ".", {}, "qwen")

    def test_stop_and_resume_preserve_raw_failures(self):
        class FailingModel:
            def __init__(self, *args): pass
            def query(self, claim, trace):
                trace["calls"].append({"prompt": claim, "response": '{"query":'})
                raise ValueError("Incomplete JSON")
            def close(self): pass

        rows = [{"id": f"climate-{i}", "claim": str(i)} for i in range(5)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "corpus.sqlite"
            database.write_text("mock corpus")
            with patch.object(core, "environment", return_value={}), patch.object(core, "check_cuda"), \
                 patch.object(climate.systems, "Corpus", return_value=SimpleNamespace(close=lambda: None)), \
                 patch("torch.cuda.manual_seed_all"):
                first = climate.run(rows, root, root / "out", database, "pinned", "qwen",
                                    {"revision": "fixed"}, generator_class=FailingModel)
                self.assertEqual(first, "needs_debugging")
                paths = sorted((root / "out/qwen/predictions").glob("*.json"))
                self.assertEqual(len(paths), 3)
                original = paths[0].read_bytes()
                second = climate.run(rows, root, root / "out", database, "pinned", "qwen",
                                     {"revision": "fixed"}, generator_class=FailingModel)
                self.assertEqual(second, "complete")
                self.assertEqual(paths[0].read_bytes(), original)
                self.assertEqual(json.loads(original)["calls"][0]["response"], '{"query":')
                # A complete run means processed, not valid or accurate.
                summary = climate.report(rows, {row["id"]: "SUPPORTS" for row in rows}, root / "out", ["qwen"])[0]
                self.assertEqual(summary["invalid"], 5)
                self.assertEqual(summary["accuracy_percent"], 0)
                with self.assertRaises(ValueError):
                    climate.run(rows, root, root / "out", database, "changed", "qwen", {"revision": "fixed"})


if __name__ == "__main__":
    unittest.main()
