"""Leakage and output preservation checks for future preprocessing runs."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocess import main, normalize_claim, save_jsonl, split_dataset, verify_splits


class PreprocessIsolation(unittest.TestCase):
    def test_case_and_whitespace_variants_stay_in_one_partition(self):
        rows = []
        for index in range(60):
            claim = "Example person %s was born." % index
            rows.extend([{"id": index * 2, "claim": claim},
                         {"id": index * 2 + 1, "claim": "  " + claim.upper().replace(" ", "   ") + "  "}])
        original = [r["claim"] for r in rows]
        splits = split_dataset(rows)
        groups = [{normalize_claim(r["claim"]) for r in split} for split in splits]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(sum(map(len, splits)), len(rows))
        self.assertEqual([r["claim"] for r in rows], original)

    def test_partition_check_rejects_normalized_overlap(self):
        def row(claim):
            return {"claim": claim, "label": "SUPPORTS"}
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Normalized claim overlap"):
                verify_splits([row("An Example.")], [row(" AN   EXAMPLE. ")], [row("Another claim.")])

    def test_existing_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "train.jsonl"
            output.write_text("preserve this dataset")
            with self.assertRaises(FileExistsError):
                save_jsonl([{"claim": "new data"}], output)
            self.assertEqual(output.read_text(), "preserve this dataset")

    def test_main_checks_all_outputs_before_loading_or_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "test.jsonl").write_text("preserve test")
            with patch("preprocess.load_fever") as load:
                with self.assertRaises(FileExistsError):
                    main(["--output-dir", str(folder)])
            load.assert_not_called()
            self.assertFalse((folder / "train.jsonl").exists())
            self.assertFalse((folder / "validation.jsonl").exists())
            self.assertEqual((folder / "test.jsonl").read_text(), "preserve test")


if __name__ == "__main__":
    unittest.main()
