"""Offline checks for independent selection, trace integrity and resumable runs."""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
import independent_systems as systems
import fever_colab_runtime as core

class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()
    def decode(self, tokens):
        return " ".join(tokens)

class SelectionTests(unittest.TestCase):
    def test_full_index_and_sentence_splitting_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "wiki.zip"
            text = " ".join("word" for _ in range(321))
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("wiki-pages/wiki-001.jsonl",
                    json.dumps(dict(id="Example", text=text, lines="0\t" + text)) + "\n")
            database = systems.build_corpus(source, root / "index", core.sha_file(source))
            systems.build_corpus(source, root / "index", core.sha_file(source))
            corpus = systems.Corpus(database)
            self.assertEqual(corpus.search('Example " OR'), ["Example"])
            excerpts = corpus.excerpts(["Example"], Tokenizer())
            self.assertEqual(len(excerpts), 3)
            self.assertEqual(sum(len(e["text"].split()) for e in excerpts), 321)
            corpus.close()

    def test_generator_selects_its_own_ids(self):
        model = object.__new__(systems.Generator)
        model.batches = lambda claim, excerpts: iter([excerpts])
        model.json_response = lambda *args: {"evidence_ids": ["E2"], "reason": "Relevant"}
        excerpts = [dict(id="E1", page_id="A", text="a"), dict(id="E2", page_id="B", text="b")]
        selected = model.select("claim", excerpts, {"calls": []}, SimpleNamespace(event=lambda *a, **kw: None))
        self.assertEqual(selected, [excerpts[1]])

    def test_unknown_selection_id_rejected(self):
        model = object.__new__(systems.Generator)
        model.batches = lambda claim, excerpts: iter([excerpts])
        model.json_response = lambda *args: {"evidence_ids": ["INVENTED"], "reason": "Relevant"}
        with self.assertRaises(ValueError):
            model.select("claim", [dict(id="E1", page_id="A", text="a")],
                         {"calls": []}, SimpleNamespace(event=lambda *a, **kw: None))

    def test_labels_cannot_enter_run_features(self):
        rows = [dict(id=i, claim=str(i)) for i in range(19998)]
        rows[0]["label"] = "SUPPORTS"
        with self.assertRaises(ValueError):
            systems.run(rows, ".", ".", ".", "hash", "qwen", {}, ".")

    def test_failures_saved_and_three_invalid_pause_then_resume(self):
        rows = [dict(id=i, claim=f"claim {i}") for i in range(19998)]
        class FailingGenerator:
            def __init__(self, *args):
                self.tokenizer = Tokenizer()
            def query(self, claim, trace):
                trace["calls"].append(dict(prompt=claim, response="not JSON"))
                raise ValueError("Invalid JSON")
            def close(self):
                pass
        class FakeCorpus:
            def __init__(self, *args): pass
            def close(self): pass
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "src/fever_colab_runtime.py").write_text("runtime")
            (root / "parts_prompt.txt").write_text("prompt")
            with patch.object(systems, "Generator", FailingGenerator), patch.object(systems, "Corpus", FakeCorpus), \
                 patch.object(core, "environment", return_value={}), \
                 patch("torch.manual_seed"), patch("torch.cuda.manual_seed_all"):
                status = systems.run(rows, root, root / "out", root / "db",
                                     "source", "qwen", {"repo": "test"}, root, minutes=1)
                self.assertEqual(status, "needs_debugging")
                status = systems.run(rows, root, root / "out", root / "db",
                                     "source", "qwen", {"repo": "test"}, root, minutes=1)
                self.assertEqual(status, "needs_debugging")
            saved = sorted((root / "out/qwen/predictions").glob("*.json"))
            self.assertEqual(len(saved), 6)
            trace = json.loads(saved[0].read_text())
            self.assertEqual(trace["label"], "INVALID")
            self.assertEqual(trace["calls"][0]["response"], "not JSON")
            labels = {str(i): "SUPPORTS" for i in range(19998)}
            result = systems.report(rows, labels, root / "out", ["qwen"])[0]
            self.assertEqual(result["n"], 6)
            self.assertEqual(result["invalid"], 6)
            self.assertEqual(result["accuracy_percent"], 0)
            self.assertEqual(result["status"], "partial")

if __name__ == "__main__":
    unittest.main()
