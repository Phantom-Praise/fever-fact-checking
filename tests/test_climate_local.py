"""Local backend tests use a tiny corpus and fake Ollama, never inference."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import climate_local as local


class LocalTests(unittest.TestCase):
    def test_readonly_corpus_keeps_tail_of_article_not_five_sentences(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.sqlite"
            c = sqlite3.connect(path)
            c.execute("CREATE VIRTUAL TABLE title_fts USING fts5(page_id UNINDEXED,title)")
            c.execute("CREATE TABLE page_sentences(page_id TEXT,sentence_id INTEGER,text TEXT)")
            c.execute("INSERT INTO title_fts VALUES('Climate','Climate')")
            for i in range(9):
                c.execute("INSERT INTO page_sentences VALUES(?,?,?)", ("Climate", i, str(i) + "x" * 650))
            c.commit()
            c.close()
            corpus = local.LocalCorpus(path)
            self.assertEqual(corpus.search('Climate " OR'), ["Climate"])
            excerpts = corpus.excerpts(["Climate"])
            self.assertEqual(len(excerpts), 18)
            for sid in range(9):
                self.assertEqual("".join(e["text"] for e in excerpts if e["sentence_id"] == sid), str(sid) + "x" * 650)
            with self.assertRaises(sqlite3.OperationalError):
                corpus.connection.execute("DELETE FROM page_sentences")
            corpus.close()

    def test_missing_requested_model_never_uses_installed_smaller_model(self):
        config = local.read_config()
        with patch.object(local, "inventory", return_value=([{"name": "gemma3:1B"}], {"version": "test"})):
            with self.assertRaisesRegex(ValueError, "not installed"):
                local.model_metadata(config, "gemma")

    def test_full_raw_response_saved_and_token_limit_rejected(self):
        model = object.__new__(local.OllamaGenerator)
        model.meta = {"name": "test", "url": "http://127.0.0.1:11434", "context_tokens": 8192,
                      "timeout_seconds": 10}
        body = {"done": True, "done_reason": "length", "message": {"content": '{"parts":['},
                "prompt_eval_count": 100, "eval_count": 1024}
        calls = []
        def post(*args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)
        model.session = SimpleNamespace(post=post)
        trace = {"calls": []}
        with self.assertRaisesRegex(ValueError, "output-token-limited"):
            model.generate("claim", trace, "classification", 1024)
        self.assertEqual(trace["calls"][0]["raw_response"], body)
        self.assertEqual(trace["calls"][0]["response"], '{"parts":[')
        self.assertEqual(calls[0]["json"]["format"], "json")

    def test_oversized_prompt_is_not_sent(self):
        model = object.__new__(local.OllamaGenerator)
        model.meta = {"context_tokens": 8192}
        model.session = SimpleNamespace(post=lambda *a, **k: self.fail("Oversized request sent"))
        trace = {"calls": []}
        with self.assertRaisesRegex(ValueError, "context guard"):
            model.generate("x" * 8192, trace, "selection")
        self.assertEqual(trace["calls"][0]["prompt"], "x" * 8192)

    def test_batching_preserves_every_candidate(self):
        model = object.__new__(local.OllamaGenerator)
        model.meta = {"context_tokens": 8192}
        excerpts = [{"id": f"E{i}", "page_id": "Climate", "text": "x" * 600} for i in range(30)]
        batches = list(model.batches("Claim", excerpts))
        self.assertGreater(len(batches), 1)
        self.assertEqual([e for b in batches for e in b], excerpts)
        self.assertTrue(all(model.fits(model.selection_prompt("Claim", b), 512) for b in batches))

    def test_classification_does_not_reinsert_oversized_source_sentence(self):
        model = object.__new__(local.OllamaGenerator)
        prompts = []
        def generate(prompt, *args):
            prompts.append(prompt)
            return json.dumps({"label": "NOT ENOUGH INFO", "parts": [{"part": "Claim", "verdict": "NOT ENOUGH INFO",
                               "evidence_ids": [], "justification": "No decisive evidence."}]})
        model.generate = generate
        model.classify("Claim", [{"id": "E1", "page_id": "Climate", "text": "Relevant excerpt",
                                  "source_sentence": "omitted source " * 10000}], {"calls": []})
        self.assertIn("Relevant excerpt", prompts[0])
        self.assertNotIn("omitted source", prompts[0])

    def test_explicit_local_execution_does_not_require_cuda(self):
        class Model:
            def __init__(self, *args): self.tokenizer = None
            def query(self, claim, trace): return claim
            def select(self, *args): return []
            def classify(self, *args): return {"label": "NOT ENOUGH INFO"}
            def close(self): pass
        class Corpus:
            def __init__(self, *args): pass
            def search(self, query): return []
            def excerpts(self, *args): return []
            def close(self): pass
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            database = folder / "db"
            database.write_text("mock")
            with patch.object(local.core, "check_cuda", side_effect=AssertionError("CUDA must not be required")), \
                 patch.object(local.core, "environment", side_effect=AssertionError("Colab packages must not be required")):
                status = local.climate.run([{"id": "climate-0", "claim": "Claim"}], folder, folder / "out",
                    database, "source", "llama", {"digest": "fixed"}, generator_class=Model,
                    corpus_class=Corpus, execution_environment={"backend": "test-local"})
                self.assertEqual(status, "complete")
            saved = json.loads((folder / "out/llama/predictions/climate-0.json").read_text())
            self.assertEqual(saved["label"], "NOT ENOUGH INFO")


if __name__ == "__main__":
    unittest.main()
