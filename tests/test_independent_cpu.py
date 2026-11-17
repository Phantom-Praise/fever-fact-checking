"""CPU backend checks without downloading language-model weights."""
import hashlib
from importlib.machinery import ModuleSpec
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
import independent_cpu as cpu

class BudgetTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]
    def encode(self, text, **kwargs):
        return text.split()

class CPUBackendTests(unittest.TestCase):
    def test_generation_uses_cpu_backend_and_retains_raw_output(self):
        model = object.__new__(cpu.CPUGenerator)
        model.tokenizer = BudgetTokenizer()
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": '{"query":"Example"}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 5}}
        model.model = SimpleNamespace(create_chat_completion=create)
        trace = {"calls": []}
        raw = model.generate("Example claim", trace, "search_query", 192)
        self.assertEqual(json.loads(raw), {"query": "Example"})
        self.assertEqual(calls[0]["temperature"], 0)
        self.assertEqual(calls[0]["max_tokens"], 192)
        self.assertEqual(trace["calls"][0]["backend"], "llama.cpp CPU")
        self.assertEqual(trace["calls"][0]["usage"]["completion_tokens"], 5)

    def test_truncated_response_preserved_and_rejected(self):
        model = object.__new__(cpu.CPUGenerator)
        model.tokenizer = BudgetTokenizer()
        model.model = SimpleNamespace(create_chat_completion=lambda **kwargs: {
            "choices": [{"message": {"content": '{"parts":['}, "finish_reason": "length"}]})
        trace = {"calls": []}
        with self.assertRaises(ValueError):
            model.generate("claim", trace, "classification")
        self.assertEqual(trace["calls"][0]["response"], '{"parts":[')

    def test_loader_forces_zero_gpu_layers(self):
        import transformers
        kwargs = {}
        def load(**values):
            kwargs.update(values)
            return SimpleNamespace(close=lambda: None)
        llama_module = ModuleType("llama_cpp")
        llama_module.__spec__ = ModuleSpec("llama_cpp", loader=None)
        llama_module.Llama = load
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "parts_prompt.txt").write_text("prompt")
            with patch.dict(sys.modules, llama_cpp=llama_module), \
                 patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=BudgetTokenizer()), \
                 patch.object(cpu, "local_weights", return_value=root / "weights.gguf"), \
                 patch.object(cpu.core, "load_hf_token", return_value=None):
                cpu.CPUGenerator({"repo": "test", "revision": "fixed"}, root)
        self.assertEqual(kwargs["n_gpu_layers"], 0)
        self.assertEqual(kwargs["n_ctx"], 4096)

    def test_weight_cache_restores_and_checksums_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meta = {"gguf_repo": "test/model", "gguf_revision": "fixed", "files": [
                {"name": "part1.gguf", "bytes": 4, "sha256": hashlib.sha256(b"test").hexdigest()}]}
            cached = root / "drive/cpu_model_cache/test--model/fixed"
            cached.mkdir(parents=True)
            (cached / "part1.gguf").write_bytes(b"test")
            with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("Unneeded download")):
                path = cpu.local_weights(meta, root / "drive", root / "local")
            self.assertEqual(path.read_bytes(), b"test")
            path.write_bytes(b"bad!")
            with self.assertRaises(ValueError):
                cpu.local_weights(meta, root / "drive", root / "local")

    def test_00_roberta_ranking_and_prediction_run_on_real_cpu(self):
        import torch
        import transformers
        class Tokenizer:
            def __call__(self, claims, text, **kwargs):
                count = len(claims) if isinstance(claims, list) else 1
                if kwargs.get("return_tensors") != "pt":
                    return {"input_ids": [0, 4, 2]}
                return transformers.BatchEncoding({
                    "input_ids": torch.tensor([[0, 4, 2]] * count),
                    "attention_mask": torch.ones((count, 3), dtype=torch.long)})
            def decode(self, ids):
                return "retained input"
        config = transformers.RobertaConfig(vocab_size=16, hidden_size=8,
            intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
            num_labels=3, max_position_embeddings=514)
        with tempfile.TemporaryDirectory() as temp:
            transformers.RobertaForSequenceClassification(config).save_pretrained(temp)
            with patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=Tokenizer()):
                selector = cpu.CPURobertaSelector(temp)
                self.assertEqual(selector.verifier.model.device.type, "cpu")
                excerpts = [{"id": "E1", "page_id": "Page", "sentence_id": 0, "text": "Sentence"}]
                chosen = selector.select("Claim", excerpts, {}, SimpleNamespace(event=lambda *a, **kw: None))
                result = selector.classify("Claim", chosen, {})
                self.assertIn(result["label"], cpu.core.LABELS)
                self.assertEqual(result["retained_input"], "retained input")
                selector.close()

if __name__ == "__main__":
    unittest.main()
