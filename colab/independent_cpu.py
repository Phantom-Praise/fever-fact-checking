"""CPU adapters for independent evidence selection and FEVER classification.

GGUF Q4 inference is a separate experiment from the earlier GPU/NF4 run.
Model files are pinned, checksummed, cached in Drive, and loaded locally.
"""
import gc
import importlib.metadata
import os
import shutil
import time
from pathlib import Path

import fever_colab_runtime as core
import independent_systems as systems

GGUF_MODELS = {
    "qwen": ("Qwen/Qwen2.5-7B-Instruct-GGUF", "q4_k_m"),
    "llama": ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "q4_k_m"),
    "mistral": ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "q4_k_m"),
    "gemma": ("google/gemma-3-4b-it-qat-q4_0-gguf", "q4_0"),
}

def environment():
    import torch
    names = ["torch", "transformers", "huggingface-hub", "llama-cpp-python",
             "numpy", "scikit-learn"]
    return {"device": "cpu", "packages": {name: importlib.metadata.version(name) for name in names},
            "threads": max(1, os.cpu_count() or 1), "torch_threads": torch.get_num_threads(),
            "adapter_sha256": core.sha_file(__file__)}

def metadata(family):
    """Read metadata only and print weight size before downloading any weights."""
    from huggingface_hub import HfApi
    repo, quantization = GGUF_MODELS[family]
    api = HfApi(token=core.load_hf_token())
    info = api.model_info(repo, files_metadata=True)
    tokenizer_info = api.model_info(systems.MODELS[family])
    files = []
    for item in info.siblings:
        name = item.rfilename
        if not (name.endswith(".gguf") and quantization in name.lower() and "mmproj" not in name.lower()):
            continue
        lfs = item.lfs
        digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if not digest or not item.size:
            raise ValueError("Missing immutable GGUF file metadata")
        files.append({"name": name, "bytes": item.size, "sha256": digest})
    files.sort(key=lambda item: item["name"])
    if not files:
        raise ValueError("No matching CPU model files")
    size = sum(item["bytes"] for item in files)
    print(f"{family}: {size / 2**30:.2f} GiB GGUF download, {quantization}, CPU only.", flush=True)
    return {"repo": systems.MODELS[family], "revision": tokenizer_info.sha,
            "gguf_repo": repo, "gguf_revision": info.sha, "files": files,
            "quantization": quantization, "backend": "llama.cpp CPU",
            "n_ctx": 4096, "n_gpu_layers": 0}

def local_weights(meta, drive_project, local_root=Path("/content/fever_cpu_models")):
    from huggingface_hub import hf_hub_download
    relative = Path(meta["gguf_repo"].replace("/", "--")) / meta["gguf_revision"]
    cache = Path(drive_project) / "cpu_model_cache" / relative
    local = Path(local_root) / relative
    local.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    for item in meta["files"]:
        target = local / item["name"]
        cached = cache / item["name"]
        if not target.is_file():
            if cached.is_file():
                print("Restoring CPU weight file from Drive:", item["name"], flush=True)
                shutil.copyfile(cached, target)
            else:
                downloaded = Path(hf_hub_download(
                    meta["gguf_repo"], item["name"], revision=meta["gguf_revision"],
                    token=core.load_hf_token(), cache_dir="/content/fever_cpu_downloads"))
                shutil.copyfile(downloaded, target)
        if target.stat().st_size != item["bytes"] or core.sha_file(target) != item["sha256"]:
            raise ValueError("CPU model checksum mismatch: " + item["name"])
        if not cached.exists():
            staging = cache / (item["name"] + "." + str(time.time_ns()) + ".pending")
            shutil.copyfile(target, staging)
            staging.replace(cached)
    return local / meta["files"][0]["name"]

class CPUGenerator(systems.Generator):
    def __init__(self, meta, root):
        from llama_cpp import Llama
        from transformers import AutoTokenizer
        self.meta = meta
        self.parts_prompt = (Path(root) / "parts_prompt.txt").read_text()
        self.tokenizer = AutoTokenizer.from_pretrained(
            meta["repo"], revision=meta["revision"], token=core.load_hf_token(),
            trust_remote_code=False)
        weights = local_weights(meta, Path(root).parent / "drive/MyDrive/FEVER_Project")
        self.model = Llama(model_path=str(weights), n_gpu_layers=0,
            n_ctx=4096, n_batch=128, n_threads=max(1, os.cpu_count() or 1),
            seed=42, use_mmap=True, verbose=False)

    def generate(self, prompt, trace, stage, max_new_tokens=512):
        messages = [{"role": "user", "content": prompt}]
        record = {"stage": stage, "prompt": prompt, "messages": messages,
                  "backend": "llama.cpp CPU", "response": None}
        trace["calls"].append(record)
        # This template estimates packing budgets. llama.cpp uses the GGUF's
        # embedded chat template; its actual usage counts are retained below.
        budget_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        estimated = len(self.tokenizer.encode(budget_text, add_special_tokens=False))
        record["estimated_input_tokens"] = estimated
        if estimated + max_new_tokens > 4096:
            raise ValueError("Prompt exceeds CPU context budget")
        response = self.model.create_chat_completion(
            messages=messages, max_tokens=max_new_tokens, temperature=0, seed=42)
        choice = response["choices"][0]
        record["response"] = choice["message"].get("content") or ""
        record["finish_reason"] = choice["finish_reason"]
        record["usage"] = response.get("usage", {})
        if choice["finish_reason"] != "stop":
            raise ValueError("Incomplete CPU response or output-token limit")
        return record["response"]

    def classify(self, claim, selected, trace):
        result = super().classify(claim, selected, trace)
        usage = trace["calls"][-1].get("usage", {})
        return {**result, "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens")}

    def close(self):
        self.model.close()
        self.model = None
        self.tokenizer = None
        gc.collect()

class CPURobertaVerifier:
    def __init__(self, checkpoint):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.max_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True).to("cpu").eval()
        if self.model.config.num_labels != 3:
            raise ValueError("Expected three-label RoBERTa")

    def predict(self, row, context):
        text = " ".join(e["text"] for p in context["payload"]["passages"] for e in p["evidence"])
        uncut = self.tokenizer(row["claim"], text, truncation=False)["input_ids"]
        encoded = self.tokenizer(row["claim"], text, truncation="only_second",
                                 max_length=512, return_tensors="pt")
        started = time.monotonic()
        with self.torch.inference_mode():
            probabilities = self.model(**encoded).logits.softmax(-1)[0].tolist()
        return {"label": core.LABELS[max(range(3), key=lambda i: probabilities[i])],
                "probabilities": probabilities, "input_tokens": int(encoded["input_ids"].shape[1]),
                "untruncated_tokens": len(uncut), "truncated": len(uncut) > 512,
                "retained_input": self.tokenizer.decode(encoded["input_ids"][0].tolist()),
                "inference_seconds": time.monotonic() - started}

    def close(self):
        self.model = None
        self.tokenizer = None
        gc.collect()

class CPURobertaSelector(systems.RobertaSelector):
    def __init__(self, checkpoint):
        self.verifier = CPURobertaVerifier(checkpoint)
        self.tokenizer = self.verifier.tokenizer
