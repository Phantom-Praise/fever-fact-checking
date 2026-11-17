"""Local Climate-FEVER commands for VS Code. Uses installed models only.

Default command is a read-only preflight. No pip install, pull or training.
Local Ollama is a separate backend/protocol from the Colab HF/NF4 experiment.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sqlite3
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "colab"))
import climate_fever as climate
import fever_colab_runtime as core
import independent_systems as systems

CONFIG = ROOT / "configs/climate_local.json"


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_config(path=CONFIG):
    config = json.loads(Path(path).read_text())
    if config["ollama_url"] not in ("http://127.0.0.1:11434", "http://localhost:11434"):
        raise ValueError("This local runner only sends data to local Ollama")
    if config["context_tokens"] != 8192 or config["excerpt_characters"] != 600:
        raise ValueError("Changing evidence/context settings requires an adapter/protocol revision")
    return config


def session_for(config):
    session = requests.Session()
    session.trust_env = False
    return session


def inventory(config):
    session = session_for(config)
    try:
        response = session.get(config["ollama_url"] + "/api/tags", timeout=10)
        response.raise_for_status()
        version = session.get(config["ollama_url"] + "/api/version", timeout=10)
        version.raise_for_status()
        return response.json()["models"], version.json()
    finally:
        session.close()


def model_metadata(config, family):
    installed, version = inventory(config)
    requested = config["models"][family]
    matches = [m for m in installed if m["name"] == requested]
    if len(matches) != 1:
        raise ValueError(f"{requested} is not installed. No download or model substitution was attempted.")
    model = matches[0]
    return {"backend": "local_ollama", "name": model["name"], "digest": model["digest"],
            "size_bytes": model.get("size"), "details": model.get("details"),
            "ollama": version, "url": config["ollama_url"],
            "context_tokens": config["context_tokens"], "timeout_seconds": config["timeout_seconds"],
            "adapter_sha256": core.sha_file(__file__)}


class LocalCorpus:
    """Reuse the existing full sentence index in read-only mode; no rebuild."""
    def __init__(self, database):
        self.connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
        tables = {r[0] for r in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"title_fts", "page_sentences"} <= tables:
            self.connection.close()
            raise ValueError("Expected the existing title_fts/page_sentences Wikipedia index")

    def search(self, query, limit=10):
        words = list(dict.fromkeys(w for w in re.findall(r"\w+", query)
                                   if w.casefold() not in systems.STOPWORDS))[:12]
        if not words:
            return []
        expression = " OR ".join('"' + word + '"' for word in words)
        return [row[0] for row in self.connection.execute(
            "SELECT page_id FROM title_fts WHERE title_fts MATCH ? ORDER BY bm25(title_fts),page_id LIMIT ?",
            (expression, limit))]

    def excerpts(self, pages, tokenizer=None):
        excerpts = []
        for page in pages:
            for sid, sentence in self.connection.execute(
                    "SELECT sentence_id,text FROM page_sentences WHERE page_id=? ORDER BY sentence_id", (page,)):
                if not sentence.strip():
                    continue
                if tokenizer is None:
                    pieces = [(sentence[start:start + 600], {"source_char_start": start,
                               "source_char_end": min(start + 600, len(sentence))})
                              for start in range(0, len(sentence), 600)]
                else:
                    tokens = tokenizer.encode(sentence, add_special_tokens=False)
                    pieces = [(tokenizer.decode(tokens[start:start + 160]),
                               {"sentence_token_start": start, "sentence_token_end": min(start + 160, len(tokens))})
                              for start in range(0, len(tokens), 160)]
                for text, offsets in pieces:
                    excerpts.append({"id": "E" + str(len(excerpts) + 1), "page_id": page,
                                     "sentence_id": sid, "text": text, "source_sentence": sentence, **offsets})
        return excerpts

    def close(self):
        self.connection.close()


class OllamaGenerator(climate.ClimateGenerator):
    def __init__(self, meta, root):
        self.meta = meta
        self.tokenizer = None  # Ollama token counts are recorded after each call.
        self.session = requests.Session()
        self.session.trust_env = False

    def fits(self, prompt, output_tokens):
        # Conservative byte-based guard, not a claim of exact tokenization.
        # Reserve additional room for Ollama's chat template.
        return len(prompt.encode("utf-8")) + output_tokens + 512 < self.meta["context_tokens"]

    def generate(self, prompt, trace, stage, max_new_tokens=512):
        record = {"stage": stage, "prompt": prompt, "response": None}
        trace["calls"].append(record)
        if not self.fits(prompt, max_new_tokens):
            raise ValueError("Local conservative context guard failed; no silent evidence truncation")
        payload = {"model": self.meta["name"], "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "format": "json", "keep_alive": "5m",
                   "options": {"num_ctx": self.meta["context_tokens"], "num_predict": max_new_tokens,
                               "temperature": 0, "seed": 42}}
        record["request"] = payload
        response = self.session.post(self.meta["url"] + "/api/chat", json=payload,
                                     timeout=(10, self.meta["timeout_seconds"]))
        response.raise_for_status()
        body = response.json()
        record.update(raw_response=body, response=body.get("message", {}).get("content", ""),
                      input_tokens=body.get("prompt_eval_count"), output_tokens=body.get("eval_count"),
                      stopped_normally=body.get("done") is True and body.get("done_reason") == "stop")
        if not record["stopped_normally"]:
            raise ValueError("Incomplete or output-token-limited Ollama response")
        if not isinstance(record["input_tokens"], int) or not isinstance(record["output_tokens"], int):
            raise ValueError("Ollama token accounting missing")
        if record["input_tokens"] + record["output_tokens"] >= self.meta["context_tokens"]:
            raise ValueError("Context limit reached; possible truncation")
        return record["response"]

    def batches(self, claim, excerpts):
        batch = []
        for excerpt in excerpts:
            if not self.fits(self.selection_prompt(claim, batch + [excerpt]), 512):
                if not batch:
                    raise ValueError("One excerpt exceeds local selection budget")
                yield batch
                batch = []
                if not self.fits(self.selection_prompt(claim, [excerpt]), 512):
                    raise ValueError("One excerpt exceeds local selection budget")
            batch.append(excerpt)
        if batch:
            yield batch

    def close(self):
        # Release model memory after this session; no ongoing generation remains.
        try:
            self.session.post(self.meta["url"] + "/api/chat",
                              json={"model": self.meta["name"], "messages": [], "keep_alive": 0}, timeout=15)
        except requests.RequestException:
            pass
        finally:
            self.session.close()


class LocalRobertaSelector(systems.RobertaSelector):
    def __init__(self, checkpoint):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        verifier = object.__new__(core.RobertaVerifier)
        verifier.torch, verifier.max_length, verifier.path = torch, 512, Path(checkpoint)
        verifier.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        verifier.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False).to(device).eval()
        if verifier.model.config.num_labels != 3:
            raise ValueError("Expected the original three-class RoBERTa checkpoint")
        self.verifier, self.tokenizer = verifier, verifier.tokenizer


def local_environment():
    import torch
    return {"backend": "local_ollama_or_torch", "machine": platform.machine(),
            "system": platform.system(), "mps_available": torch.backends.mps.is_available(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "requests", "scikit-learn")},
            "adapter_sha256": core.sha_file(__file__), "generative_chunking": "600 characters",
            "context_guard": "UTF-8 bytes + output reserve + 512 template reserve < 8192; not exact preflight tokens"}


def inspect(config):
    rows, manifest = climate.verify_data(resolve(config["data"]))
    database = resolve(config["database"])
    corpus = LocalCorpus(database)
    try:
        example = corpus.search(rows[0]["claim"])
    finally:
        corpus.close()
    try:
        installed, version = inventory(config)
        names = {m["name"] for m in installed}
        error = None
    except requests.RequestException as exc:
        names, version, error = set(), None, str(exc)
    return {"dataset": "Climate-FEVER", "claims": len(rows), "labels": manifest["label_counts"],
            "database": str(database), "database_bytes": database.stat().st_size,
            "evidence_source": "Existing full FEVER Wikipedia sentence index; no per-claim gold input",
            "example_search_pages": example, "ollama": version, "ollama_error": error,
            "models": {family: {"name": name, "installed": name in names} for family, name in config["models"].items()},
            "roberta_checkpoint": config.get("roberta_checkpoint"),
            "inference_started": False, "downloads_started": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "pilot", "run", "report", "trace"), nargs="?", default="inspect")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--family", choices=list(systems.MODELS) + ["roberta"], default="llama")
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot", help="Scope for report/trace")
    parser.add_argument("--claim-id", help="Saved trace ID, e.g. climate-0")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.command == "inspect":
        print(json.dumps(inspect(config), indent=2))
        return
    rows, manifest = climate.verify_data(resolve(config["data"]))
    scope = "pilot" if args.command == "pilot" else "full" if args.command == "run" else args.scope
    if scope == "pilot":
        rows = rows[:12]
    output = resolve(config["output"]) / ("pilot12_v1" if scope == "pilot" else "all1535_v1")
    if args.command == "trace":
        if args.claim_id and not re.fullmatch(r"climate-\d+", args.claim_id):
            parser.error("Invalid claim ID")
        paths = sorted((output / args.family / "predictions").glob("*.json"))
        path = (output / args.family / "predictions" / (args.claim_id + ".json")) if args.claim_id else paths[-1] if paths else None
        if path is None or not path.exists():
            parser.error("No saved trace at this scope/family")
        print(path.read_text())
        return
    if args.command in ("pilot", "run"):
        checkpoint = resolve(config["roberta_checkpoint"]) if config.get("roberta_checkpoint") else None
        if args.family == "roberta" and (checkpoint is None or not (checkpoint / "config.json").exists()):
            parser.error("Set roberta_checkpoint to the verified last-3-block checkpoint directory; no replacement checkpoint is downloaded.")
        metadata = model_metadata(config, args.family) if args.family != "roberta" else None
        print(f"{scope}: {len(rows)} claims | {args.family} | local only | output {output}", flush=True)
        status = climate.run(rows, ROOT, output, resolve(config["database"]),
            {"dataset_sha256": manifest["source_sha256"], "source": "Existing historical FEVER Wikipedia sentence index"},
            args.family, metadata, checkpoint, minutes=args.minutes, generator_class=OllamaGenerator,
            roberta_class=LocalRobertaSelector, corpus_class=LocalCorpus, execution_environment=local_environment())
        print("STATUS:", status, flush=True)
    labels = json.loads((resolve(config["data"]) / "labels.json").read_text())
    labels = {row["id"]: labels[row["id"]] for row in rows}
    results = climate.report(rows, labels, output, [args.family])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
