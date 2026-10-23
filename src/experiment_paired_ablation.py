"""Paired FEVER RQ3 evaluation and evidence ablations.

All conditions use identical claim IDs, label order, max length, batch size,
device and evaluation split. Checkpoints are frozen; no training is performed.
"""
import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
DEFAULT_RETRIEVED = "data/processed/retrieved/validation_first_100_top5_title_fast.jsonl"


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def predict(checkpoint, records, evidence, device, batch_size, max_length):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    if model.config.num_labels != 3:
        raise ValueError(f"Expected 3 labels in {checkpoint}")
    model.eval()
    predictions = []
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(0, len(records), batch_size):
            batch = records[index:index + batch_size]
            claims = [row["claim"] for row in batch]
            if evidence is None:
                inputs = tokenizer(claims, padding="max_length",
                                   truncation=True, max_length=max_length,
                                   return_tensors="pt")
            else:
                inputs = tokenizer(claims, evidence[index:index + batch_size],
                                   padding="max_length", truncation=True,
                                   max_length=max_length, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            predictions.extend(model(**inputs).logits.argmax(-1).cpu().tolist())
    synchronize(device)
    elapsed = time.perf_counter() - started
    del model
    return predictions, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved", default=DEFAULT_RETRIEVED)
    parser.add_argument("--claim-checkpoint", default="checkpoints/tuning_weighted/H2_seed42/best")
    parser.add_argument("--evidence-checkpoint", default="checkpoints/roberta_evidence/epoch_1")
    parser.add_argument("--gold-evidence", default="data/processed/evidence/validation.jsonl")
    parser.add_argument("--output", default="reports/part2/paired")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("batch-size and max-length must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = read_jsonl(args.retrieved)
    if not records:
        raise ValueError("Retrieved evidence file is empty")
    ids = [str(row["id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate claim IDs in retrieved file")
    gold = [int(row["label_id"]) for row in records]
    if any(label not in (0, 1, 2) for label in gold):
        raise ValueError("Invalid FEVER label_id")
    retrieved = [row.get("evidence_text", "") for row in records]
    # Deterministic derangement: no claim receives its own evidence.
    shuffled = retrieved[1:] + retrieved[:1] if len(records) > 1 else [""]
    blank = [""] * len(records)
    conditions = [
        ("claim_only_H2", args.claim_checkpoint, None),
        ("retrieved_evidence", args.evidence_checkpoint, retrieved),
        ("empty_evidence", args.evidence_checkpoint, blank),
        ("wrong_claim_evidence", args.evidence_checkpoint, shuffled),
    ]
    # Optional oracle diagnostic on the SAME claim IDs, not an end-to-end system.
    oracle_count = 0
    gold_path = Path(args.gold_evidence)
    if gold_path.exists():
        gold_by_id = {str(row["id"]): row for row in read_jsonl(gold_path)}
        oracle_count = sum(key in gold_by_id for key in ids)
        if oracle_count == len(ids):
            oracle = [gold_by_id[key]["evidence_text"] for key in ids]
            conditions.append(("gold_evidence_oracle", args.evidence_checkpoint, oracle))
        else:
            print(f"Skipping oracle: gold evidence for {oracle_count}/{len(ids)} IDs")
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device} | Claims={len(records)} | Labels={dict(Counter(gold))}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    prediction_rows = []
    for name, checkpoint, evidence in conditions:
        print(f"Running {name}...", flush=True)
        predictions, seconds = predict(checkpoint, records, evidence, device,
                                       args.batch_size, args.max_length)
        precision, recall, class_f1, support = precision_recall_fscore_support(
            gold, predictions, labels=[0, 1, 2], zero_division=0
        )
        summary = {
            "condition": name,
            "checkpoint": checkpoint,
            "n": len(records),
            "accuracy": float(accuracy_score(gold, predictions)),
            "macro_f1": float(f1_score(gold, predictions, labels=[0, 1, 2],
                                       average="macro", zero_division=0)),
            "inference_seconds": seconds,
            "inference_seconds_per_claim": seconds / len(records),
            "prediction_counts": dict(Counter(predictions)),
            "confusion_matrix": confusion_matrix(gold, predictions,
                                                  labels=[0, 1, 2]).tolist(),
            "per_class": {
                LABELS[i]: {"precision": float(precision[i]),
                            "recall": float(recall[i]),
                            "f1": float(class_f1[i]),
                            "support": int(support[i])}
                for i in range(3)
            },
        }
        summaries.append(summary)
        for row, pred in zip(records, predictions):
            prediction_rows.append({"condition": name, "id": row["id"],
                                    "gold_label_id": row["label_id"],
                                    "predicted_label_id": pred,
                                    "correct": int(pred == int(row["label_id"]))})
        print(f"  Accuracy={summary['accuracy']:.4f} | "
              f"Macro-F1={summary['macro_f1']:.4f} | "
              f"Inference={seconds:.3f}s")
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"dataset": args.retrieved, "seed": args.seed,
                   "batch_size": args.batch_size, "max_length": args.max_length,
                   "device": str(device), "label_mapping": LABELS,
                   "n": len(records), "conditions": summaries,
                   "notes": [
                       "All conditions evaluated on identical claim IDs.",
                       "Timing excludes retrieval and model loading; first condition may include device warmup.",
                       "Gold evidence is an oracle diagnostic and can leak label via empty NEI evidence.",
                       "Wrong-claim evidence is a deterministic cyclic shift, not gold evidence.",
                       "These are development results unless run on a held-out test set."
                   ]}, stream, indent=2)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "condition", "n", "accuracy", "macro_f1",
            "inference_seconds", "inference_seconds_per_claim", "checkpoint"
        ])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames}
                         for row in summaries)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(f"Saved {output / 'summary.json'}, metrics.csv, predictions.csv")


if __name__ == "__main__":
    main()
