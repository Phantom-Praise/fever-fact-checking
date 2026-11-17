import argparse
import csv
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/retrieved/validation_first_100_top5_title_fast.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/roberta_evidence/epoch_1",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError("No evaluation records found.")

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Examples: {len(records)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint
    ).to(device)
    model.eval()

    predictions = []
    confidence_scores = []

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        for start_idx in range(0, len(records), args.batch_size):
            batch = records[start_idx:start_idx + args.batch_size]

            encoded = tokenizer(
                [record["claim"] for record in batch],
                [record.get("evidence_text", "") for record in batch],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)

            predictions.extend(
                probabilities.argmax(dim=-1).cpu().tolist()
            )
            confidence_scores.extend(
                probabilities.max(dim=-1).values.cpu().tolist()
            )

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    gold = [int(record["label_id"]) for record in records]
    accuracy = accuracy_score(gold, predictions)
    macro_f1 = f1_score(
        gold, predictions, labels=[0, 1, 2],
        average="macro", zero_division=0
    )

    print("\n=== RETRIEVED-EVIDENCE EVALUATION ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Total model inference time: {elapsed:.3f}s")
    print(f"Mean model inference time: {elapsed / len(records):.4f}s/claim")
    print("\nPer-class metrics:")
    print(classification_report(
        gold, predictions,
        labels=[0, 1, 2],
        target_names=["Class 0", "Class 1", "Class 2"],
        zero_division=0,
    ))

    output_dir = Path("reports/part2/retrieved_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for record, prediction, confidence in zip(
        records, predictions, confidence_scores
    ):
        rows.append({
            "id": record.get("id"),
            "claim": record["claim"],
            "gold_label_id": record["label_id"],
            "predicted_label_id": prediction,
            "confidence": round(confidence, 6),
            "correct": int(prediction == int(record["label_id"])),
            "evidence_count": len(record.get("retrieved_evidence", [])),
        })

    with open(output_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "retrieved_evidence_roberta_zero_shot_transfer",
        "checkpoint": args.checkpoint,
        "input_file": args.input,
        "num_examples": len(records),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "total_model_inference_seconds": elapsed,
        "mean_model_inference_seconds_per_claim": elapsed / len(records),
        "note": (
            "Model inference only; excludes Wikipedia retrieval time. "
            "Checkpoint trained with gold evidence, evaluated with retrieved evidence."
        ),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
