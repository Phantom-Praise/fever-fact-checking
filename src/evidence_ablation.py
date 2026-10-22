
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

CHECKPOINT = "checkpoints/roberta_evidence/epoch_1"
DATA_FILE = Path("data/processed/evidence/validation.jsonl")
OUTPUT_FILE = Path("results/evidence_ablation.csv")

MAX_EXAMPLES = 2000
BATCH_SIZE = 8
MAX_LENGTH = 256
SEED = 42

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


class AblationDataset(Dataset):
    def __init__(self, examples, tokenizer, use_evidence):
        self.examples = examples
        self.tokenizer = tokenizer
        self.use_evidence = use_evidence

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        evidence = (
            example["evidence_text"]
            if self.use_evidence else ""
        )

        encoding = self.tokenizer(
            example["claim"],
            evidence,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {
            k: v.squeeze(0)
            for k, v in encoding.items()
        }
        item["labels"] = torch.tensor(
            int(example["label_id"]),
            dtype=torch.long,
        )
        return item


def evaluate(model, tokenizer, examples, device, use_evidence):
    dataset = AblationDataset(
        examples, tokenizer, use_evidence
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    gold = []
    predictions = []

    model.eval()

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {
                k: v.to(device) for k, v in batch.items()
            }

            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1)

            gold.extend(labels.tolist())
            predictions.extend(preds.cpu().tolist())

    accuracy = accuracy_score(gold, predictions)
    macro_f1 = f1_score(
        gold, predictions, labels=[0, 1, 2],
        average="macro", zero_division=0
    )

    print(
        classification_report(
            gold,
            predictions,
            labels=[0, 1, 2],
            target_names=LABELS,
            zero_division=0,
        )
    )

    return accuracy, macro_f1


def main():
    torch.manual_seed(SEED)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    with open(DATA_FILE, encoding="utf-8") as f:
        examples = [
            json.loads(line)
            for _, line in zip(range(MAX_EXAMPLES), f)
        ]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT
    ).to(device)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for use_evidence in [True, False]:
        condition = (
            "with_gold_evidence"
            if use_evidence else "without_evidence"
        )

        print(f"\nCondition: {condition}")

        accuracy, macro_f1 = evaluate(
            model, tokenizer, examples, device, use_evidence
        )

        print(
            f"Accuracy: {accuracy:.4f} | "
            f"Macro-F1: {macro_f1:.4f}"
        )

        results.append({
            "condition": condition,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "accuracy", "macro_f1"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
