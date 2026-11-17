
import json
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

CHECKPOINT = "checkpoints/roberta_epoch_1"
VALIDATION_FILE = Path("data/processed/validation.jsonl")
MAX_EXAMPLES = 2000
BATCH_SIZE = 8

LABEL_NAMES = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


class FeverDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        encoded = self.tokenizer(
            example["claim"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }

        item["labels"] = torch.tensor(
            int(example["label_id"]),
            dtype=torch.long,
        )

        return item


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    examples = []

    with open(VALIDATION_FILE, encoding="utf-8") as file:
        for line in file:
            if len(examples) >= MAX_EXAMPLES:
                break
            examples.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT
    ).to(device)

    model.eval()

    loader = DataLoader(
        FeverDataset(examples, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    gold_labels = []
    predictions = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            predicted = outputs.logits.argmax(dim=-1)

            gold_labels.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())

    print("\nBASELINE DIAGNOSTICS")
    print("-" * 45)

    print("Examples:", len(gold_labels))
    print("Gold distribution:", Counter(gold_labels))
    print("Predicted distribution:", Counter(predictions))

    print(
        "\nAccuracy:",
        round(accuracy_score(gold_labels, predictions), 4),
    )
    print(
        "Macro-F1:",
        round(
            f1_score(
                gold_labels,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            ),
            4,
        ),
    )

    print("\nClassification report:")
    print(
        classification_report(
            gold_labels,
            predictions,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            gold_labels,
            predictions,
            labels=[0, 1, 2],
        )
    )


if __name__ == "__main__":
    main()
