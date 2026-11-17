
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "FacebookAI/roberta-base"
TRAIN_FILE = Path("data/processed/evidence/train.jsonl")
VAL_FILE = Path("data/processed/evidence/validation.jsonl")
OUTPUT_DIR = Path("checkpoints/roberta_evidence")
LOG_FILE = Path("results/main_training_log.csv")

SEED = 42
BATCH_SIZE = 8
MAX_LENGTH = 256
LEARNING_RATE = 2e-5
EPOCHS = 2
TRAIN_FRACTION = 0.20
MAX_VAL = 2000


def load_data(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class EvidenceDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        encoded = self.tokenizer(
            example["claim"],
            example["evidence_text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }
        item["labels"] = torch.tensor(
            int(example["label_id"]), dtype=torch.long
        )
        return item


def evaluate(model, loader, device):
    model.eval()
    gold, predictions = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}

            logits = model(**inputs).logits
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            gold.extend(labels.cpu().tolist())
            predictions.extend(
                logits.argmax(dim=-1).cpu().tolist()
            )

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(gold, predictions),
        "macro_f1": f1_score(
            gold,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ),
        "predictions": Counter(predictions),
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device, flush=True)

    train_examples = load_data(TRAIN_FILE)
    val_examples = load_data(VAL_FILE)

    random.shuffle(train_examples)
    train_examples = train_examples[
        :int(len(train_examples) * TRAIN_FRACTION)
    ]
    val_examples = val_examples[:MAX_VAL]

    print("Training examples:", len(train_examples))
    print("Validation examples:", len(val_examples))

    # Verify evidence availability by class.
    for label in range(3):
        group = [
            x for x in train_examples
            if int(x["label_id"]) == label
        ]
        empty = sum(
            not x["evidence_text"].strip() for x in group
        )
        print(
            f"Class {label}: {len(group)} examples, "
            f"{empty} with empty evidence"
        )

    counts = Counter(
        int(x["label_id"]) for x in train_examples
    )
    n = len(train_examples)

    weights = torch.tensor(
        [n / (3 * counts[i]) for i in range(3)],
        dtype=torch.float32,
        device=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    ).to(device)

    train_loader = DataLoader(
        EvidenceDataset(train_examples, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        EvidenceDataset(val_examples, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "train_loss", "val_loss",
                "val_accuracy", "val_macro_f1",
                "predicted_0", "predicted_1",
                "predicted_2", "elapsed_minutes",
            ],
        )
        writer.writeheader()

        for epoch in range(1, EPOCHS + 1):
            model.train()
            start = time.time()
            running_loss = 0.0

            for step, batch in enumerate(train_loader, 1):
                labels = batch.pop("labels").to(device)
                inputs = {
                    k: v.to(device) for k, v in batch.items()
                }

                optimizer.zero_grad()
                logits = model(**inputs).logits

                loss = F.cross_entropy(
                    logits, labels, weight=weights
                )

                if not torch.isfinite(loss).item():
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch}, "
                        f"step {step}"
                    )

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if step % 250 == 0:
                    print(
                        f"Epoch {epoch} | "
                        f"Step {step}/{len(train_loader)} | "
                        f"Loss {running_loss / step:.4f}",
                        flush=True,
                    )

            train_loss = running_loss / len(train_loader)
            metrics = evaluate(model, val_loader, device)
            elapsed = (time.time() - start) / 60
            counts_pred = metrics["predictions"]

            print(
                f"\nEpoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={metrics['loss']:.4f}, "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}",
                flush=True,
            )
            print(
                "Prediction counts:", dict(counts_pred),
                flush=True,
            )

            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "predicted_0": counts_pred.get(0, 0),
                "predicted_1": counts_pred.get(1, 0),
                "predicted_2": counts_pred.get(2, 0),
                "elapsed_minutes": round(elapsed, 2),
            })
            f.flush()

            checkpoint = OUTPUT_DIR / f"epoch_{epoch}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

    print("\nMain-model training complete.")


if __name__ == "__main__":
    main()
