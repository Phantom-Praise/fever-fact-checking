import csv
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "FacebookAI/roberta-base"

TRAIN_FILE = Path("data/processed/train.jsonl")
VALIDATION_FILE = Path("data/processed/validation.jsonl")

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

NUM_LABELS = 3
MAX_LENGTH = 256

# Smaller batch size is safer on a MacBook GPU.
BATCH_SIZE = 8

LEARNING_RATE = 2e-5

# This is an implementation/sanity-check run,
# not the final experiment.
EPOCHS = 1

TRAIN_FRACTION = 0.20

# Keep validation reasonably quick for this test run.
MAX_VALIDATION_EXAMPLES = 2000

RANDOM_SEED = 42


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class FeverDataset(Dataset):

    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        encoding = self.tokenizer(
            example["claim"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                example["label_id"],
                dtype=torch.long
            )
        }


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_jsonl(file_path):

    examples = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            examples.append(json.loads(line))

    return examples


def sample_training_data(examples, fraction):

    random.seed(RANDOM_SEED)

    sample_size = int(
        len(examples) * fraction
    )

    return random.sample(
        examples,
        sample_size
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, dataloader, device):

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            batch_predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            predictions.extend(
                batch_predictions.cpu().tolist()
            )

            labels.extend(
                batch_labels.cpu().tolist()
            )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    return accuracy, macro_f1


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------

def main():

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------
    # Device
    # ----------------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    print("\nFEVER RoBERTa Training")
    print("--------------------------------")
    print("Device:", device)

    # ----------------------------------------------
    # Load data
    # ----------------------------------------------

    print("\nLoading datasets...")

    train_examples = load_jsonl(
        TRAIN_FILE
    )

    validation_examples = load_jsonl(
        VALIDATION_FILE
    )

    train_examples = sample_training_data(
        train_examples,
        TRAIN_FRACTION
    )

    validation_examples = (
        validation_examples[
            :MAX_VALIDATION_EXAMPLES
        ]
    )

    print(
        "Training examples:",
        len(train_examples)
    )

    print(
        "Validation examples:",
        len(validation_examples)
    )

    # ----------------------------------------------
    # Model/tokenizer
    # ----------------------------------------------

    print("\nLoading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS
        )
    )

    model.to(device)

    # ----------------------------------------------
    # DataLoaders
    # ----------------------------------------------

    train_dataset = FeverDataset(
        train_examples,
        tokenizer
    )

    validation_dataset = FeverDataset(
        validation_examples,
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE
    )

    # ----------------------------------------------
    # Optimizer
    # ----------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ----------------------------------------------
    # Training
    # ----------------------------------------------

    log_file = RESULTS_DIR / "training_log.csv"

    print("\nStarting training...")
    print("--------------------------------")

    start_time = time.time()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for batch_number, batch in enumerate(
            train_loader,
            start=1
        ):

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            # Print progress every 100 batches.
            if batch_number % 100 == 0:

                average_loss = (
                    total_loss / batch_number
                )

                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_number}/"
                    f"{len(train_loader)} | "
                    f"Loss: {average_loss:.4f}"
                )

        average_loss = (
            total_loss / len(train_loader)
        )

        # ------------------------------------------
        # Validation
        # ------------------------------------------

        print("\nRunning validation...")

        accuracy, macro_f1 = evaluate(
            model,
            validation_loader,
            device
        )

        elapsed_minutes = (
            time.time() - start_time
        ) / 60

        print("\nEPOCH RESULTS")
        print("--------------------------------")

        print(
            f"Training loss: "
            f"{average_loss:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Validation macro-F1: "
            f"{macro_f1:.4f}"
        )

        print(
            f"Elapsed time: "
            f"{elapsed_minutes:.2f} minutes"
        )

        # ------------------------------------------
        # CSV logging
        # ------------------------------------------

        with open(
            log_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            if file.tell() == 0:
                writer.writerow([
                    "epoch",
                    "training_loss",
                    "validation_accuracy",
                    "validation_macro_f1",
                    "elapsed_minutes"
                ])

            writer.writerow([
                epoch + 1,
                average_loss,
                accuracy,
                macro_f1,
                elapsed_minutes
            ])

        # ------------------------------------------
        # Save checkpoint
        # ------------------------------------------

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"roberta_epoch_{epoch + 1}"
        )

        model.save_pretrained(
            checkpoint_path
        )

        tokenizer.save_pretrained(
            checkpoint_path
        )

        print(
            "\nCheckpoint saved to:",
            checkpoint_path
        )

    print("\nTraining sanity check complete.")


if __name__ == "__main__":
    main()