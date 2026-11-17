import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "FacebookAI/roberta-base"
TRAIN_FILE = Path("data/processed/train.jsonl")

LABEL_MAP = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}


def load_sample(file_path):
    """Load one FEVER example for a model pipeline test."""

    with open(file_path, "r", encoding="utf-8") as file:
        example = json.loads(file.readline())

    return example


def main():

    print("Loading RoBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading RoBERTa classification model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    example = load_sample(TRAIN_FILE)

    claim = example["claim"]

    print("\nFEVER SAMPLE")
    print("--------------------------------")
    print("Claim:", claim)
    print("Gold label:", example["label"])

    inputs = tokenizer(
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    print("\nTOKENIZATION")
    print("--------------------------------")
    print("Input shape:", inputs["input_ids"].shape)
    print("Number of tokens:", inputs["input_ids"].shape[1])

    # Test one forward pass.
    with torch.no_grad():
        outputs = model(**inputs)

    print("\nMODEL OUTPUT")
    print("--------------------------------")
    print("Logits shape:", outputs.logits.shape)
    print("Logits:", outputs.logits)

    print("\nRoBERTa baseline pipeline test passed.")


if __name__ == "__main__":
    main()