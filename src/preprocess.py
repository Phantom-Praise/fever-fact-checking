import json
from pathlib import Path
from collections import Counter

from sklearn.model_selection import GroupShuffleSplit

# -----------------------------
# Configuration
# -----------------------------

INPUT_FILE = Path("data/raw/train.jsonl")
OUTPUT_DIR = Path("data/processed")


# Convert FEVER's text labels into numbers
LABEL_MAP = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}


# -----------------------------
# Load FEVER
# -----------------------------

def load_fever(file_path):
    examples = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            example = json.loads(line)

            # Make sure required fields exist
            required_fields = [
                "id",
                "claim",
                "label",
                "verifiable",
                "evidence"
            ]

            for field in required_fields:
                if field not in example:
                    raise ValueError(
                        f"Missing '{field}' on line {line_number}"
                    )

            # Make sure the label is valid
            if example["label"] not in LABEL_MAP:
                raise ValueError(
                    f"Unknown label: {example['label']}"
                )

            # Create a clean version of the example
            cleaned_example = {
                "id": example["id"],
                "claim": example["claim"].strip(),
                "label": example["label"],
                "label_id": LABEL_MAP[example["label"]],
                "verifiable": example["verifiable"],
                "evidence": example["evidence"]
            }

            examples.append(cleaned_example)

    return examples


# -----------------------------
# Dataset Summary
# -----------------------------

def print_summary(examples):

    labels = Counter(
        example["label"]
        for example in examples
    )

    print("\nPREPROCESSING SUMMARY")
    print("--------------------------------")

    print("Total valid examples:", len(examples))

    print("\nLabel mapping:")
    for label, label_id in LABEL_MAP.items():
        print(f"  {label} -> {label_id}")

    print("\nClass distribution:")
    for label, count in labels.items():
        percentage = (count / len(examples)) * 100

        print(
            f"  {label}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    print("--------------------------------")


def split_dataset(examples):
    """
    Split FEVER into approximately:
    80% training
    10% validation
    10% testing

    Examples with identical claim text are kept
    in the same split to prevent claim leakage.
    """

    groups = [
        example["claim"]
        for example in examples
    ]

    # --------------------------------
    # First split: 80% train / 20% temp
    # --------------------------------

    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=0.80,
        random_state=42
    )

    train_indices, temp_indices = next(
        first_split.split(
            examples,
            groups=groups
        )
    )

    train_examples = [
        examples[i]
        for i in train_indices
    ]

    temp_examples = [
        examples[i]
        for i in temp_indices
    ]

    # --------------------------------
    # Second split: temp → 50/50
    #
    # Gives approximately:
    # 10% validation
    # 10% test
    # --------------------------------

    temp_groups = [
        example["claim"]
        for example in temp_examples
    ]

    second_split = GroupShuffleSplit(
        n_splits=1,
        train_size=0.50,
        random_state=42
    )

    validation_indices, test_indices = next(
        second_split.split(
            temp_examples,
            groups=temp_groups
        )
    )

    validation_examples = [
        temp_examples[i]
        for i in validation_indices
    ]

    test_examples = [
        temp_examples[i]
        for i in test_indices
    ]

    return (
        train_examples,
        validation_examples,
        test_examples
    )



def verify_splits(
    train_examples,
    validation_examples,
    test_examples
):
    """
    Check split sizes, label distributions,
    and claim overlap.
    """

    print("\nDATASET SPLIT")
    print("--------------------------------")

    total = (
        len(train_examples)
        + len(validation_examples)
        + len(test_examples)
    )

    splits = {
        "Train": train_examples,
        "Validation": validation_examples,
        "Test": test_examples
    }

    for name, split in splits.items():

        percentage = (
            len(split) / total
        ) * 100

        print(
            f"{name}: "
            f"{len(split)} examples "
            f"({percentage:.2f}%)"
        )

        label_counts = Counter(
            example["label"]
            for example in split
        )

        for label, count in label_counts.items():

            label_percentage = (
                count / len(split)
            ) * 100

            print(
                f"   {label}: "
                f"{count} "
                f"({label_percentage:.2f}%)"
            )

    # --------------------------------
    # Leakage check
    # --------------------------------

    train_claims = {
        example["claim"]
        for example in train_examples
    }

    validation_claims = {
        example["claim"]
        for example in validation_examples
    }

    test_claims = {
        example["claim"]
        for example in test_examples
    }

    train_validation_overlap = (
        train_claims & validation_claims
    )

    train_test_overlap = (
        train_claims & test_claims
    )

    validation_test_overlap = (
        validation_claims & test_claims
    )

    print("\nCLAIM LEAKAGE CHECK")
    print("--------------------------------")

    print(
        "Train ↔ Validation overlap:",
        len(train_validation_overlap)
    )

    print(
        "Train ↔ Test overlap:",
        len(train_test_overlap)
    )

    print(
        "Validation ↔ Test overlap:",
        len(validation_test_overlap)
    )

    print("--------------------------------")

def save_jsonl(examples, output_path):
    """
    Save examples in JSON Lines format.
    """

    with open(output_path, "w", encoding="utf-8") as file:

        for example in examples:
            json.dump(example, file)
            file.write("\n")



# -----------------------------
# Main
# -----------------------------

def main():

    print("Loading FEVER dataset...")

    examples = load_fever(INPUT_FILE)

    print_summary(examples)

    print("\nCreating group-aware train/validation/test split...")

    (
        train_examples,
        validation_examples,
        test_examples
    ) = split_dataset(examples)

    verify_splits(
        train_examples,
        validation_examples,
        test_examples
    )


    print("\nSaving processed datasets...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_jsonl(
        train_examples,
        OUTPUT_DIR / "train.jsonl"
    )

    save_jsonl(
        validation_examples,
        OUTPUT_DIR / "validation.jsonl"
    )

    save_jsonl(
        test_examples,
        OUTPUT_DIR / "test.jsonl"
    )

    print("Saved:")
    print("  data/processed/train.jsonl")
    print("  data/processed/validation.jsonl")
    print("  data/processed/test.jsonl")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()