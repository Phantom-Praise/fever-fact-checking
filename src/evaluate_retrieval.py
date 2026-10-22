
import json
from pathlib import Path

GOLD_PATH = Path("data/processed/validation.jsonl")
RETRIEVED_DIR = Path("data/processed/retrieved")

STRATEGIES = ["and", "or", "hybrid"]
LIMIT = 100
TOP_K = 5


def gold_evidence_groups(example):
    groups = []

    for group in example.get("evidence", []):
        references = set()

        for item in group:
            if len(item) >= 4:
                page_id = item[2]
                sentence_id = item[3]

                if page_id is not None and sentence_id is not None:
                    references.add((page_id, int(sentence_id)))

        if references:
            groups.append(references)

    return groups


def load_gold():
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Gold FEVER data not found: {GOLD_PATH}. "
            "Update GOLD_PATH to your actual raw validation file."
        )

    with GOLD_PATH.open(encoding="utf-8") as f:
        return {
            str(ex["id"]): ex
            for ex in map(json.loads, f)
        }


def evaluate(strategy, gold_by_id):
    path = RETRIEVED_DIR / (
        f"validation_first_{LIMIT}_top{TOP_K}_{strategy}.jsonl"
    )

    eligible = 0
    any_sentence_hits = 0
    complete_group_hits = 0
    missing_gold = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            example = gold_by_id.get(str(result["id"]))

            if example is None:
                missing_gold += 1
                continue

            if example["label"] == "NOT ENOUGH INFO":
                continue

            groups = gold_evidence_groups(example)

            if not groups:
                continue

            retrieved = {
                (item["page_id"], int(item["sentence_id"]))
                for item in result["retrieved_evidence"]
            }

            eligible += 1

            if any(retrieved & group for group in groups):
                any_sentence_hits += 1

            if any(group <= retrieved for group in groups):
                complete_group_hits += 1

    print(f"\nStrategy: {strategy.upper()}")
    print("Eligible verifiable claims:", eligible)
    print("IDs not found in gold data:", missing_gold)

    if eligible:
        print(
            "Any gold sentence recall@5:",
            f"{any_sentence_hits / eligible:.2%}",
        )
        print(
            "Complete evidence group recall@5:",
            f"{complete_group_hits / eligible:.2%}",
        )


def main():
    gold_by_id = load_gold()

    for strategy in STRATEGIES:
        evaluate(strategy, gold_by_id)


if __name__ == "__main__":
    main()
