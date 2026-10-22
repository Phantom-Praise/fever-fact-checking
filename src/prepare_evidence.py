
import json
from pathlib import Path

DATA_DIR = Path("data/processed")
WIKI_DIR = Path("data/raw/wiki-pages")
OUTPUT_DIR = Path("data/processed/evidence")

SPLITS = ["train", "validation"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def evidence_keys(example):
    """Collect all valid gold evidence sentence references."""
    keys = set()

    for group in example.get("evidence", []):
        for item in group:
            if len(item) < 4:
                continue

            page_id, sentence_id = item[2], item[3]

            if page_id is not None and sentence_id is not None:
                keys.add((str(page_id), str(sentence_id)))

    return keys


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    examples_by_split = {}
    needed = set()

    for split in SPLITS:
        examples = list(load_jsonl(DATA_DIR / f"{split}.jsonl"))
        examples_by_split[split] = examples

        for example in examples:
            needed.update(evidence_keys(example))

        print(f"{split}: {len(examples):,} claims")

    print(f"Unique evidence references: {len(needed):,}")

    evidence_lookup = {}
    files = sorted(WIKI_DIR.glob("wiki-*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"No Wikipedia JSONL files found in {WIKI_DIR}"
        )

    for index, path in enumerate(files, 1):
        print(f"Scanning {path.name} ({index}/{len(files)})", flush=True)

        for page in load_jsonl(path):
            page_id = str(page.get("id", ""))

            if not page_id:
                continue

            for line in page.get("lines", "").split("\n"):
                parts = line.split("\t", 2)

                if len(parts) < 2:
                    continue

                sentence_id = parts[0]
                sentence_text = parts[1].strip()
                key = (page_id, sentence_id)

                if key in needed and sentence_text:
                    evidence_lookup[key] = sentence_text

    print(
        f"Found {len(evidence_lookup):,} / "
        f"{len(needed):,} referenced sentences"
    )

    for split, examples in examples_by_split.items():
        output_path = OUTPUT_DIR / f"{split}.jsonl"
        missing_claims = 0
        written = 0

        with open(output_path, "w", encoding="utf-8") as out:
            for example in examples:
                label_id = int(example["label_id"])

                # Choose one complete gold evidence group, rather
                # than combining unrelated alternative groups.
                candidates = []

                for group in example.get("evidence", []):
                    keys = []
                    for item in group:
                        if len(item) >= 4 and item[2] is not None \
                                and item[3] is not None:
                            keys.append((str(item[2]), str(item[3])))

                    if keys and all(k in evidence_lookup for k in keys):
                        candidates.append(keys)

                if label_id != 2 and not candidates:
                    missing_claims += 1
                    continue

                selected = candidates[0] if candidates else []
                evidence_text = " ".join(
                    evidence_lookup[key] for key in selected
                )

                record = {
                    "id": example["id"],
                    "claim": example["claim"],
                    "evidence_text": evidence_text,
                    "label_id": label_id,
                }

                out.write(json.dumps(record) + "\n")
                written += 1

        print(
            f"{split}: saved {written:,} examples; "
            f"skipped {missing_claims:,} without complete gold evidence"
        )
        print("Output:", output_path)

    print("\nEvidence preparation complete.")


if __name__ == "__main__":
    main()
