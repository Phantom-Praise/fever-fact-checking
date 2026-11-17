"""Inspect FEVER claim language before designing symbolic verification rules.

Reads the exact development claims; no gold labels are used to select patterns.
Outputs representative claims for manual rule design, not classifier results.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

CANDIDATES = {
    "birthplace": r"\b(?:was|is) born in\b",
    "deathplace": r"\b(?:died in|place of death)\b",
    "occupation": r"\b(?:is|was) (?:an? )?(?:actor|actress|singer|writer|politician|director|footballer|musician|scientist|artist)\b",
    "nationality": r"\b(?:is|was) (?:an? )?(?:American|British|Canadian|French|German|Indian|Australian|Italian|Japanese)\b",
    "location": r"\b(?:is|was) (?:located|situated|based) in\b",
    "membership": r"\b(?:member of|part of|belongs to|played for|worked for)\b",
    "founded": r"\b(?:was founded|founded|established)\b",
    "date": r"\b(?:in|on) (?:the year )?\d{4}\b",
    "family": r"\b(?:married to|spouse of|son of|daughter of|father of|mother of)\b",
}
COMPILED = {key: re.compile(pattern, re.I) for key, pattern in CANDIDATES.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/retrieved/validation_first_100_top5_title_fast.jsonl")
    parser.add_argument("--output", default="reports/part2/symbolic/claim_patterns.json")
    parser.add_argument("--examples-per-category", type=int, default=6)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    buckets = {name: [] for name in CANDIDATES}
    unmatched = []
    for record in records:
        claim = record["claim"]
        hits = [name for name, pattern in COMPILED.items() if pattern.search(claim)]
        if not hits:
            unmatched.append({"id": record["id"], "claim": claim})
        for name in hits:
            buckets[name].append({"id": record["id"], "claim": claim})
    summary = {
        "n": len(records),
        "note": "Keyword matches are exploratory and do NOT establish that claims can be safely verified.",
        "category_counts": {name: len(rows) for name, rows in buckets.items()},
        "unmatched_count": len(unmatched),
        "examples": {name: rows[:args.examples_per_category] for name, rows in buckets.items()},
        "unmatched_examples": unmatched[:20],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
