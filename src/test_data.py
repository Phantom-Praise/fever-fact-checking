import json
from collections import defaultdict

file_path = "data/raw/train.jsonl"

# Store every example under its claim text
claims = defaultdict(list)

with open(file_path, "r") as file:
    for line in file:
        example = json.loads(line)

        claim = example["claim"]

        claims[claim].append({
            "id": example["id"],
            "label": example["label"],
            "verifiable": example["verifiable"]
        })


# Find repeated claims
duplicate_claims = {
    claim: examples
    for claim, examples in claims.items()
    if len(examples) > 1
}


# Find repeated claims with different labels
conflicting_claims = {}

for claim, examples in duplicate_claims.items():

    labels = {example["label"] for example in examples}

    if len(labels) > 1:
        conflicting_claims[claim] = examples


print("\nFEVER DUPLICATE INVESTIGATION")
print("--------------------------------")

print("Claims appearing more than once:",
      len(duplicate_claims))

print("Repeated claims with conflicting labels:",
      len(conflicting_claims))

print("\nExample repeated claims WITH conflicting labels:")

for claim, examples in list(conflicting_claims.items())[:5]:

    print("\nCLAIM:", claim)

    for example in examples:
        print(
            "   ID:", example["id"],
            "| Label:", example["label"],
            "| Verifiable:", example["verifiable"]
        )

print("\n--------------------------------")