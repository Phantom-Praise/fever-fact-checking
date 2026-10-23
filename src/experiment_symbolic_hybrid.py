"""Conservative Wikidata symbolic-stage experiment (RQ3/RQ4).

Uses the same saved FEVER retrieval predictions in both conditions.
Symbolic decisions are SUPPORTS only for unambiguous, matched birth/death
locations, dates, occupations, and entity types in Wikidata; unsupported/ambiguous claims abstain. This avoids
open-world REFUTES assertions. Network timing is recorded separately.
"""
import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import requests
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "FEVER-Fact-Checking-Research/1.0 (academic experiment; contact via GitHub Phantom-Praise)"}
PATTERNS = [
    (re.compile(r"^(.+?) (?:was|is) born in (.+?)\.?$", re.I), "P19"),
    (re.compile(r"^(.+?) died in (.+?)\.?$", re.I), "P20"),
]
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


class WikidataVerifier:
    def __init__(self, timeout=8):
        self.session = requests.Session()
        self.timeout = timeout
        self.cache = {}
        self.calls = 0
        self.failures = 0

    def query(self, **params):
        key = json.dumps(params, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        self.calls += 1
        try:
            response = self.session.get(API, params={**params, "format": "json"},
                                        headers=HEADERS, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise ValueError(data["error"])
            self.cache[key] = data
            return data
        except (requests.RequestException, ValueError) as exc:
            self.failures += 1
            print(f"Wikidata request failed: {exc}", flush=True)
            self.cache[key] = {}
            return {}

    def search_exact(self, name):
        data = self.query(action="wbsearchentities", search=name,
                          language="en", limit=5)
        matches = [item for item in data.get("search", [])
                   if normalize(item.get("label", "")) == normalize(name)]
        # Ambiguous names must abstain, not arbitrarily choose an entity.
        return matches[0]["id"] if len(matches) == 1 else None

    def entity(self, entity_id):
        data = self.query(action="wbgetentities", ids=entity_id,
                          props="claims|labels|aliases", languages="en")
        return data.get("entities", {}).get(entity_id, {})

    def claim_entity_ids(self, entity, property_id):
        values = set()
        for statement in entity.get("claims", {}).get(property_id, []):
            if statement.get("rank") == "deprecated":
                continue
            value = statement.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            if isinstance(value, dict) and value.get("id"):
                values.add(value["id"])
        return values

    def verify_typed(self, claim):
        occupation = OCCUPATION_PATTERN.fullmatch(claim.strip())
        entity_type = TYPE_PATTERN.fullmatch(claim.strip())
        match = occupation or entity_type
        if not match:
            return None
        subject = match.group(1).strip()
        term = match.group(2).casefold()
        target = OCCUPATIONS[term] if occupation else TYPES[term]
        subject_id = self.search_exact(subject)
        if not subject_id:
            return None, "unresolved_or_ambiguous_subject"
        entity = self.entity(subject_id)
        property_id = "P106" if occupation else "P31"
        values = self.claim_entity_ids(entity, property_id)
        if target in values:
            return 0, f"wikidata_{property_id}_exact_match"
        return None, "occupation_or_type_not_matched"

    def verify_date(self, claim):
        for pattern, property_id in DATE_PATTERNS:
            match = pattern.fullmatch(claim.strip())
            if not match:
                continue
            subject_id = self.search_exact(match.group(1).strip())
            if not subject_id:
                return None, "unresolved_or_ambiguous_subject"
            entity = self.entity(subject_id)
            dates = set()
            for statement in entity.get("claims", {}).get(property_id, []):
                if statement.get("rank") == "deprecated":
                    continue
                value = statement.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                if isinstance(value, dict) and value.get("time"):
                    dates.add(value["time"])
            if len(dates) != 1:
                return None, "missing_or_ambiguous_date"
            date = next(iter(dates))
            year_match = re.match(r"^[+-](\\d{4,})-", date)
            if not year_match:
                return None, "unparseable_date"
            year = int(year_match.group(1))
            term = match.group(2).strip().lower()
            if term.isdigit() and len(term) == 4:
                # Decade claims are only true when year falls within [decade, decade+9].
                return (0, f"wikidata_{property_id}_decade_match") if int(term) <= year < int(term) + 10 else (None, "date_not_matched")
            decades = {"sixties": 1960, "seventies": 1970, "eighties": 1980, "nineties": 1990}
            if term in decades:
                start = decades[term]
                return (0, f"wikidata_{property_id}_decade_match") if start <= year < start + 10 else (None, "date_not_matched")
            # Exact dates: avoid making claims from partial precision.
            import datetime
            try:
                expected = datetime.datetime.strptime(term.replace(",", ""), "%B %d %Y")
            except ValueError:
                return None, "unsupported_date_format"
            date_match = re.match(r"^[+-]\\d{4,}-(\\d{2})-(\\d{2})T", date)
            if date_match and year == expected.year and int(date_match.group(1)) == expected.month and int(date_match.group(2)) == expected.day:
                return 0, f"wikidata_{property_id}_exact_date_match"
            return None, "date_not_matched"
        return None

    def verify(self, claim):
        for handler in (self.verify_typed, self.verify_date):
            result = handler(claim)
            if result is not None:
                return result
        for pattern, property_id in PATTERNS:
            match = pattern.fullmatch(claim.strip())
            if not match:
                continue
            subject, object_text = match.group(1).strip(), match.group(2).strip()
            subject_id = self.search_exact(subject)
            if not subject_id:
                return None, "unresolved_or_ambiguous_subject"
            entity = self.entity(subject_id)
            statements = entity.get("claims", {}).get(property_id, [])
            ids = set()
            for statement in statements:
                if statement.get("rank") == "deprecated":
                    continue
                mainsnak = statement.get("mainsnak", {})
                value = mainsnak.get("datavalue", {}).get("value", {})
                if isinstance(value, dict) and value.get("id"):
                    ids.add(value["id"])
            if not ids:
                return None, "no_structured_property"
            # Require one unique value to avoid disputed or multi-valued facts.
            if len(ids) != 1:
                return None, "multiple_property_values"
            place = self.entity(next(iter(ids)))
            names = [place.get("labels", {}).get("en", {}).get("value", "")]
            names += [alias.get("value", "") for alias in
                      place.get("aliases", {}).get("en", [])]
            if normalize(object_text) in {normalize(name) for name in names if name}:
                return 0, f"wikidata_{property_id}_exact_match"
            # Wikidata is open-world; a non-match is NOT sufficient for REFUTES.
            return None, "object_not_matched"
        return None, "unsupported_claim_pattern"


def load_predictions(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if r["condition"] == "retrieved_evidence"]
    return {str(r["id"]): int(r["predicted_label_id"]) for r in selected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved", default="data/processed/retrieved/validation_first_100_top5_title_fast.jsonl")
    parser.add_argument("--predictions", default="reports/part2/paired/predictions.csv")
    parser.add_argument("--output", default="reports/part2/symbolic")
    parser.add_argument("--timeout", type=int, default=8)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.retrieved).read_text().splitlines() if line.strip()]
    predictions = load_predictions(args.predictions)
    if not records or any(str(r["id"]) not in predictions for r in records):
        raise ValueError("Retrieved claims and saved predictions must contain identical IDs")
    verifier = WikidataVerifier(args.timeout)
    rows = []
    for i, record in enumerate(records, 1):
        start = time.perf_counter()
        symbolic_prediction, reason = verifier.verify(record["claim"])
        elapsed = time.perf_counter() - start
        baseline = predictions[str(record["id"])]
        hybrid = symbolic_prediction if symbolic_prediction is not None else baseline
        rows.append({
            "id": record["id"], "claim": record["claim"],
            "gold_label_id": int(record["label_id"]),
            "baseline_prediction": baseline,
            "hybrid_prediction": hybrid,
            "symbolic_prediction": "" if symbolic_prediction is None else symbolic_prediction,
            "symbolic_resolved": int(symbolic_prediction is not None),
            "reason": reason, "symbolic_seconds": elapsed,
            "retrieval_calls_saved": int(symbolic_prediction is not None),
            "transformer_calls_saved": int(symbolic_prediction is not None),
        })
        if i % 10 == 0:
            print(f"{i}/{len(records)} | symbolic resolved={sum(r['symbolic_resolved'] for r in rows)}", flush=True)
    gold = [r["gold_label_id"] for r in rows]
    base = [r["baseline_prediction"] for r in rows]
    hybrid = [r["hybrid_prediction"] for r in rows]
    resolved = sum(r["symbolic_resolved"] for r in rows)
    summary = {
        "n": len(rows), "symbolic_resolved": resolved,
        "symbolic_coverage": resolved / len(rows),
        "symbolic_correct": sum(r["symbolic_resolved"] and
                                r["hybrid_prediction"] == r["gold_label_id"] for r in rows),
        "estimated_retrieval_calls_avoided": resolved,
        "estimated_transformer_calls_avoided": resolved,
        "wikidata_http_calls": verifier.calls,
        "wikidata_http_failures": verifier.failures,
        "symbolic_wall_seconds": sum(r["symbolic_seconds"] for r in rows),
        "reason_counts": dict(Counter(r["reason"] for r in rows)),
        "baseline": {"accuracy": accuracy_score(gold, base),
                     "macro_f1": f1_score(gold, base, labels=[0, 1, 2],
                                          average="macro", zero_division=0),
                     "confusion_matrix": confusion_matrix(gold, base, labels=[0, 1, 2]).tolist()},
        "hybrid": {"accuracy": accuracy_score(gold, hybrid),
                   "macro_f1": f1_score(gold, hybrid, labels=[0, 1, 2],
                                        average="macro", zero_division=0),
                   "confusion_matrix": confusion_matrix(gold, hybrid, labels=[0, 1, 2]).tolist()},
        "limitations": [
            "Hybrid reuses saved retrieval-model predictions; no live retrieval or transformer timing.",
            "Calls avoided are counterfactual counts, not measured latency savings.",
            "Wikidata live HTTP response times and contents can change.",
            "Symbolic rules cover exact birth/death places, occupations, entity types and selected dates; abstain otherwise.",
            "A correct Wikidata property is not necessarily contemporaneous with FEVER annotations.",
            "Development sample of 100 claims; not a held-out final test."
        ],
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in
                      ("n", "symbolic_resolved", "symbolic_coverage",
                       "symbolic_correct", "wikidata_http_calls",
                       "symbolic_wall_seconds", "baseline", "hybrid")}, indent=2))
    print(f"Saved {out / 'summary.json'} and predictions.csv")


if __name__ == "__main__":
    main()
