import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


WIKI_DIR = Path("data/raw/wiki-pages")

# Implementation-stage subset.
# We can scale this later.
NUM_WIKI_FILES = 5

TOP_K = 5


def tokenize(text):
    """
    Simple tokenizer for BM25.
    Converts text to lowercase words.
    """
    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def parse_sentences(lines):
    """
    Convert FEVER's Wikipedia 'lines' field
    into sentence IDs and sentence text.
    """

    sentences = []

    for line in lines.split("\n"):

        if not line.strip():
            continue

        parts = line.split("\t")

        if len(parts) < 2:
            continue

        sentence_id = parts[0]
        sentence_text = parts[1].strip()

        if not sentence_text:
            continue

        sentences.append(
            (sentence_id, sentence_text)
        )

    return sentences


def load_wikipedia_corpus():
    """
    Load sentences from a subset of the
    FEVER Wikipedia corpus.
    """

    documents = []

    wiki_files = sorted(
        WIKI_DIR.glob("wiki-*.jsonl")
    )[:NUM_WIKI_FILES]

    print(
        f"Loading {len(wiki_files)} "
        f"Wikipedia files..."
    )

    for file_number, wiki_file in enumerate(
        wiki_files,
        start=1
    ):

        print(
            f"Reading {wiki_file.name} "
            f"({file_number}/{len(wiki_files)})"
        )

        with open(
            wiki_file,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                page = json.loads(line)

                page_id = page.get("id", "")

                if not page_id:
                    continue

                sentences = parse_sentences(
                    page.get("lines", "")
                )

                for sentence_id, sentence_text in sentences:

                    documents.append({
                        "page_id": page_id,
                        "sentence_id": sentence_id,
                        "text": sentence_text
                    })

    return documents


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        print("\nTokenizing evidence corpus...")

        tokenized_corpus = [
            tokenize(document["text"])
            for document in documents
        ]

        print("Building BM25 index...")

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    def retrieve(self, claim, k=TOP_K):

        query = tokenize(claim)

        scores = self.bm25.get_scores(query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results = []

        for index in ranked_indices:

            document = self.documents[index]

            results.append({
                "page_id": document["page_id"],
                "sentence_id": document["sentence_id"],
                "text": document["text"],
                "score": float(scores[index])
            })

        return results


def main():

    print("\nFEVER BM25 RETRIEVER")
    print("--------------------------------")

    documents = load_wikipedia_corpus()

    print(
        "\nEvidence sentences loaded:",
        len(documents)
    )

    retriever = BM25Retriever(
        documents
    )

    # Real FEVER claim from our dataset.
    claim = (
        "Nikolaj Coster-Waldau worked with "
        "the Fox Broadcasting Company."
    )

    print("\nQUERY")
    print("--------------------------------")
    print(claim)

    results = retriever.retrieve(
        claim,
        k=TOP_K
    )

    print("\nTOP BM25 EVIDENCE")
    print("--------------------------------")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(f"\nRank {rank}")

        print(
            "Page:",
            result["page_id"]
        )

        print(
            "Sentence:",
            result["sentence_id"]
        )

        print(
            "BM25 score:",
            round(result["score"], 4)
        )

        print(
            "Text:",
            result["text"]
        )

    print(
        "\nBM25 retrieval pipeline passed."
    )


if __name__ == "__main__":
    main()