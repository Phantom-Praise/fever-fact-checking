import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from retriever import load_wikipedia_corpus, BM25Retriever


MODEL_NAME = "FacebookAI/roberta-base"
CHECKPOINT = "checkpoints/roberta_epoch_1"

TOP_K = 5
MAX_LENGTH = 256


def build_evidence_text(results):
    """
    Combine retrieved Wikipedia sentences
    into one evidence passage.
    """

    evidence_sentences = [
        result["text"]
        for result in results
    ]

    return " ".join(evidence_sentences)


def main():

    print("\nRETRIEVAL-AUGMENTED FACT-CHECKING MODEL")
    print("----------------------------------------")

    # ----------------------------------------
    # Device
    # ----------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    # ----------------------------------------
    # Build retrieval system
    # ----------------------------------------

    print("\nLoading Wikipedia evidence...")

    documents = load_wikipedia_corpus()

    print(
        "Evidence sentences:",
        len(documents)
    )

    retriever = BM25Retriever(
        documents
    )

    # ----------------------------------------
    # Load our trained RoBERTa checkpoint
    # ----------------------------------------

    print("\nLoading trained RoBERTa checkpoint...")

    tokenizer = (
        AutoTokenizer
        .from_pretrained(CHECKPOINT)
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(CHECKPOINT)
    )

    model.to(device)
    model.eval()

    # ----------------------------------------
    # Example FEVER claim
    # ----------------------------------------

    claim = (
        "Nikolaj Coster-Waldau worked with "
        "the Fox Broadcasting Company."
    )

    print("\nCLAIM")
    print("--------------------------------")
    print(claim)

    # ----------------------------------------
    # Retrieve evidence
    # ----------------------------------------

    retrieved = retriever.retrieve(
        claim,
        k=TOP_K
    )

    evidence = build_evidence_text(
        retrieved
    )

    print("\nRETRIEVED EVIDENCE")
    print("--------------------------------")

    for rank, result in enumerate(
        retrieved,
        start=1
    ):
        print(
            f"{rank}. "
            f"{result['page_id']} "
            f"[sentence {result['sentence_id']}]"
        )

        print(
            f"   {result['text']}"
        )

    # ----------------------------------------
    # Claim + evidence
    # ----------------------------------------

    inputs = tokenizer(
        claim,
        evidence,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # ----------------------------------------
    # Classification
    # ----------------------------------------

    with torch.no_grad():

        outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    id_to_label = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO"
    }

    predicted_label = (
        id_to_label[prediction]
    )

    print("\nMODEL INPUT")
    print("--------------------------------")

    print(
        "Tokenized input shape:",
        inputs["input_ids"].shape
    )

    print("\nMODEL OUTPUT")
    print("--------------------------------")

    print(
        "Logits:",
        outputs.logits.cpu()
    )

    print(
        "Predicted label:",
        predicted_label
    )

    print(
        "\nRetrieval-augmented "
        "pipeline passed."
    )


if __name__ == "__main__":
    main()