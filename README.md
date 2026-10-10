# Retrieval-Augmented Fact-Checking with Transformer Models

## COSC 874 — Deep Learning II

### Project Overview

This project implements a fact-checking pipeline for verifying factual claims using the FEVER dataset. The planned system combines symbolic knowledge-base verification, information retrieval, and Transformer-based classification.

The final architecture is:

Claim → Symbolic Knowledge Base → BM25 Retrieval → RoBERTa → Classification

The classification labels are:

- SUPPORTS
- REFUTES
- NOT ENOUGH INFO

## Dataset

The project uses the FEVER fact-verification dataset.

The downloaded training dataset contains 145,449 examples:

- SUPPORTS: 80,035 (55.03%)
- REFUTES: 29,775 (20.47%)
- NOT ENOUGH INFO: 35,639 (24.50%)

Each example contains a claim, label, verifiability information, and evidence annotations.

## Preprocessing

The preprocessing pipeline is implemented in:

`src/preprocess.py`

It performs:

1. Dataset loading and JSON validation
2. Required-field validation
3. Label encoding
4. Class-distribution analysis
5. Group-aware train/validation/test splitting
6. Claim-leakage verification
7. Processed dataset generation

Because repeated claim text was discovered during exploratory analysis, identical claims are kept in the same partition to reduce data leakage.

The resulting split is:

| Split | Examples | Percentage |
|---|---:|---:|
| Training | 116,356 | 80% |
| Validation | 14,545 | 10% |
| Test | 14,548 | 10% |

Claim-overlap verification produced zero overlap between all three partitions.

## Transformer Baseline

The baseline model uses `FacebookAI/roberta-base`.

The implementation is located in:

`src/baseline.py`

The baseline currently verifies that:

- the RoBERTa tokenizer loads successfully,
- FEVER claims can be tokenized,
- RoBERTa can process the tokenized input, and
- the classification head produces three output logits corresponding to the three FEVER classes.

## Training Configuration

Training initialization is implemented in:

`src/train.py`

Current configuration:

- Model: RoBERTa-base
- Number of classes: 3
- Maximum sequence length: 256
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 3
- Device support: Apple MPS, CUDA, or CPU

The current development machine successfully detects Apple MPS acceleration.

## Project Structure

fever-fact-checking/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── test_data.py
│   ├── preprocess.py
│   ├── baseline.py
│   └── train.py
├── results/
├── checkpoints/
├── requirements.txt
└── README.md

## Next Steps

The next implementation stages are:

1. Complete RoBERTa fine-tuning on the FEVER training split.
2. Evaluate the classifier using accuracy, precision, recall, and macro-F1.
3. Implement BM25 evidence retrieval.
4. Measure evidence retrieval using Recall@1, Recall@5, and Recall@10.
5. Integrate Wikidata as the symbolic knowledge-base stage.
6. Compare the Transformer-only, retrieval-augmented, LLM-as-judge, and hybrid approaches.
7. Measure latency and the percentage of claims resolved by the symbolic stage.

## Reproducibility

Install the required Python packages with:

pip install -r requirements.txt

Run preprocessing with:

python src/preprocess.py

Test the RoBERTa baseline with:

python src/baseline.py

Test training initialization with:

python src/train.py