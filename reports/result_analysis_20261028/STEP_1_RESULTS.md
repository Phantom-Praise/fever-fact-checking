# Result Analysis — Step 1: Evidence RoBERTa baseline and LLM experiments

## Comparison framing

Evidence RoBERTa is the baseline because it performs the evidence-based classification task: it receives the claim and retrieved Wikipedia text. The Llama pipeline is the focus of the main experimental comparison. Both approaches use evidence, but RoBERTa produces classification scores while Llama receives an instruction and generates a structured label and brief explanation.

This framing was revised after seeing the results. Claim-only and combined RoBERTa remain documented supporting experiments. They are not removed because of the revised focus: combined RoBERTa still achieved higher label accuracy on these same 120 claims (60.83%).

## Table 1. Matched evidence-based comparison

Same 120 official FEVER development assessment claims, 40 per class; disjoint from the 60 calibration claims and earlier 300-claim pilot.

| System | Role | Accuracy | Macro-F1 | Strict FEVER |
|---|---|---:|---:|---:|
| Evidence RoBERTa, 512-token input | **Baseline** | 54.17% | 0.4334 | 32.50% |
| Original Llama pipeline | Original LLM condition | 51.67% | 0.4973 | 43.33% |
| Llama with new prompt + retrieval | LLM experimental variant, before fallback | 57.50% | 0.5697 | 47.50% |
| Llama with new prompt + retrieval + claim-only fallback | Hybrid variant | 59.17% | 0.5821 | 47.50% |

The original Llama condition and evidence RoBERTa use the same original five evidence sentences. RoBERTa uses a 512-token input limit; the LLM has its own prompt and context budget. This is a system comparison, not an isolated architecture experiment. The changed Llama conditions use different retrieval and prompting, so their gain cannot be attributed to the LLM alone. The 59.17% result includes a claim-only fallback and must not be described as a pure LLM result.

Accuracy measures correct labels. Macro-F1 gives each of the three classes equal weight. Strict FEVER also requires a complete annotated evidence group among the first five submitted sentences for SUPPORTS/REFUTES; correct NEI labels do not require evidence.

The changed pure LLM pipeline is 3.33 percentage points above the evidence baseline in observed accuracy. The hybrid is 5.00 points above it. These are measured sample differences; no significance claim is made for these new comparisons. The previously computed hybrid-versus-original-LLM accuracy interval is −2.50 to +18.33 points and does not apply to hybrid-versus-RoBERTa.

## Table 2. Full-benchmark literature context

| System | Development accuracy | Strict FEVER | Source |
|---|---:|---:|---|
| Our evidence RoBERTa, original 256-token condition | 52.37% | 35.07% | Measured on all 19,998 development claims |
| GEAR | 74.84% | 70.69% | Published development reference; not rerun |
| KGAT (RoBERTa-large) | 78.29% | 76.11% | Published development reference; not rerun |

Published values come from the Dev LA and FEVER columns of [KGAT Table 2](https://aclanthology.org/2020.acl-main.655.pdf#page=6). Matching benchmark split and metric makes these useful context. Architecture, retrieval, model size and training resources differ. GEAR and KGAT use evidence graphs; this project did not reproduce those graph architectures. Published classification macro-F1 is not supplied here; retrieval F1 must not substitute for it.

The 120-claim LLM scores are excluded from this full-benchmark table because Llama was not evaluated on all 19,998 claims. The baseline's 54.17% on the smaller sample and 52.37% on the full benchmark describe different populations and input-limit conditions.

## Figures and sources

The [existing interactive dashboard](http://127.0.0.1:8765/result-analysis/) preserves the earlier fusion-centered overview and its supporting figures. The baseline designation in this revised Step 1 document is evidence RoBERTa; the dashboard overview has not yet been reframed around it.

Measurement sources:

- `reports/part2/llm_configurations_20261028/assessment_summary.json`
- `reports/part2/llm_configurations_20261028/assessment_component_diagnostic.json`
- `reports/part2/official_fever_dev_20261023/summary.json`
- `derived_assessment_evidence_baseline.json` (strict FEVER derived from saved 512-token outputs and gold annotations; 39/120)

No retraining or new model calls were performed. Earlier results and the previous Step 1 framing are preserved. This document organizes Step 1; later assignment steps cover detailed errors, hypotheses and limitations.
