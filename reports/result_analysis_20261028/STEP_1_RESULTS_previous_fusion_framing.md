# Result Analysis — Step 1: Tables and figures

## Table 1. Controlled local comparison

The combined RoBERTa system is the main model for the full-benchmark comparison. All three systems were evaluated on the same 19,998 official FEVER development claims.

| System | Role | Accuracy | Macro-F1 | Strict FEVER |
|---|---|---:|---:|---:|
| Claim-only RoBERTa H2 | Baseline: claim only | 58.78% | 0.5775 | 12.23% |
| Evidence RoBERTa | Component: claim + retrieved evidence | 52.37% | 0.4175 | 35.07% |
| **Combined RoBERTa** | **Main model: frozen score fusion** | **61.70%** | **0.6078** | **45.33%** |

Accuracy measures correct SUPPORTS / REFUTES / NOT ENOUGH INFO labels. Macro-F1 gives each class equal weight. Strict FEVER additionally requires a complete annotated evidence group among the first five submitted sentences for SUPPORTS/REFUTES. Correct NEI answers do not require evidence.

The claim-only model submits no evidence; its strict FEVER score comes only from correct NEI answers. This is an input ablation, so the strict-score difference includes the ability to submit evidence. All scores are development results, not blind-test results.

## Table 2. Published development-set context

| System | Accuracy | Strict FEVER | Source |
|---|---:|---:|---|
| Our combined RoBERTa | 61.70% | 45.33% | Measured locally |
| GEAR | 74.84% | 70.69% | Published development reference; not rerun |
| KGAT (RoBERTa-large) | 78.29% | 76.11% | Published development reference; not rerun |

Published values come from the **Dev LA and FEVER columns of [KGAT Table 2](https://aclanthology.org/2020.acl-main.655.pdf#page=6)**. They use the same benchmark split and metric definitions, making them useful benchmark references. Retrieval, architectures, model sizes and training resources differ; these are not equal-resource reproductions or isolated architecture comparisons. Published classification macro-F1 is not supplied in this table; retrieval F1 must not substitute for it.

## Table 3. Separate local LLM assessment

These 120 official-development claims are separate from the 60 calibration claims and earlier 300-claim pilot. Keep these scores separate from the 19,998-claim comparison.

| System | Accuracy | Macro-F1 | Strict FEVER |
|---|---:|---:|---:|
| Original Llama pipeline | 51.67% | 0.4973 | 43.33% |
| New prompt + retrieval, before fallback | 57.50% | 0.5697 | 47.50% |
| Selected hybrid pipeline | 59.17% | 0.5821 | 47.50% |

The hybrid adds a fixed claim-only fallback to the changed LLM pipeline. The observed accuracy difference from the original is +7.50 percentage points; the paired 95% interval is −2.50 to +18.33 points, so the improvement remains uncertain.

## Figures

[Open the interactive results figures](http://127.0.0.1:8765/result-analysis/).

- Figure 1: full-benchmark bars comparing the three local models.
- Figure 2: separate 120-claim LLM assessment bars.
- Figure 3: literature-context bars under **Scope & sources**, with accuracy and strict FEVER views.

Chart axes begin at zero. Measured and published results are identified separately. The full-benchmark comparison uses the original 256-token evidence-input condition. The separate 512-token study added only two correct final labels and is documented in the full analysis.

## Measurement sources

- `reports/part2/official_fever_dev_20261023/summary.json`
- `reports/part2/llm_configurations_20261028/assessment_summary.json`
- `reports/part2/llm_configurations_20261028/assessment_component_diagnostic.json`
- `reports/result_analysis_20261028/literature_sources.json`

This document organizes Step 1 only. Class-level analysis, qualitative examples, hypothesis conclusions and limitations belong to the later assignment steps. No experiments were rerun for this document.
