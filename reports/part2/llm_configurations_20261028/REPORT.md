# Controlled local LLM configuration search

Five predeclared configurations. Select on 60 calibration claims, then assess baseline and the frozen selection on 120 disjoint claims. No retraining.

| Calibration configuration | Accuracy | Macro-F1 | Strict FEVER | Evidence recall |
|---|---:|---:|---:|---:|
| Original prompt + original retrieval | 46.67% | 0.4627 | 40.00% | 55.00% |
| Comparison prompt + original retrieval | 53.33% | 0.5258 | 43.33% | 55.00% |
| Original prompt + entity retrieval | 55.00% | 0.5528 | 48.33% | 65.00% |
| Comparison prompt + entity retrieval | 60.00% | 0.5902 | 51.67% | 65.00% |
| Both changes + claim-only NEI fallback [SELECTED] | 63.33% | 0.6182 | 53.33% | 65.00% |

## Disjoint assessment

| Configuration | Accuracy | Macro-F1 | Strict FEVER |
|---|---:|---:|---:|
| Original prompt + original retrieval | 51.67% | 0.4973 | 43.33% |
| Both changes + claim-only NEI fallback | 59.17% | 0.5821 | 47.50% |

Selected minus original: +7.50 percentage points; 25 errors fixed, 16 introduced. Paired 95% interval: -2.50 to 18.33 points.

- Exploratory 60-calibration/120-assessment study; small samples and one generation per claim.
- Fresh LLM assessment subset is disjoint from the earlier LLM pilot, but official development has previously been evaluated with RoBERTa. It is not a blind official test.
- Entity retrieval changes candidate inclusion and sentence ranking together; it does not isolate only one retrieval subcomponent.
- Prompt variant adds explicit comparison rules and fictional examples together; no individual clause is isolated.
- Hybrid E can improve labels without improving annotated evidence; report strict FEVER alongside label accuracy.
- All response failures remain INVALID in the full denominator. Model-generated explanations are not internal reasoning traces.
- No model retraining, default changes, new downloads or paid API requests. Existing results are preserved.

Paired macro-F1 difference 95% interval: -0.0107 to +0.1809.
Paired strict-FEVER difference: +4.17 percentage points; 95% interval -5.83 to +14.17 points.
Selected hybrid component audit: raw selected LLM accuracy 57.50%; fallback changed 4 labels, fixing 3 errors and introducing 1. This does not change the frozen selection.
