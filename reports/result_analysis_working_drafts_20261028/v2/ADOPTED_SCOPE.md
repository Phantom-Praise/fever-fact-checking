# Adopted narrower research scope

Status: **Adopted on October 28, 2026 for this analysis.** This supersedes the October 23 discussion draft for the current analysis; it does not rewrite the original proposal or earlier reports.

## Title

Evidence Inputs and Decision Rules in a Local FEVER Fact-Checker

## Main research question

How do evidence inputs and decision rules affect label accuracy and evidence-grounded verification in a local FEVER fact-checker?

## Scope and timing of the revision

The study is restricted to three-label FEVER verification with the existing Wikipedia snapshot, frozen RoBERTa models, and the measured local Llama configurations. It is an empirical evaluation of evidence inputs and decision rules, not a new graph architecture or a general evaluation of every LLM.

The revised questions and working hypotheses below were written after inspecting results. They are retrospective research framing, not original or preregistered predictions. Earlier protocols remain as recorded; in particular, the latest five-configuration selection was frozen before its disjoint assessment. That within-experiment ordering does not make this later research reframing prospective.

## What happens to the original questions?

| Earlier scope | Decision in this adopted scope | What may be claimed |
|---|---|---|
| Original RQ1: whether FEVER represents LLM-generated factual-error types | Move to future work, explicitly unanswered | Generated-claim preparation and automated pilots exist; independent human validation is incomplete. Renaming the question does not complete that work. |
| Original local-judge question | Narrow to the measured local model, evidence and configuration conditions | Discuss model-specific classification/error patterns; do not generalize to every LLM or claim a completed human study. |
| Earlier neural/symbolic comparison and efficiency questions | Analyze frozen fusion and the implemented symbolic route as separate comparisons | Report measured gains and the zero-coverage negative result. No claim that symbolic methods generally fail. |
| Earlier corpus-size, generated-claim and graph-system ambitions | Preserve their records; exclude broad unsupported claims from the central contribution | Do not claim graph-model reproduction, a full blind benchmark, corpus-scale generality, or verified generated-claim truth labels. |

The new numbering always uses **Revised RQ1–RQ4**. It must not be read as a claim that all original questions were completed.

| Question | Narrowed question | Retrospective working hypothesis |
| --- | --- | --- |
| Revised RQ1 | How do evidence relevance and input length affect verification? | H1a: claim-related evidence outperforms deliberately mismatched evidence. H1b: doubling the evidence-model input limit produces a reliable accuracy gain. |
| Revised RQ2 | Does frozen RoBERTa fusion outperform its individual components on the official development set? | H2: validation-selected fusion improves label accuracy and macro-F1 over each individual model. |
| Revised RQ3 | Can a selected local LLM pipeline configuration improve over its original setup on disjoint assessment claims? | H3: the calibration-selected configuration improves accuracy and macro-F1 on the separate assessment sample. |
| Revised RQ4 | Does the implemented symbolic-first route avoid model calls and reduce latency? | H4: symbolic routing avoids expensive fallback calls and reduces mean inference-pipeline time without lowering label accuracy. |
