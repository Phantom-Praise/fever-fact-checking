# Colab starter for FEVER

## Debug notebook update (v2)

Use `FEVER_Colab_Debug.ipynb` for live monitoring. Reuse the original `FEVER_Colab_Bundle.zip`; no second 430 MB download is needed. The notebook verifies that original bundle and applies a checksummed runtime update in Colab's temporary project directory. It uses a new `colab_full_dev_debug_v2` results folder so earlier predictions are preserved.

During evaluation, watch the current claim/stage, progress bar, per-claim prediction, invalid count and approximate ETA. A heartbeat reports elapsed stage time every 20 seconds. The exact model responses and prompts remain in Drive, with a short console preview. If Colab shortens the displayed output, the saved session JSONL log still contains the events.

After a session pauses or fails, run the inspection cell. Set a claim ID to inspect it, enable `SHOW_FULL_TRACE` for the prompt/response/passages, and optionally enable `SHOW_REFERENCE_LABEL` to compare a saved prediction with FEVER. The kernel cannot execute that inspection cell concurrently with a busy evaluation cell; live updates appear in the evaluation cell itself.

Evaluation status/log files are under `RUN_ROOT/monitor/CONDITION/`. Training logs are under the selected training condition's `monitor/` folder. Training also displays optimizer-step progress, loss, learning rate, validation metrics and saved checkpoints. A stale status file after an unexpected runtime shutdown describes the last observed stage, not proof that the job is still running.

Three consecutive preparation/infrastructure failures or ten consecutive invalid outputs stop evaluation for review. Completed failures stay in the results. This is ordinary task monitoring and does not bypass free-Colab limits.

Open `output/colab_20261104/FEVER_Colab_Starter.ipynb` in Google Colab. Put the supplied `FEVER_Colab_Bundle.zip` in Google Drive, select a T4 GPU, and run sections 1–5. These inspect hardware and inputs; they do not start a long evaluation or training.

Then enable model downloads after the size estimate and run section 6's one-claim engineering check. The default model is Qwen2.5-3B-Instruct. Llama-3.2-3B-Instruct needs user-obtained Hugging Face access and a read token in Colab Secrets. The token is never printed or written into a protocol.

## Included

- All **19,998 official FEVER development claims**, with reference labels in a separate file.
- Original title-FTS/BM25 top-ten article rankings exported from the local project. The first five articles are fetched in Colab, chunked, embedded and searched in a temporary FAISS + SQLite database.
- The unchanged existing task-trained RoBERTa checkpoint.
- Actual article passages as inputs, not article titles alone. RoBERTa still truncates to its architectural input budget, which is recorded.
- Qwen and Llama 3B four-bit inference adapters; one GPU classifier is resident at a time.
- Exact prompts, responses, source revisions, token counts, invalid/error records and Drive resume checkpoints.
- Optional RoBERTa continued fine-tuning of the classification head plus the final 1, 2 or 3 blocks.

## Important scope

The bundled **96 training + 96 validation examples are training-software pilots**, not final training sets. They use retrieved historical sentence evidence for all three classes, including nonempty NEI inputs. They are disjoint by normalized claim from each other and official development. They do not replicate the full-article RAG training distribution.

The full evaluation target is never reduced to those pilots. Section 8 refuses final scoring until all 19,998 prediction files exist for a condition. A timed session can stop and resume; invalid model outputs still count as errors.

Llama 3.2 3B/NF4 and Hugging Face greedy decoding differ from the earlier Ollama Llama 3.1 8B/Q4_K_M/schema-constrained experiment. This is a new experiment. Do not pool or overwrite old measurements. The prompt is retained, while the generation backend and declared 512-token output cap change.

The original checkpoint's historical training audit found three normalized official-development text overlaps; they are not silently removed from the full benchmark. This benchmark was previously inspected, so it is not an untouched test set.

Llama and Qwen fine-tuning, Reddit evaluation, a replacement symbolic architecture, and guaranteed unattended free-cloud execution are **not** implemented by this starter. Last-layer unfreezing is implemented for RoBERTa only. No cloud performance or speedup has been measured yet.

## Full RoBERTa training data (separate preparation)

The exporter can also produce full training/validation inputs from the official training pool, with a fixed normalized-claim group split and official-development overlap exclusion:

```sh
.venv/bin/python colab/export_colab_bundle.py --output output/colab_full_training_export --include-training
```

This performs many CPU/SQLite retrieval queries and can take hours. It does **not** run by default and was not executed to create the starter. Run only when you choose that scope. Use a new output folder; completed bundles are preserved.

The training data use the same historical retriever for all labels and retain original human labels even if retrieval fails. Retrieval failures and the corpus/input difference need reporting. These are not secretly substituted live-RAG contexts.

After uploading the full bundle, select `TRAIN_SCOPE="full"`. Each layer condition starts from the same checkpoint, uses seed 42 and fixed hyperparameters, and saves optimizer checkpoints to Drive. Model selection uses training-pool validation, never official-development labels.

## Free Colab sessions

Free Colab does not guarantee a particular GPU, session length, or execution after browser disconnection. The session budget and Drive checkpoints protect progress; they do not bypass service limits. Changed protocols or GPU/software identities reject resume rather than mix incompatible results. Reuse the same environment or choose a new run name.

Public documentation:
- https://research.google.com/colaboratory/faq.html
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer
- https://huggingface.co/docs/transformers/v4.50.0/en/quantization/bitsandbytes

## Local verification

```sh
.venv/bin/python -m unittest discover -s tests -p test_colab_runtime.py -v
```

Checks cover response truncation/schema/citations, label aggregation, immutable outputs, protocol mismatch rejection, actual last-block trainability on a tiny random RoBERTa, and partial scoring. They make no network requests and perform no training. The notebook's code cells are syntax checked. Actual Colab installation, GPU inference, Google Drive behavior and runtime speed remain to be tested in your account.
