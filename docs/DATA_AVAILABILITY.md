# Data availability

This repository publishes **code, configurations, tests, experimental protocols, aggregate statistics and redacted
reports**. It does **not** publish the experiment data, model outputs or model weights. The public code is
reproducible on the public example; the reported experiments are **not** fully reproducible from this repository
alone.

## What is published

| Item | Where |
|---|---|
| Training / data-construction / evaluation / statistics code (P6-v1 … v5) | `rag_finetuning/` |
| Training configs (paths replaced by `workspace/…`) and pre-registration records (`freeze.json`, amendments) | `rag_finetuning/experiments/*/configs/`, `rag_finetuning/configs/` |
| Per-item **hashes** of the frozen evaluation sets (no text) | `rag_finetuning/experiments/*/data/frozen/*.manifest.json`, `…/data/*/manifest.json` |
| Aggregate metrics, paired-bootstrap results, gate results, training / memory logs | `results/**/summary.json`, `final/STATISTICAL_COMPARISON.json`, `final/FINAL_METRICS_TABLE.csv`, `run_record.json`, `train_log*` |
| Reports, including all negative results, deviations and failure analyses | `rag_finetuning/**/*.md` |
| Hand-written public example (fictional documents, hand-written answers) | `examples/minimal/` |

## What is not published, and why

| Item | Reason |
|---|---|
| test_v5 questions, reference answers, required facts, gold evidence, corpus (`data/test_v5/`, `data/frozen/test_v5.jsonl`) | Final held-out test set; kept private so that it can still be used as an uncontaminated benchmark. Only its manifest (per-item sha256) is public. |
| Per-item test_v5 results (`results/test_v5/`, `final/per_item_test_v5.csv`, `final/failure_data.json`) | They contain the test_v5 annotations and model answers. |
| All other datasets (v1–v4 corpora and splits, hard_dev, v5 SFT data, drafts, review sheets) | Outside the authorised public scope of this release. |
| Preference pairs and candidate answers (`data/train_pref/`) | Complete preference data is explicitly excluded; much of it is Qwen2.5-3B output (Qwen Research License). |
| Per-item model outputs (`generations_*.jsonl`, `per_question.csv`, `per_item*.csv`, `dev_eval/*.jsonl`, review dashboards) | Model outputs whose redistribution was not cleared for this release; several also embed dataset text. |
| Model weights, LoRA adapters, checkpoints | Never committed. Adapter sha256 values are recorded in `configs/test_v5_final_freeze.json`. |

The full list (grouped by directory, without the file names of unpublished sets) is in
[`EXCLUDED_FILES.md`](EXCLUDED_FILES.md).

## Reproducing with private data

If you have the private data, place it back at the original relative paths (for example
`rag_finetuning/experiments/p6_v5/data/frozen/hard_dev.jsonl`). The 17 data-dependent tests then run instead of
being skipped, and the experiment scripts work unchanged. Set the workspace for weights, adapters and generations
by running from the repository root (scripts use `workspace/…`), or symlink `workspace/` to your own location.
`run_eval_v5` also honours `P6_RUNS`.
