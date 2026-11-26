# Excluded files

Source: tracked files under `rag_finetuning/` at source commit `87839f1`.

| | Files |
|---|---|
| Tracked | 1156 |
| Published | 271 (plus 2 compat modules copied from the platform package) |
| Excluded | **885**, listed below |

What else is not in this release:
- The rest of the source repository (the P1–P5 knowledge-base platform: `kb_platform/`, `ragas_eval/` except
  the two compat modules, `frontend/`, `config/`, `scripts/`, `tests_local/`) is outside the project scope.
- The untracked duplicate `rag_finetuning/experiments/p6_v5/data/train_v5/structured_questions.jsonl` was
  never considered for export.
- Weights, adapters, checkpoints, workspace logs and caches were never in Git and are not exported.

Directories holding unpublished datasets are listed **without file names**, so that the topics of unpublished
sets are not revealed.

| Path (grouped) | Files | Reason |
|---|---|---|
| `rag_finetuning/data/built/…` | 4 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/data/corpus/…` | 22 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/data/qa/…` | 3 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/evaluation/e2e_example_qa.py` | 1 | platform-integration (P1-P5) |
| `rag_finetuning/experiments/p6_v2/blind/…` | 14 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v2/frozen_inputs/…` | 2 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v2/prompts/fewshot_demos.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v2/results/blind_v1/assistant_audit_C_vs_D.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v2/results/blind_v1/error_cases.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/experiments/p6_v2/results/blind_v1/generations_<group>.jsonl` | 5 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v2/results/blind_v1/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v2/results/blind_v1/summary.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/experiments/p6_v2/results/diagnostic_47/error_cases.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/experiments/p6_v2/results/diagnostic_47/generations_<group>.jsonl` | 5 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v2/results/diagnostic_47/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v2/results/diagnostic_47/summary.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/experiments/p6_v3/ablations/seed_replicate/error_cases.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/experiments/p6_v3/ablations/seed_replicate/generations_<group>.jsonl` | 2 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v3/ablations/seed_replicate/generations_C_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v3/ablations/seed_replicate/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v3/ablations/test_v3/error_cases.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/experiments/p6_v3/ablations/test_v3/generations_<group>.jsonl` | 2 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v3/ablations/test_v3/generations_abl_coarse_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v3/ablations/test_v3/generations_abl_minus_new_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v3/ablations/test_v3/generations_abl_no_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v3/ablations/test_v3/generations_abl_refusal_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v3/ablations/test_v3/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v3/data/corpus/…` | 40 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v3/data/drafts/…` | 314 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v3/data/frozen/…` | 3 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v3/data/test_v3/…` | 13 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v3/data/v3/…` | 7 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v3/reports/HUMAN_REVIEW.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v3/results/reg_p6v1_47/error_cases.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/experiments/p6_v3/results/reg_p6v1_47/generations_<group>.jsonl` | 6 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v3/results/reg_p6v1_47/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v3/results/reg_p6v2_63/error_cases.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/experiments/p6_v3/results/reg_p6v2_63/generations_<group>.jsonl` | 6 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v3/results/reg_p6v2_63/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v3/results/test_v3/error_cases.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/experiments/p6_v3/results/test_v3/generations_<group>.jsonl` | 6 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v3/results/test_v3/llm_judge.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v3/results/test_v3/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v3/results/training_runs.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v3/results/v1_audit/audit_v1.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v4/HUMAN_REVIEW_V4.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v4/data/corpus/…` | 132 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v4/data/frozen/…` | 2 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v4/data/test_v4/…` | 14 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v4/data/v4/…` | 3 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v4/results/diagnosis/diagnosis_items.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/experiments/p6_v4/results/diagnosis/diagnosis_labels.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v4/results/metric_audit/metric_audit.json` | 1 | json contains item text (key:claims) |
| `rag_finetuning/experiments/p6_v4/results/metric_audit/per_item_claim_support.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v4/results/test_v4/generations_<group>.jsonl` | 5 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v4/results/test_v4/generations_A_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v4/results/test_v4/generations_B_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v4/results/test_v4/generations_C42_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v4/results/test_v4/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v4/results/test_v4/summary.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v4/results/test_v4/v4_failure_labels.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v4/results/training_runs/v4_3b_s42_attempt2_STALLED.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v5/HUMAN_REVIEW_CHECKLIST.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/HUMAN_REVIEW_DASHBOARD.html` | 1 | review dashboard embeds per-item data/outputs |
| `rag_finetuning/experiments/p6_v5/PLANNING_V2_PER_QUESTION.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/PREFERENCE_PAIR_AUDIT.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/data/frozen/…` | 1 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v5/data/frozen/…` | 1 | test_v5 annotations / per-item results |
| `rag_finetuning/experiments/p6_v5/data/hard_dev/…` | 15 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v5/data/review/…` | 4 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v5/data/test_v5/…` | 27 | test_v5 annotations / per-item results |
| `rag_finetuning/experiments/p6_v5/data/train_pref/…` | 25 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v5/data/train_v5/…` | 4 | dataset / corpus / preference data (not in public scope) |
| `rag_finetuning/experiments/p6_v5/diagnostics/oracle_ablation/per_item_diff.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/diagnostics/oracle_ablation/per_item_per_condition.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/diagnostics/oracle_ablation/scorer_disputes_review.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/diagnostics/oracle_ablation/summary.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v5/diagnostics/scorer_review/disputes_by_category.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/diagnostics/scorer_review/disputes_summary.json` | 1 | json contains item text (key:required_facts) |
| `rag_finetuning/experiments/p6_v5/diagnostics/scorer_review/malformed_403_cases.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/diagnostics/truncation/truncated_pairs.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/diagnostics/truncation/truncation_audit.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v5/final/failure_data.json` | 1 | json contains item text (key:content) |
| `rag_finetuning/experiments/p6_v5/final/per_item_test_v5.csv` | 1 | test_v5 annotations / per-item results |
| `rag_finetuning/experiments/p6_v5/results/error_ledger/error_ledger.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/error_ledger/summary.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_dpo/generations_<group>.jsonl` | 9 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_dpo/generations_A_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_dpo/generations_C42_<group>.jsonl` | 3 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_dpo/generations_D_<group>.jsonl` | 3 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_dpo/per_fact.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_dpo/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_round1/generations_<group>.jsonl` | 5 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_round1/generations_A_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_round1/generations_D_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_round1/per_fact.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_round1/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_s42/generations_<group>.jsonl` | 6 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_s42/generations_A_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_s42/generations_C42_<group>.jsonl` | 3 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_s42/generations_D_<group>.jsonl` | 3 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_s42/per_fact.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_s42/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_s43/generations_<group>.jsonl` | 7 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_s43/generations_A_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_s43/generations_C42_<group>.jsonl` | 3 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_s43/generations_D_<group>.jsonl` | 3 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_s43/per_fact.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_s43/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_scorer402/generations_<group>.jsonl` | 5 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_scorer402/generations_A_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_scorer402/generations_C42_<group>.jsonl` | 3 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_scorer402/generations_D_<group>.jsonl` | 3 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_scorer402/per_fact.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_scorer402/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_scorer403/generations_<group>.jsonl` | 5 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_scorer403/generations_A_<group>.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_scorer403/generations_C42_<group>.jsonl` | 3 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_scorer403/generations_D_<group>.jsonl` | 3 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_scorer403/per_fact.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/hard_dev_scorer403/per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/p42/run/dev_eval/checkpoint-108.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/p42/run/dev_eval/checkpoint-144.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/p42/run/dev_eval/checkpoint-146.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/p42/run/dev_eval/checkpoint-36.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/p42/run/dev_eval/checkpoint-72.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/p42/run/dev_eval/init_sft.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/p42_p43/dual_seed_summary.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v5/results/p42_p43/per_item_dpo.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/p43/run/dev_eval/checkpoint-108.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/p43/run/dev_eval/checkpoint-144.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/p43/run/dev_eval/checkpoint-146.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/p43/run/dev_eval/checkpoint-36.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/p43/run/dev_eval/checkpoint-72.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/p43/run/dev_eval/init_sft.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/planning_v2/generations_C42_<group>.jsonl` | 3 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/planning_v2/generations_D_<group>.jsonl` | 2 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/planning_v2/probe_7b_extract_raw.json` | 1 | json contains item text (key:chunk_head) |
| `rag_finetuning/experiments/p6_v5/results/planning_v2/probe_7b_extract_variant.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v5/results/s42/per_item_changes.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/s42/run/dev_eval/checkpoint_epoch1.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/s42/run/dev_eval/checkpoint_epoch2.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/s42/run/dev_eval/checkpoint_epoch3.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/s42/sensitivity_dv_m13_m16.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v5/results/s42_s43/per_item_two_seeds.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/s43/per_item_changes.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/s43/run/dev_eval/checkpoint_epoch1.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/s43/run/dev_eval/checkpoint_epoch2.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/s43/run/dev_eval/checkpoint_epoch3.jsonl` | 1 | per-item model outputs or data (jsonl) |
| `rag_finetuning/experiments/p6_v5/results/s43/sensitivity_dv_m13_m16.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/experiments/p6_v5/results/scorer_402/changed_items.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/experiments/p6_v5/results/test_v5/generations_<group>.jsonl` | 9 | test_v5 annotations / per-item results |
| `rag_finetuning/experiments/p6_v5/results/test_v5/per_fact.csv` | 1 | test_v5 annotations / per-item results |
| `rag_finetuning/experiments/p6_v5/results/test_v5/per_question.csv` | 1 | test_v5 annotations / per-item results |
| `rag_finetuning/experiments/p6_v5/results/test_v5/summary.json` | 1 | test_v5 annotations / per-item results |
| `rag_finetuning/experiments/p6_v5/scripts/analyze_test_v5.py` | 1 | test_v5 annotations / per-item results |
| `rag_finetuning/experiments/p6_v5/scripts/review_dashboard_template.html` | 1 | review dashboard embeds per-item data/outputs |
| `rag_finetuning/reports/results/changes_base_vs_v1.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/reports/results/changes_base_vs_v2.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/reports/results/dataset_manifest.json` | 1 | json contains item text (key:question) |
| `rag_finetuning/reports/results/e2e_example_qa.json` | 1 | platform-integration (P1-P5) |
| `rag_finetuning/reports/results/llm_judge_test.json` | 1 | json contains item text (long_cjk_str) |
| `rag_finetuning/reports/results/test_per_question.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/reports/test_review_sheet.csv` | 1 | per-item CSV (outputs / annotations / review) |
| `rag_finetuning/tests/local_acl_regression.py` | 1 | platform-integration (P1-P5) |