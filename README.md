# llm-evidence-alignment

**Evidence-Aware LLM Post-Training & Preference Optimization.** This is research code for teaching a small
instruction model (Qwen2.5-3B, 4-bit QLoRA on a single 8 GB laptop GPU) to answer from retrieved evidence. The
goals are to cite the right chunks, cover every required fact spread over several chunks, refuse when the
evidence is missing, flag partial answers and conflicts, and resist prompt injection.

> **Status: research engineering release v1.0.0. No model is promoted.**
> - In the final, pre-registered, held-out evaluation (test_v5, 125 items), **neither the SFT nor the DPO
>   candidate passed all seven promotion gates on both seeds**. No claim of a successful production model
>   replacement is made.
> - All metrics come from a rule-based scorer and AI-written annotations. **No result has been independently
>   human-reviewed; every result is provisional.**
>
> 中文说明见 [README_CN.md](README_CN.md).

## Research Objective

Can preference optimization on top of evidence-first SFT make a 3B model integrate **multiple pieces of
evidence** (the main failure of earlier versions) without regressing refusal, citation or safety behaviour, under
a fixed 8 GB GPU budget?

All claims are judged against gates fixed before the final test set was run.

## System Architecture

```
            ┌───────────────────── data engineering (P6-v3…v5) ─────────────────────┐
 fictional  │ evidence-first construction → required facts + supporting chunks       │
 policy  ──►│ leakage checks (doc-disjoint splits, n-gram / hash checks vs dev/test) │
 corpora    │ preference pairs: real errors + constructed errors, QC & exclusions    │
            └──────────────┬───────────────────────────────┬────────────────────────┘
                           ▼                               ▼
                 QLoRA SFT (S42 / S43)  ──init/ref──►  DPO (P42 / P43)
                 Qwen2.5-3B NF4, LoRA r=8            use_logits_to_keep, 8 GB
                           │                               │
                           └──── checkpoint selection on hard_dev (pre-registered rule) ───┐
                                                                                          ▼
   frozen platform prompt (compat/rag_pipeline.py) ─► generation (greedy, 384 tok) ─► rule scorer 4.0.3
                                                                                          │
                     source-document clustered paired bootstrap ◄────────────────────────┘
                     + 7 pre-registered promotion gates (vs v3)
```

| Module | Path |
|---|---|
| QLoRA / SFT / DPO training | `rag_finetuning/scripts/train_qlora.py`, `experiments/p6_v3/scripts/train_v3.py`, `p6_v4/scripts/train_v4.py`, `p6_v5/scripts/train_sft_v5.py`, `p6_v5/scripts/train_dpo_v5.py` |
| Evidence-first data construction | `p6_v5/scripts/build_pref_ef.py`, `p6_v3/scripts/build_v3.py`, `p6_v4/scripts/build_v4.py` |
| Preference-pair construction and QC | `p6_v5/scripts/build_pref_pairs.py`, `build_final_pref.py`, `check_pref_leakage.py`, `review_assist.py` |
| Data isolation and leakage detection | `rag_finetuning/scripts/build_dataset.py`, `p6_v5/scripts/check_pref_leakage.py`, `tests/test_p6v5_planning_leak.py` |
| Oracle-evidence ablation | `p6_v5/diagnostics/oracle_ablation.py` |
| Planning inference experiments (negative) | `p6_v5/scripts/evidence_planning.py`, `planning_v2.py` |
| Rule scorers and regression tests | `evaluation/scoring.py` (v1), `p6_v3/scripts/scoring_v2.py`, `p6_v4/scripts/scoring_v3.py`, `p6_v5/scripts/scoring_v4.py` (4.0.3), `tests/test_p6v5_scorer_regression.py` |
| Dual-seed experiment management, checkpoint selection | `p6_v5/scripts/train_*_v5.py`, `analyze_s42.py`, `analyze_dpo.py`, `summarize_s42_s43.py` |
| Clustered paired bootstrap and gates | `p6_v3/scripts/scoring_v2.py::paired_cluster`, `p6_v5/scripts/analyze_test_v5.py` |
| GPU-memory optimisation and numerical-equivalence checks | `p6_v5/diagnostics/dpo_*` |

Paths in the table are under `rag_finetuning/experiments/` unless they start with `rag_finetuning/`.

## Experimental Methodology

- **Pre-registration:** protocols, gates, scorer version and data hashes were frozen before each test run. See
  `experiments/*/configs/freeze.json`, `p6_v5/EVALUATION_PROTOCOL.md` and `p6_v5/configs/test_v5_final_freeze.json`.
  Amendments are recorded and never silently applied.
- **Two seeds per configuration** (42 / 43). Each seed is reported separately. Seed means are descriptive only,
  because both seeds are scored on the same items.
- **Checkpoint selection** used only the development set hard_dev, under a pre-registered rule. test_v5 was run
  once, after all training ended.
- **Statistics:** paired bootstrap clustered by source document (23 clusters on test_v5), 95% CI. Item-level
  McNemar tests are reported for reference only.
- **Evaluation groups on test_v5:**

  | Group | Model |
  |---|---|
  | A | 3B base |
  | B | v3 SFT |
  | C42 / C43 | v4 SFT |
  | S42 / S43 | v5 evidence-first SFT |
  | P42 / P43 | v5 DPO |
  | D | Qwen2.5-7B Q4_K_M via Ollama, reference |

  All groups used the same evidence, the same prompt (the frozen platform prompt) and the same scorer.

## Evidence-first Data Engineering

- **Construction:** questions are built *from* selected evidence chunks and their required facts, instead of
  asking a model to draft answers freely.
  - 161 of 212 evidence-first natural samples passed every automatic check.
  - An earlier free-drafting route with a 7B model passed only 14 of 259.
- **SFT data:**
  - Initial file: 476 examples (315 v3 examples + 161 evidence-first examples).
  - **Formal training data: 472 examples.** Amendment 13 removed 4 over-length samples.
- **Leakage controls:** document-disjoint splits, n-gram and hash checks against hard_dev and test_v5, and
  manifests with per-item sha256.

## QLoRA / SFT / DPO

- **Model and quantization:** Qwen2.5-3B-Instruct at revision `aa8e7253…`, 4-bit NF4 with double quantization,
  bf16 compute.
- **SFT:** 3 epochs; epoch 3 selected on hard_dev for both seeds.
- **DPO:**
  - Initial policy and reference model: the same-seed SFT adapter.
  - Data: **580 final preference pairs**, filtered from 791 candidates. Every exclusion is logged.
  - Training: 2 epochs (146 steps); checkpoint-108 selected for both seeds.
  - Wall time: about 1.8–1.9 h per seed on the laptop GPU.

## Final Evaluation Results (test_v5, 125 items, scorer 4.0.3, provisional)

| Metric | A | B (v3) | C42 | C43 | S42 | S43 | P42 | P43 | D (7B)\* |
|---|---|---|---|---|---|---|---|---|---|
| Multi-evidence complete (/80, primary) | 38 | 35 | 50 | 45 | 44 | 42 | 55 | **57** | 54 |
| Content correct (/125) | 57 | 65 | 83 | 78 | 77 | 77 | 82 | **92** | 83 |
| Whole refusal on answerable (/113, lower is better) | 15 | 11 | 4 | 3 | 9 | 6 | 3 | **0** | 1 |
| Correct refusal (/12) | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 11 | 12 |
| Partial answer correct (/8) | 2 | 3 | 5 | 5 | 6 | 7 | 3 | 5 | **8** |
| True conflict detected (/8) | 0 | 0 | 2 | 2 | **6** | 3 | 2 | 5 | 2 |
| Injection resisted (/6) | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| Latency p50, s | 4.5 | 7.9 | 10.3 | 10.0 | 9.7 | 9.5 | 11.5 | 12.7 | 4.1\* |

\* D runs on a different inference stack (Ollama / llama.cpp, GGUF Q4_K_M). The 3B groups run on transformers +
bitsandbytes NF4 with an unmerged LoRA. D's latency is therefore **not comparable** with the 3B groups.

**Headline result: DPO vs. same-seed SFT (multi-evidence completeness)**

| Comparison | Change | 95% clustered-bootstrap CI |
|---|---|---|
| P42 vs S42 | 44 → 55 of 80, **+13.8 pp** | [+3.6, +24.4] |
| P43 vs S43 | 42 → 57 of 80, **+18.8 pp** | [+8.2, +29.4] |

Both seeds also show less whole refusal and higher citation accuracy (CIs exclude 0).

**Promotion gates** (7 gates, all must hold on **both** seeds; compared with B = v3; gate 2 is compared with C42):

| Candidate | Gates failed | Verdict |
|---|---|---|
| S42 | 2 (content vs v4) | not promoted |
| S43 | 1, 2 | not promoted |
| P42 | 6c (out-of-evidence numbers: 7 vs limit 6), 7 (latency 1.47 × v3) | not promoted |
| P43 | 7 (latency 1.61 × v3) | not promoted |

**Quality vs. promotion.** On the quality metrics, both DPO seeds are significantly better than v3:
- multi-evidence completeness: +25.0 / +27.5 pp;
- content correctness: +13.6 / +21.6 pp.

That is **not** the same as passing promotion. The combined gate set, which includes latency and numeric-safety
gates, was failed on both seeds.

Full reports:
- [FINAL_EVALUATION_REPORT](rag_finetuning/experiments/p6_v5/final/FINAL_EVALUATION_REPORT.md)
- [PROMOTION_GATE_REPORT](rag_finetuning/experiments/p6_v5/final/PROMOTION_GATE_REPORT.md)
- [FINAL_FAILURE_ANALYSIS](rag_finetuning/experiments/p6_v5/final/FINAL_FAILURE_ANALYSIS.md)
- [P6_RESEARCH_SUMMARY](rag_finetuning/experiments/p6_v5/final/P6_RESEARCH_SUMMARY.md)
- aggregate numbers: [FINAL_METRICS_TABLE.csv](rag_finetuning/experiments/p6_v5/final/FINAL_METRICS_TABLE.csv),
  [STATISTICAL_COMPARISON.json](rag_finetuning/experiments/p6_v5/final/STATISTICAL_COMPARISON.json)

## Statistical Analysis

| Comparison | Result |
|---|---|
| DPO vs v4 (C42 / C43) | Undetermined. vs C42 the CIs include 0 (multi-evidence +6.2 / +8.7 pp); vs C43 both seeds are significantly better. |
| 3B DPO vs 7B Q4 (content, multi-evidence) | **No statistically significant difference was observed in this evaluation** (P43 − D content +7.2 pp, CI [−3.4, +17.8]). This does **not** establish equivalence or non-inferiority: no margin was pre-specified, n = 125, and the scoring is not human-reviewed. |
| v5 SFT vs v4 | Did not exceed v4 on the primary metric. |
| Seed agreement (P42 vs P43) | 66/80 items agree on multi-evidence, 101/125 on content. |
| Small subsets (cross-document, 3 or more chunks; n = 17 each) | Too small to judge. |

## GPU Memory Optimization

- **Problem:** DPO at max length 1024 ran out of memory on 8 GB. It also failed at 768, and at 1024 with
  precomputed reference log-probs.
- **Cause:** materialising full-vocabulary logits (about 152k) over the prompt.
- **Fix (amendment 3):** `use_logits_to_keep=True`, which computes logits only for the completion tokens.
- **Equivalence check:**
  - log-prob difference ≤ 1.5e-5;
  - gradient cosine similarity ≥ 0.99991 (4 runs per pair).
- **Worst-case check:** the longest sequences and completions were tested at `P6_MEM_FRACTION=0.74`, which
  works; 0.80 runs out of memory.
- **Formal DPO peak reserved memory:** 5834 / 5798 MiB.
- **3B inference:** 2160 MiB allocated / 3116 MiB reserved.
- **Reports:** `p6_v5/DPO_MEMORY_SMOKE_REPORT.md`, `DPO_LOGITS_EQUIVALENCE_REPORT.md`,
  `DPO_WORST_CASE_MEMORY_REPORT.md`, `DPO_FEASIBILITY_RETEST_REPORT.md`.

## Negative Results (kept on purpose)

- **P6-v1 (1.5B QLoRA):** format improved but content did not (28 → 25 / 47), and the model over-refused.
- **P6-v2:** few-shot prompting gave no reliable content gain for 7B.
- **P6-v3:** over-refusal was traced to a refusal string repeated 28 times in the training targets. 3B v3
  regressed on multi-evidence (6 → 3 / 19); conflict detection was 0/4 for every model.
- **P6-v4:** content beat v3, but multi-evidence was not fixed; the gate failed on both seeds and v4 was not
  recommended.
- **Planning (extract a fact list, then generate):** fell below plain RAG (17 / 20 vs 26 / 43). Oracle fact lists
  reached 41 / 43, which locates the bottleneck in fact identification.
- **DPO:**
  - partial-answer handling regressed (P42: 3 / 8);
  - the effect on conflict detection reversed between seeds;
  - one new refusal error (P43);
  - longer answers with more verbatim copying.
- **Prompt injection:** no method improved it. 4 of the 6 injection items fail in all 9 groups, including 7B.

## Limitations

- **Scorer:** rule-based string matching. It rewards verbatim copying, can count a correct arithmetic or date
  derivation as an "out-of-evidence number", and depends on fixed gap phrases for partial answers.
- **Annotations:** AI-written. Human review was not completed:

  | Material | Reviewed |
  |---|---|
  | preference pairs | 16 / 167 |
  | hard_dev | 15 / 26 |
  | test_v5 annotations | 0 / 125 |

- **Sample size:** test_v5 is small (125 items, 23 source documents). Several subsets have only 6–12 items, so a
  single item can flip a gate.
- **Hardware and inference:** latency was measured on one laptop GPU. The 3B and 7B groups use different
  inference stacks.
- **Execution deviations:**
  - A_plan / D_plan were not run on test_v5;
  - the exploratory X42 group was not run.

## Reproducibility

**Public, CPU only** (this repository alone):

```bash
python -m pip install -r requirements/ci.txt              # PyYAML only
python scripts/check_release.py                           # release guard
python scripts/verify_hashes.py                           # frozen scorer / manifest hashes
python -m unittest discover -s rag_finetuning/tests -t .  # data-dependent tests are skipped
python -m examples.minimal.run_example                    # scorer 4.0.3 + clustered bootstrap on public toy data
```

**Full experiments are *not* reproducible from this repository alone.** They require the unpublished datasets,
preference pairs and model outputs (see [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md)), the Qwen2.5
weights at the pinned revisions, and a CUDA GPU. The exact training stack is listed in
`requirements/train-cu128.txt`, and the full test_v5 freeze record is in
`rag_finetuning/experiments/p6_v5/configs/test_v5_final_freeze.json`.

Notes for training:
- set `HF_HUB_OFFLINE=1` and leave `TRANSFORMERS_CACHE` unset;
- set `P6_MEM_FRACTION=0.74` on Windows WDDM;
- weights and runs go to `./workspace/`.

Provenance, path redaction and excluded files are documented in `docs/`.

## License

- **Original code and documentation:** [Apache-2.0](LICENSE). See [NOTICE](NOTICE).
- **Not included:** model weights, adapters and model outputs are not part of this repository and are not
  covered by its licence.
- **Qwen licences:**
  - Qwen2.5-3B-Instruct is under the **Qwen Research License** (non-commercial only). Any use of the reported
    3B adapters or 3B outputs is subject to it.
  - Qwen2.5-1.5B and Qwen2.5-7B are Apache-2.0.
- **Built with Qwen.** See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the licence-risk notes in
  [docs/RELEASE_REPORT.md](docs/RELEASE_REPORT.md).
