# Release report: v1.0.0

v1.0.0 marks the completion of the **research engineering** work. It does **not** mean that any model passed
promotion. No model was trained, no test_v5 run was made and no experiment was added for this release.

## 1. Scope

**Published:**
- original code;
- configs, with paths redacted;
- tests;
- protocols and pre-registration records;
- aggregate statistics;
- reports, with paths redacted and the failure analysis redacted for test_v5;
- a hand-written public example.

**Not published:** test_v5 annotations and per-item results, all experiment datasets, the complete preference
data, per-item model outputs, weights, adapters and checkpoints. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)
and [EXCLUDED_FILES.md](EXCLUDED_FILES.md).

Every published file is listed in [RELEASE_FILES.txt](RELEASE_FILES.txt).

## 2. Directory structure

```
llm-evidence-alignment/
├── README.md, README_CN.md, LICENSE (Apache-2.0), NOTICE, THIRD_PARTY_NOTICES.md
├── pyproject.toml                     package llm-evidence-alignment 1.0.0 (module rag_finetuning); extra [train]
├── requirements/ci.txt                PyYAML (CPU checks)
├── requirements/train-cu128.txt       exact GPU stack used for every run
├── .github/workflows/tests.yml        CPU-only CI, Python 3.11 / 3.12, permissions: contents: read
├── scripts/check_release.py           release guard (weights, private data, abs paths, e-mails, secrets, size)
├── scripts/verify_hashes.py           frozen scorer / manifest hashes against the freeze records
├── examples/minimal/                  hand-written fictional data + answers; scorer 4.0.3 + clustered bootstrap
├── docs/                              DATA_AVAILABILITY, PROVENANCE (+ provenance.json), PATH_REDACTION,
│                                      EXCLUDED_FILES, RELEASE_REPORT, RELEASE_FILES.txt
└── rag_finetuning/
    ├── compat/                        verbatim copies of the platform prompt + Ollama client
    ├── common.py, evaluation/, inference/, scripts/, configs/, reports/     P6-v1
    ├── tests/                         86 original tests (17 skip without private data) + 4 public-example tests
    └── experiments/
        ├── p6_v2/                     prompt ablation, blind-set protocol (reports, prompts, manifests)
        ├── p6_v3/                     scorer v2, data engineering, 1.5B/3B SFT, ablations, aggregate results
        ├── p6_v4/                     scorer v3, structured data, dual seeds, metric audit, OOM / stall records
        └── p6_v5/                     scorer v4 (4.0.3), Oracle ablation, Planning v2/v2.1, evidence-first data,
                                       preference QC, SFT / DPO dual seeds, memory diagnostics,
                                       EVALUATION_PROTOCOL, configs/freeze.json, final/ (test_v5 reports)
```

## 3. Security and leakage checks

All checks were run on a **fresh clone** of the repository.

| Check | Tool | Result |
|---|---|---|
| Weights / checkpoints / adapters / DB / pickle | `scripts/check_release.py` | 0 |
| Unpublished data paths: test_v5 annotations, corpora, preference pairs, generations, review dashboards | `scripts/check_release.py` | 0 |
| Per-item data files outside the allow-list | `scripts/check_release.py` | 0 (allow-list: the public example, training-loss logs, `FINAL_METRICS_TABLE.csv`) |
| Local absolute paths (Windows drive paths, user home) | `scripts/check_release.py` | 0. Source had 107 affected files: 97 redacted, 10 excluded |
| E-mail addresses | `scripts/check_release.py` + manual grep | 0. Commit author uses the GitHub `users.noreply` address |
| Credential patterns (OpenAI / HF / GitHub / AWS / Slack / Google keys, private keys) | `scripts/check_release.py` | 0 |
| Secrets (entropy + keyword plugins) | detect-secrets 1.5.0, `scan --all-files` | 941 findings, **all** "Hex High Entropy String" in 16 files. Each is a sha256 hash in a manifest or provenance record, or a model / commit revision. No keyword, private-key or cloud-key finding |
| test_v5 content | n-gram scan (10 chars, ≥ 6 CJK) against all 125 test_v5 questions, reference answers and evidence, plus every required fact of 4 or more characters | 0 after redacting `FINAL_FAILURE_ANALYSIS.md`. The only residual matches are fixed answer templates shared by all dataset versions (refusal sentence, conflict template) and one phrase in the P6-v2 report about a v2 blind-set item that predates test_v5 |
| Private research repository | grep | Never accessed. Only named as `<PRIVATE_REPO>` in constraint text |
| P1–P5 platform code | review | Only the two verbatim `compat/` modules (prompt and Ollama client) |
| File size | `scripts/check_release.py` | Largest file 1.4 MiB (an aggregate summary JSON); total about 7.3 MiB |
| Git history | `git log` | New history: release commits only. The source repository's history was not copied |

**Source repository unchanged.** Before and after the export:
- remote: `origin` set to the original rag-pipeline URL (shown redacted);
- branch list: same 7 local branches;
- `main` and `origin/main`: both at `91977bf`;
- working tree: only the pre-existing untracked file.

**gitleaks** is not installed on the release machine, so it was not run. detect-secrets and the repository's own
guard were used instead.

## 4. Verification in a clean environment

The clean environment was a fresh `git clone` and a new Python 3.12 venv with only PyYAML 6.0.3 installed.

| Step | Result |
|---|---|
| `python scripts/check_release.py` | OK (300 files) |
| `python scripts/verify_hashes.py` | 5 / 5 frozen hashes match their freeze records |
| `python -m unittest discover -s rag_finetuning/tests -t .` | 85 run, OK, 17 skipped (unpublished data) |
| `python -m examples.minimal.run_example` | output identical to `examples/minimal/expected_output.json` |
| `python -m build` + install wheel | builds sdist + wheel; installed package imports and reports scorer 4.0.3 |
| Same tests with the private data placed back (scratch copy, deleted afterwards) | 90 run, OK, 0 skipped (86 original + 4 new) |
| Static checks | all modules compile; every internal import resolves; 48 relative Markdown links resolve |

**CI:** see section 7. It was filled in after the first push.

## 5. Reproducibility boundary

**Reproducible publicly:** scorer behaviour, statistics code, the release guard, hash verification, the public
example and all non-data unit tests.

**Not reproducible publicly:** any reported experimental number. This needs the unpublished data, the Qwen
weights at the pinned revisions and a GPU. The repository does **not** claim that the public can fully reproduce
the final experiments.

## 6. Licence risks (remaining)

1. **Qwen2.5-3B-Instruct is under the Qwen Research License (non-commercial).**
   - Nothing derived from it is distributed here: no adapters, no bulk outputs.
   - The reports quote a small number of short 3B outputs for error analysis. These are research documentation,
     but their status under the licence (whose definition of *Materials* does not explicitly cover outputs) has
     not been confirmed by legal review.
   - Anyone who publishes the adapters or outputs later must comply with §3 (licence copy and notice) and §4b
     ("Built with Qwen").
   - The Apache-2.0 licence of this repository does not extend to Qwen outputs.
2. **Synthetic datasets were not published.** They were authored with an AI coding assistant and partly
   rewritten by Qwen2.5-7B (Apache-2.0). If they are released later, they need their own licence decision and a
   check of the AI assistant provider's terms.
3. **The `compat/` modules** come from the authors' own platform repository, which has no LICENSE file. They are
   relicensed here under Apache-2.0 by the same authors. The owner should confirm this.
4. **Copyright holder:** NOTICE uses "The llm-evidence-alignment contributors". The owner may want to replace it
   with a personal or organisational name before making the repository public.
5. **Third-party packages** are not vendored, so only the run-time licences listed in THIRD_PARTY_NOTICES.md
   apply.

## 7. GitHub publication

| Item | Value |
|---|---|
| Repository | https://github.com/liujiaqi110/llm-evidence-alignment |
| Visibility | **public**. The repository was created as public; the owner decided to keep it public before the first push (2026-11-26) |
| Default branch | `main` (fresh history; no history of the source repository) |
| Tag | `v1.0.0` → `da5e343` (annotated). Research engineering version complete; **no model promoted** |
| CI | GitHub Actions `tests`. All steps succeeded on Python 3.11 and 3.12 (release guard, frozen hashes, unit tests, minimal example, package build) |

CI runs:

| Run | Ref | Commit | Result |
|---|---|---|---|
| [45736687708](https://github.com/liujiaqi110/llm-evidence-alignment/actions/runs/45736687708) | `main` | `da5e343` | success |
| [45736701793](https://github.com/liujiaqi110/llm-evidence-alignment/actions/runs/45736701793) | `v1.0.0` | `da5e343` | success |

The commit that adds this section is after the tag, so it is on `main` but not in `v1.0.0`.
