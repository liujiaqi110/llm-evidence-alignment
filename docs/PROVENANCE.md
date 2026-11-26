# Provenance

## Source and history

- **Source:** the `rag_finetuning/` tree of the authors' local research checkout, branch
  `feature/rag-preference-v5`, commit `87839f1`.
- **New history:** this repository was created with a fresh `git init`. The source repository's Git history was
  **not** copied, because it contains test annotations, model outputs and unrelated platform code.
- **Source left untouched:** the source repository, its branches and its remote were not modified by the export.

## Per-file record

[`provenance.json`](provenance.json) lists, for every published file:
- its source path and source commit;
- the sha256 of the source blob;
- whether the file was modified for this release.

Of the 273 exported files, 112 were modified. The modifications are limited to:

1. **Path redaction** and the `ragas_eval.*` → `rag_finetuning.compat.*` import rewrite. See
   [`PATH_REDACTION.md`](PATH_REDACTION.md).
2. **Tests that need unpublished data now skip when the data is absent.**
   - The helper is `rag_finetuning/tests/_private_data.py`.
   - Affected tests: `TestDataset`, `TestDataV3`, the P6-v2 prompt and freeze module, and
     `test_frozen_inputs_have_no_label_fields`.
   - Assertions are unchanged. With the private data in place, all 86 original tests run.
3. **`p6_v5/final/FINAL_FAILURE_ANALYSIS.md`** (public edition). A few sentences, facts and numbers quoted from
   test_v5 answers, evidence or required facts were removed. They are marked "公开版不引用原句" or "（略）".
   Statistics and conclusions are unchanged.
4. **`RELEASE_CHECKLIST.md` / `REPOSITORY_EXTRACTION_PLAN.md`**: two descriptive path patterns were reworded.
   These two files are kept as historical planning documents; [`RELEASE_REPORT.md`](RELEASE_REPORT.md) records
   what was actually done.

## Compat modules

`rag_finetuning/compat/rag_pipeline.py` and `rag_finetuning/compat/local_llm.py` are **verbatim** copies of
`ragas_eval/rag_pipeline.py` and `ragas_eval/local_llm.py` at the same source commit. They come from the authors'
own knowledge-base platform and are included because:
- the frozen system prompt and refusal sentence used by every P6 result are defined there;
- the Ollama client used for the 7B reference group is defined there.

Their source sha256 values are in `provenance.json`.

## Differences from `REPOSITORY_EXTRACTION_PLAN.md`

| Plan | Release | Reason |
|---|---|---|
| package `src/lea/` with rewritten imports | package name `rag_finetuning` kept | Keeps every experiment script runnable under its recorded module path (`python -m rag_finetuning.experiments…`) with minimal edits |
| publish v1–v4 datasets, hard_dev, preference pairs, model outputs | **not published** | The release authorisation limits the public scope to code, configs, tests, methods, aggregate statistics and redacted reports. Complete preference data and uncleared model outputs are explicitly excluded |
| `scripts/rescore.py` to recompute all tables from public outputs | not provided | Per-item outputs are not public. Rescoring needs the private data (see `DATA_AVAILABILITY.md`) |

## Authorship note

- The code, protocols and reports were written in a human-directed workflow with an AI coding assistant
  (Claude Code).
- The synthetic experiment datasets (not published) were authored with the same assistant and partly rewritten
  by Qwen2.5-7B.
- No result in this repository has been independently human-reviewed. See the status notes in every report.
