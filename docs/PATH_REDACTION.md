# Path redaction

The source tree contained local absolute paths, such as the experiment workspace, the source repository, the
Hugging Face cache and the Ollama model directory.

When this release was exported:
- **107** tracked source files contained such paths. This includes the 86 files found by the earlier extraction
  scan plus 21 files added later in P6-v5.
- **97** of these files are published, with the paths replaced.
- **10** were excluded for other reasons (per-item model outputs).
- `scripts/check_release.py` finds **0** remaining absolute paths.

## Rules

| Original (any `/`, `\` or `\` separator) | In code / YAML | In reports / JSON records |
|---|---|---|
| experiment workspace (`<drive>:/AI/rag_ft_workspace`) | `workspace` (relative to the repository root) | `<WORKSPACE>` |
| source repository checkout | — | `<SOURCE_REPO>` |
| private research repository (never accessed; only named in constraints) | — | `<PRIVATE_REPO>` |
| Hugging Face cache | — | `<HF_HOME>` |
| Ollama model / data directories | — | `<OLLAMA_MODELS>`, `<OLLAMA_DATA>` |
| source repository remote URL | — | `<SOURCE_REPO_REMOTE>` |
| `ragas_eval.rag_pipeline`, `ragas_eval.local_llm` imports | `rag_finetuning.compat.rag_pipeline`, `rag_finetuning.compat.local_llm` (27 imports in 19 files) | unchanged in prose |

Counts by replacement:

| Replacement | Occurrences |
|---|---|
| `<WORKSPACE>` | 105 |
| code/YAML `workspace` | 54 |
| `<SOURCE_REPO>` | 18 |
| `<PRIVATE_REPO>` | 3 |
| `<OLLAMA_MODELS>` | 2 |
| `<OLLAMA_DATA>` | 2 |
| `<HF_HOME>` | 1 |
| `<NEW_REPO>` | 1 |
| `<SOURCE_REPO_REMOTE>` | 1 |

Two descriptive path patterns in the release checklist and extraction plan were reworded by hand.

## Effect on recorded hashes

- **No hash-frozen file contained a path.** `scripts/verify_hashes.py` checks all five hash-frozen files against
  the records that froze them:
  - scorers `scoring_v2.py`, `scoring_v3.py`, `scoring_v4.py`;
  - `eng.py`;
  - the test_v5 manifest.
- **Line endings:** two of these files (`scoring_v4.py`, `test_v5.manifest.json`) were frozen as Windows CRLF
  bytes. They are kept byte-exact and marked `-text` in `.gitattributes`.
- **Other changed files:** scripts and configs that were edited only for path replacement now hash differently
  from their source versions. Their source sha256 and source commit are listed in
  [`PROVENANCE.md`](PROVENANCE.md).
