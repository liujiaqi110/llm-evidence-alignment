# Third-party notices

This repository vendors **no** third-party code. The table lists what the experiments used at run time and the
licences checked for this release. Licence texts were read from the exact files that were used (model snapshot
`LICENSE` files in the local Hugging Face cache, the Ollama model's licence layer, and package `dist-info`).

## Models (not redistributed)

| Component | Exact version used | Licence (checked) | Notes for users of this repository |
|---|---|---|---|
| Qwen2.5-3B-Instruct | HF revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` | **Qwen Research License Agreement** (release date 2024-09-19; snapshot `LICENSE` sha256 `ef52482b…`) | **Non-commercial use only.** §3c attribution notice reproduced in `NOTICE`. §4b requires "Built with Qwen" / "Improved using Qwen" in the documentation of any model trained with Qwen outputs that you distribute. All reported v3/v4/v5 adapters (B, C, S, P groups) are derivatives of this model and are **not** published. |
| Qwen2.5-1.5B-Instruct | HF revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Apache-2.0 | Used in P6-v1/v2/v3 1.5B runs. |
| Qwen2.5-7B-Instruct (Ollama `qwen2.5:7b`) | Q4_K_M, digest `845dbda0ea48…`, Ollama 0.40.17 | Apache-2.0 (licence layer of the Ollama model) | 7B reference group D; also used to rewrite questions and draft candidates in data construction. |

## Software

| Package | Version | Licence |
|---|---|---|
| PyTorch | 2.11.0+cu128 | BSD-3-Clause |
| transformers | 4.57.6 | Apache-2.0 |
| peft | 0.22.0 | Apache-2.0 |
| trl | 0.23.1 | Apache-2.0 |
| accelerate | 1.16.0 | Apache-2.0 |
| datasets | 5.0.0 | Apache-2.0 |
| safetensors | 0.7.0 | Apache-2.0 |
| bitsandbytes | 0.50.2 | MIT |
| PyYAML | 6.0.3 | MIT |
| Ollama | 0.40.17 | MIT |

## Data

- **Published:** only the hand-written example set in `examples/minimal/` (Apache-2.0, part of this repository),
  the frozen-set manifests (per-item sha256 only) and aggregate statistics.
- **Not published:** the experiment datasets (synthetic fictional policy documents and Q&A, authored with an AI
  coding assistant and partly rewritten by Qwen2.5-7B), the preference pairs, candidate answers and all per-item
  model outputs (many are Qwen2.5-3B outputs, i.e. covered by the Qwen Research License), and the test_v5
  annotations. See `docs/DATA_AVAILABILITY.md`.
- Reports quote a small number of short model outputs for error analysis. These excerpts come from research
  use of the models and are included only as research documentation.

## Remaining licence risks

See the "Licence risks" section of `docs/RELEASE_REPORT.md`.
