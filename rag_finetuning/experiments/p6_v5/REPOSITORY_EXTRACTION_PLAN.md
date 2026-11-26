# 独立仓库抽取规划：`llm-evidence-alignment`（只是规划，未执行）

> 本阶段**没有**创建远程仓库，没有推送，也没有删除或移动现有项目中的任何文件。
>
> 下面的命令、目录和清单都是草案，执行前需要用户确认。不访问 `<PRIVATE_REPO>`（私有研究仓库）。

## 1. 现状

| 项目 | 现状 |
|---|---|
| 来源仓库 | `<SOURCE_REPO>`，remote `origin = <SOURCE_REPO_REMOTE>`。P6 工作在本地分支上（`feature/rag-qlora-experiment` → `rag-sft-v3` → `rag-sft-v4` → `rag-preference-v5`），这些分支**都没有推送** |
| 与 P6 混在一起的其他内容 | P1～P5 知识库平台（`kb_platform/`、`ragas_eval/` 的大部分、`frontend/`、`config/`、`scripts/`、`tests_local/`），**不属于**本次抽取范围，也不修改 |
| 许可证 | 仓库根目录**没有 LICENSE 文件**，需要用户选择（建议 Apache-2.0 或 MIT） |
| 测试 | P6 的 8 组单元测试共 83 项，纯 CPU、只依赖 PyYAML，本机全部通过 |

## 2. P6-v1～v5 的代码依赖与版本关系

```
P6-v1  rag_finetuning/{common.py, scripts/, evaluation/, inference/, data/, configs/, reports/}
         └─ QLoRA 训练（train_qlora）、数据构建（build_dataset）、评分器 v1（evaluation/scoring.py）
P6-v2  experiments/p6_v2/   提示词消融、盲测集、冻结输入        ← 依赖 v1 common / scoring
P6-v3  experiments/p6_v3/   评分器 v2（scoring_v2）、语料与问答生成、train_v3、test_v3   ← 依赖 v1、v2 prompts
P6-v4  experiments/p6_v4/   评分器 v3（scoring_v3）、工程化推理（eng）、train_v4、test_v4 ← 依赖 v3 scoring_v2
P6-v5  experiments/p6_v5/   评分器 v4（4.0.2）、证据规划、偏好数据、DPO、诊断          ← 依赖 v4 scoring_v3、eng
```

对 P1～P5 平台代码的依赖，即 `rag_finetuning` 中 `from ragas_eval ...` 的引用：

| 被依赖模块 | 用到的符号 | 引用次数 | 处理方式 |
|---|---|---|---|
| `ragas_eval/rag_pipeline.py`（188 行，只依赖标准库） | `build_messages`、`SYSTEM_PROMPT`、`validate_citations`、`is_refusal_text` | 11 | **复制**为 `src/lea/compat/rag_pipeline.py`，**一字不改**，并记录来源 commit 和 sha256。提示词是所有 P6 结果的一部分，不能改 |
| `ragas_eval/local_llm.py` | `OllamaClient`、`GenerationError`、`GenResult`、`gpu_memory_used_mib` | 15 | 复制到 `src/lea/compat/local_llm.py`，记录来源 |
| `ragas_eval.acl` / `kb_service` / `rag_retriever` / `rag_service` / `retrieval_*`、`kb_platform.rag_backend`、`ragas_eval.tests.*` | 平台检索、ACL、端到端问答 | 各 1 | **不抽取**。引用它们的 `evaluation/e2e_example_qa.py`、`tests/local_acl_regression.py`、`evaluation/llm_judge.py` 的平台部分属于平台集成，留在原仓库 |

依赖方向是单向的：P6 依赖平台的两个小模块，平台不依赖 P6。复制这两个模块后，新仓库可以独立运行；原仓库不需要任何改动。

## 3. 目录与模块边界（草案）

```
llm-evidence-alignment/
├── README.md / README_CN.md        项目说明、结论摘要（负面结果放在显眼处）、复现入口
├── LICENSE                         待用户选择
├── CITATION.cff（可选）
├── pyproject.toml                  包名 lea；可选依赖组：[cpu]（测试 / 评分）、[train]（GPU 训练）
├── requirements/
│   ├── ci.txt                      PyYAML（CPU 测试）
│   └── train-cu128.lock.txt        训练环境的完整锁定版本（transformers 4.57.6、peft 0.22.0、trl 0.23.1、bitsandbytes 0.50.2、torch 2.11.0+cu128）
├── src/lea/
│   ├── compat/                     从平台复制的 rag_pipeline、local_llm（原样，带来源说明）
│   ├── common.py                   （v1 common）
│   ├── scoring/                    稳定：v1 scoring、v2、v3、v4（4.0.2）；4.0.3 提议单独放在 proposals/
│   ├── data/                       稳定：build_dataset、分词 / 去重 / 泄漏检查工具
│   └── train/                      稳定：train_qlora、SFT / DPO 入口
├── experiments/                    实验代码（按版本保留原结构，只调整 import 路径）
│   ├── p6_v1/ … p6_v5/
├── data/                           数据（见第 4 节）
├── reports/                        所有报告与审计表（按版本）
├── tests/                          P6 单元测试（83 项，纯 CPU）
├── workspace/（.gitignore）        本地权重、检查点、日志的默认位置
└── .github/workflows/tests.yml     最小 CI（见第 6 节）
```

稳定代码与实验代码的划分：

| 类别 | 内容 | 规则 |
|---|---|---|
| 稳定代码 `src/lea/` | 评分器各版本、数据构建、训练入口、平台兼容层 | 有单元测试；改动必须有版本号和修订记录 |
| 实验代码 `experiments/` | 各版本的构造、运行、诊断脚本 | 保留原样，作为结果的出处；只改 import 路径和本地路径 |
| 数据 `data/` | 原创合成语料、构建产物、偏好对、审核材料 | 见第 4 节 |
| 模型权重 | 适配器、检查点、量化缓存 | **不进仓库**；只发布 sha256 与训练记录 |
| 报告 `reports/` | 协议、修订、报告、审计 CSV、负面结果 | 全部保留 |

## 4. 数据发布范围

| 数据 | 是否公开 | 说明 |
|---|---|---|
| P6-v1 / v3 / v4 训练语料与构建产物 | 公开 | 原创合成的虚构制度文档 |
| P6-v1 test / val、P6-v2 盲测、test_v3、test_v4 及其参考答案 | 公开 | 已完成评测的旧测试集 |
| hard_dev（v5 开发集） | 公开 | 开发集，不作为泛化证据 |
| **test_v5 标注**：`data/test_v5/built/`、`data/test_v5/*.yaml`、corpus、`data/frozen/test_v5.jsonl` | **不公开** | 独立评测还没完成。只公开 `test_v5.manifest.json` 和 sha256 作为预注册承诺，评测完成后再补发 |
| v5 偏好对、SFT 新样本、候选回答、排除记录 | 公开 | 标注 `provisional（未经人工审核）` |
| 人工审核材料（审核台 HTML、抽样记录） | 公开 | 人工字段全部为空；审核完成后，另行发布审核结果 |
| 旧 7B 草稿 `data/train_v5/` | 公开（标注“未通过审查、未用于训练”） | 作为数据排除历史的证据。其中 `structured_questions.jsonl` 是未跟踪的大副本，与 `train_pref/structured.jsonl` 重复，不发布 |
| 模型权重 / 适配器 / 检查点 | **不公开** | 只发布 sha256、run_record |
| 运行日志 | 只发布摘要 | `run_record.json`、`status` 日志、OOM 摘录；不发布完整的工作区日志 |

## 5. 安全与合规检查（本次扫描结果）

| 检查 | 结果 | 处理 |
|---|---|---|
| 密钥 / Token（sk-、hf_、ghp_、AKIA、私钥、api_key 赋值） | `rag_finetuning`、`ragas_eval`、`kb_platform` 中**未发现** | 发布前用 gitleaks 或同类工具再扫一次 |
| 邮箱 | `rag_finetuning` 中未发现；`ragas_eval/redaction.py` 中的只是邮箱域名列表（脱敏规则），不是个人信息 | 不抽取 redaction.py |
| Windows 用户名、个人目录（用户主目录路径） | `rag_finetuning` 中未发现 | — |
| **本地绝对路径** | **86 个文件**：`<WORKSPACE>`（约 246 处）、`<SOURCE_REPO>`（15 处）、`<OLLAMA_MODELS>`、`<OLLAMA_DATA>`（3 处）。分布在 26 个 Python 脚本、17 个 YAML 配置和若干结果 JSON / 报告中 | 代码与配置：改为环境变量 `LEA_WORKSPACE`（默认 `./workspace`）和 `LEA_OLLAMA_URL`。历史结果 JSON 与报告：导出时把路径替换为 `<WORKSPACE>`，并在 `reports/PATH_REDACTION.md` 记录替换规则。被哈希记录的文件（冻结数据、训练数据、评分器）**都不含路径**，替换不影响任何已记录的哈希 |
| test_v5 内容外泄 | 125 道题的题目和参考答案只出现在 `data/test_v5/` 与 `data/frozen/test_v5.jsonl` 中，其他报告、脚本、数据都没有 | 发布时排除这两处 |
| 大文件 / 权重 | `rag_finetuning` 中没有 > 2 MB 的文件，也没有 .safetensors / .bin / .pt / .gguf | `.gitignore` 加入 `workspace/`、`*.safetensors` 等 |
| 私有研究仓库 | 没有引用 `<PRIVATE_REPO>` | — |

第三方许可证（发布前需核对原文）：

| 组件 | 用途 | 注意事项 |
|---|---|---|
| Qwen2.5-3B-Instruct | 训练基座；数据中含它的输出（候选回答、偏好对、评测输出） | **3B 版本的许可证与 7B 不同**（据我所知是 Qwen Research License，限制商业使用），对模型输出的再分发可能有条款，需要用户对照模型卡原文确认 |
| Qwen2.5-7B-Instruct（Ollama qwen2.5:7b） | 参照模型、问法改写 | 据我所知为 Apache-2.0，需核对 |
| transformers / peft / trl / accelerate / datasets | 训练 | Apache-2.0 |
| bitsandbytes | 4-bit 量化 | MIT |
| PyTorch | 训练 | BSD-3-Clause |
| Ollama | 本地推理 | MIT |

本仓库只发布代码、合成数据和模型输出，不发布权重。README 需写明：模型输出受基座模型许可证约束。

## 6. 可复现环境与最小 CI

- **CPU 环境**（评分、数据检查、全部单元测试）：Python 3.12 + PyYAML 6.0.3。83 项测试不需要 GPU、Ollama 或网络。
- **训练环境**：见 `requirements/train-cu128.lock.txt`。注意以下几点：
  - Windows WDDM 下需要 `P6_MEM_FRACTION=0.74`；
  - 需要 `HF_HUB_OFFLINE=1`，并且不能设置 `TRANSFORMERS_CACHE`；
  - 模型 revision 固定为 `aa8e72537993ba99e69dfaafa59ed015b17504d1`。
- **CI**（`.github/workflows/tests.yml`）：
  ```yaml
  on: [push, pull_request]
  permissions: { contents: read }
  jobs:
    tests:
      runs-on: ubuntu-latest
      timeout-minutes: 10
      env: { HF_HUB_OFFLINE: "1", PYTHONIOENCODING: utf-8 }
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with: { python-version: "3.12" }
        - run: python -m pip install -r requirements/ci.txt
        - run: python -m unittest discover -s tests -t . -v
        - run: python scripts/check_release.py   # 发布前检查：无权重 / 无 test_v5 标注 / 无绝对路径 / 冻结哈希一致
  ```
- **复现入口**：
  - `scripts/verify_hashes.py`：核对所有冻结数据和训练数据的哈希；
  - `scripts/rescore.py`：用公开的模型输出重算全部表格。不需要 GPU 就能复现报告中的数字。

## 7. 历史与负面结果的保留

新仓库使用全新的 git 历史，不复制旧仓库的 `.git`。保留历史的方式是把**内容**完整带过去：

- 所有负面实验：Planning v1 / v2 / v2.1 没有超过原始 RAG；v4 没有显著好于 v3；旧 7B 草稿只有 5.4% 通过；
- 所有 OOM 与失败记录：v3 / v4 的训练 OOM，v5 的 DPO 冒烟测试 r1～r6、最坏情况测试；
- 评分器修订：4.0.1、4.0.2，4.0.3 提议，以及 `freeze.json → amendments` 全文；
- 数据排除历史：`excluded.jsonl`、`assemble_report*.json`、v1.0 存档；
- `reports/PROVENANCE.md`：记录每个文件来自原仓库哪个分支、哪个 commit（如 `2181fc8`），让结论可以回溯。

## 8. 执行步骤（需要用户确认后再做）

1. 用户确认：仓库名、许可证、test_v5 的公开时间、Qwen 许可证核对结果；
2. 在新目录（建议 `<NEW_REPO>`）用导出脚本**复制**文件，不移动、不删除原文件；
3. 重写 import 路径和本地路径，运行 83 项测试和哈希核对；
4. 运行发布检查（`RELEASE_CHECKLIST.md`）；
5. `git init`，首次本地提交；
6. 用户再次确认后，再创建远程仓库并推送。本阶段不执行。
