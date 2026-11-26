# P6：RAG 生成模型 QLoRA 微调与独立评估

本目录是知识库平台第六阶段（需求 4.1「知识库项目模型微调」）的全部代码、数据和报告。
P6 只针对**生成模型**做微调，**不改动** BM25 / bge-m3 Dense / RRF / ACL 的任何代码。
P6 不修改 `ragas_eval/`、`kb_platform/`、`scripts/` 下的任何文件，生产演示仍默认使用 Ollama `qwen2.5:7b`。

- 实验结论：见 [reports/REPORT.md](reports/REPORT.md)。
- 指标口径：见报告第 4 节。

## 目录

| 路径 | 内容 |
|---|---|
| `common.py` | 语料切片：按 `##` 小节切分，Chunk ID 与平台同规则（md5 前 12 位）。复用平台 Prompt rag_v1.0（直接 import，不复制）。 |
| `configs/` | `qlora_r8.yaml`（预先确定的正式配置 v1）、`qlora_r8_v2_noderived.yaml`（事后消融） |
| `data/corpus/{train,val,test}/` | 原创合成语料：22 篇虚构制度文档，**按文档切分**，三个集合之间没有共享文档 |
| `data/qa/*.yaml` | 题目规格：问题、金标准小节、参考答案、评分要点、禁止内容、格式要求 |
| `data/built/*.jsonl` | 构建产物：证据片段及顺序、参考答案、来源、生成方式、审核状态。附 `manifest.json`（sha256、统计、泄漏检查） |
| `scripts/build_dataset.py` | 构建与校验。参考答案必须通过评分器自检；同时检查数据泄漏，并生成测试集审核表 |
| `scripts/smoke_fwd_bwd.py` | GPU 前向 / 反向冒烟测试 |
| `scripts/train_qlora.py` | QLoRA SFT：只对回答部分计算 loss，不静默截断，输出完整运行记录 |
| `scripts/collect_reports.py` | 把仓库外运行目录中的汇总结果收集到 `reports/results/` |
| `evaluation/generate.py` | 固定条件下逐题生成：同证据、同 Prompt、贪心解码、`max_new_tokens=384` |
| `evaluation/scoring.py` | 规则评分（纯标准库）、Wilson 区间、配对 bootstrap、McNemar |
| `evaluation/evaluate.py` | 汇总、配对比较、逐题 CSV |
| `evaluation/llm_judge.py` | 辅助 LLM 评审（qwen2.5:7b，结果为 provisional，不评自己的回答） |
| `evaluation/e2e_example_qa.py` | 通过平台原有 RAG 链路（RRF 检索 + 安全层）做端到端对比 |
| `inference/hf_llm.py` | 可选的本地推理适配层，接口与 OllamaClient 相同，挂在 ACL 与引用校验之后 |
| `tests/test_p6_dataset_and_scoring.py` | CI 可运行的单元测试（无模型） |
| `tests/local_acl_regression.py` | 本地 GPU 测试：用真实微调模型跑 ACL / 引用安全回归 |
| `reports/` | 实验报告、`results/` 汇总数据、`test_review_sheet.csv` 人工审核表 |

## 环境

- 训练环境：`<WORKSPACE>\venv`。
  - 用 `--system-site-packages` 复用 conda `ai` 环境里的 torch 2.11 + cu128，没有修改该环境。
  - 训练依赖装在 venv 内：transformers 4.57.6、peft 0.22.0、trl 0.23.1、bitsandbytes 0.50.2、accelerate 1.16.0、sentence-transformers 5.1.2。
- P1–P5 的 `.venv` 没有任何改动。
- 模型：`Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，约 3.1 GB，存放在 `<HF_HOME>\hub`。
- 运行目录 `<WORKSPACE>\runs`：存放 Adapter、日志、逐题原始生成。它在仓库外，**不进入 Git**。
- 注意：本机设置的 `TRANSFORMERS_CACHE` 指向另一个目录。运行前请先执行 `unset TRANSFORMERS_CACHE` 并设置 `HF_HUB_OFFLINE=1`。

## 复现

以下命令都在仓库根目录执行。`$PY` 指训练 venv 的 python，`$W` 指 `<WORKSPACE>/runs`。

```bash
python -m rag_finetuning.scripts.build_dataset                       # 构建 + 校验 + 泄漏报告
python -m unittest discover -s rag_finetuning/tests -t . -v          # 单元测试（无 GPU）
$PY -m rag_finetuning.scripts.smoke_fwd_bwd --config rag_finetuning/configs/qlora_r8.yaml
$PY -m rag_finetuning.evaluation.generate --model base --split test --out $W/eval/test_base_nf4          # 基线
$PY -m rag_finetuning.scripts.train_qlora --config rag_finetuning/configs/qlora_r8.yaml --max-steps 10 --tag smoke10
$PY -m rag_finetuning.scripts.train_qlora --config rag_finetuning/configs/qlora_r8.yaml                 # 正式训练 v1
$PY -m rag_finetuning.evaluation.generate --model base --adapter $W/qlora_r8_v1/adapter --split test --out $W/eval/test_qlora_r8_v1
python -m rag_finetuning.evaluation.generate --model ollama:qwen2.5:7b --split test --out $W/eval/test_ollama_7b   # 参考组
python -m rag_finetuning.evaluation.evaluate --split test --run base=$W/eval/test_base_nf4 \
       --run v1=$W/eval/test_qlora_r8_v1 --pair base,v1 --out $W/eval/report
python -m rag_finetuning.evaluation.llm_judge --run base=... --run v1=... --out $W/eval/judge_test.json
$PY -m rag_finetuning.evaluation.e2e_example_qa --data build/example_data --llm qlora --adapter $W/qlora_r8_v1/adapter --out $W/e2e/v1
$PY -m rag_finetuning.tests.local_acl_regression --adapter $W/qlora_r8_v1/adapter --base-too
python -m rag_finetuning.scripts.collect_reports --runs $W
```

## 不提交到 Git 的内容

以下内容都在 `.gitignore` 中，或放在仓库外的 E 盘目录：

- Adapter 权重（`*.safetensors`）
- Checkpoint
- 优化器状态
- 原始训练日志与逐题原始生成（jsonl）
- 模型缓存

仓库里只保存：原创合成数据集、汇总指标、逐题评分表、运行记录（参数、Loss 曲线、显存、版本号）。
