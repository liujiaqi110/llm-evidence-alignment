# P6-v3：高质量 SFT 数据工程与 QLoRA 优化实验

- 分支：`feature/rag-sft-v3`，基于 P6-v2 的 `2e8e2e6`。
- P6-v1 和 P6-v2 的全部结果原样保留，本阶段**没有修改**它们的代码、数据或报告。
- 本阶段也没有修改 P1–P5 的检索、ACL、认证或前端代码。

## 阅读顺序

1. [reports/REPORT.md](reports/REPORT.md)：完整验收报告，回答本阶段的 8 个问题。
2. [reports/MODEL_DECISION.md](reports/MODEL_DECISION.md)：模型晋级判定。
3. [reports/ERROR_ANALYSIS.md](reports/ERROR_ANALYSIS.md)：改善、退化与失败案例。
4. [DATASET_CARD.md](DATASET_CARD.md)：data_v3 的来源、构造规则、质量检查和已知问题。
5. [TRAINING_CONFIG.md](TRAINING_CONFIG.md)：训练配置与实际运行记录。
6. [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)：测试集隔离、指标定义、统计方法、晋级门槛（训练前冻结）。
7. `reports/HUMAN_REVIEW.csv`（公开版未包含：含逐题数据，见 docs/DATA_AVAILABILITY.md）：人工复核表，**尚未进行人工审核**。

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/audit_v1.py` | P6-v1 训练管线与数据审查，含 GPU 对数概率诊断 |
| `scripts/gen_corpus.py`、`scripts/gen_qa.py` | 用本机 qwen2.5:7b 起草训练 / 开发文档与问答草稿，原始草稿缓存在 `data/drafts/` |
| `scripts/build_v3.py` | 构建 data_v3 及 4 个单因素消融变体，并运行自动质量检查 |
| `scripts/build_test_v3.py` | 构建并封存手写的独立测试集 test_v3（81 题） |
| `scripts/scoring_v2.py` | 评分器 v2（冲突、拒答、陈述—证据支持、按文档聚类的配对 bootstrap） |
| `scripts/train_v3.py` | QLoRA 训练：复用 P6-v1 管线，每个 epoch 存 checkpoint，在开发集上按预注册规则选停止点 |
| `scripts/run_eval.py` | 冻结输入 / 统一推理 / 分析 |
| `scripts/judge_v3.py` | 辅助 LLM 评审（qwen2.5:7b，不评 F 组，结果为 provisional） |
| `configs/` | 训练配置与 `freeze.json`（训练前冻结的评估方案） |
| `data/` | 语料、草稿、data_v3、test_v3、冻结输入 |
| `results/`、`ablations/` | 汇总指标、逐题对照、错误案例、训练记录 |

## 复现

在仓库根目录执行。训练与 HF 推理要用训练 venv，并先执行 `unset TRANSFORMERS_CACHE`、设置 `HF_HUB_OFFLINE=1`。

```bash
python -m rag_finetuning.experiments.p6_v3.scripts.build_v3 --calibrate
python -m rag_finetuning.experiments.p6_v3.scripts.build_test_v3
$PY -m rag_finetuning.experiments.p6_v3.scripts.train_v3 --config rag_finetuning/experiments/p6_v3/configs/v3_1p5b_full.yaml
$PY -m rag_finetuning.experiments.p6_v3.scripts.run_eval gen --group C --input test_v3 --out <dir>/test_v3/C
python -m rag_finetuning.experiments.p6_v3.scripts.run_eval analyze --input test_v3 \
       --ref rag_finetuning/experiments/p6_v3/data/test_v3/built/test_v3.jsonl --runs <dir>/test_v3 --out <out>
python -m unittest rag_finetuning.tests.test_p6v3_scoring_and_data
```

模型权重、Adapter、checkpoint 和大型缓存只保存在 `<WORKSPACE>`，不提交到 Git。
