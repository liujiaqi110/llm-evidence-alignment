# P6-v4：多证据、冲突识别与 SFT 数据工程

**结论**：v4 没有达到预注册的晋级标准，不作为推荐模型。详见 `REPORT.md` 与 `MODEL_DECISION.md`。

> 所有结果都是自动评分，provisional，未经人工审核。

## 文档

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 总报告，含 8 个问题的回答 |
| `METRIC_AUDIT.md` | v3 指标重算与口径审计 |
| `ERROR_ANALYSIS.md` | v3 退化诊断（第 1–5 节）+ v4 错误分析（第 6 节） |
| `DATASET_CARD.md` | data_v4 数据卡 |
| `EVALUATION_PROTOCOL.md` | test_v4 冻结方案、评分规则、门槛 |
| `MODEL_COMPARISON.md` | 各组对比（test_v4 与 test_v3 回归） |
| `ABLATION_REPORT.md` | 工程方案 E1 / E2 与微调贡献的拆分 |
| `MODEL_DECISION.md` | 门槛判定与决策 |
| `HUMAN_REVIEW_V4.csv` | 人工审核材料（v3 test 81 题；人工字段为空） |

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/` | metric_audit、diagnose_v3、make_review_v4、scoring_v3、build_test_v4、build_v4、eng、eng_dev、train_v4、run_eval_v4、regression_v4 |
| `configs/` | 训练配置（s42 / s43）、`freeze.json`（冻结哈希、门槛、规则） |
| `data/test_v4/` | 11 个手写文档、题目 YAML、构建结果 |
| `data/v4/` | 训练集、开发集、manifest |
| `data/frozen/` | test_v4、reg_test_v3 冻结副本 |
| `results/` | 指标审计、诊断、工程开发集标定、训练记录、test_v4 与回归结果 |

## 复现

```bash
# 训练（训练环境；Windows 上建议设置显存上限）
P6_MEM_FRACTION=0.74 python -m rag_finetuning.experiments.p6_v4.scripts.train_v4 --config rag_finetuning/experiments/p6_v4/configs/v4_3b_s42.yaml
# 生成（每组只允许一次，已有结果时拒绝覆盖）
python -m rag_finetuning.experiments.p6_v4.scripts.run_eval_v4 gen --group C42 --input test_v4 --out <runs>/eval/test_v4/C42
# 分析
python -m rag_finetuning.experiments.p6_v4.scripts.run_eval_v4 analyze --input test_v4 --ref rag_finetuning/experiments/p6_v4/data/test_v4/built/test_v4.jsonl --runs <runs>/eval/test_v4 --out rag_finetuning/experiments/p6_v4/results/test_v4
python -m rag_finetuning.experiments.p6_v4.scripts.regression_v4
```

模型权重、检查点、原始生成和日志都在 `<WORKSPACE>\runs\p6_v4`，不提交到仓库。
