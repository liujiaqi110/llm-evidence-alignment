# llm-evidence-alignment

**面向证据的 LLM 后训练与偏好优化（Evidence-Aware LLM Post-Training & Preference Optimization）。**

本仓库是研究代码：在单块 8 GB 笔记本 GPU 上，用 4-bit QLoRA 训练小型指令模型 Qwen2.5-3B，让它依据检索到的证据作答。目标包括：
- 引用正确的片段；
- 写全分散在多个片段中的必要事实；
- 证据缺失时拒答；
- 指出部分回答和冲突；
- 抵抗提示词注入。

> **状态：研究工程版本 v1.0.0，没有任何模型通过晋级。**
> - 在最终预注册的独立测试 test_v5（125 题）上，SFT 和 DPO 候选都**没有在两个种子上同时通过全部 7 条晋级门槛**，因此不宣称生产模型替换成功。
> - 所有指标都来自规则评分器和 AI 编写的标注，**没有经过独立人工审核，全部为 provisional**。
>
> English version: [README.md](README.md)

## 研究目标

在 8 GB 显存预算下，在 Evidence-first SFT 之上做偏好优化（DPO），能否让 3B 模型整合**多条证据**（前几个版本的主要失败点），同时不让拒答、引用和安全行为退化？

所有结论都按最终测试运行前冻结的门槛判定。

## 系统架构

各模块的代码位置见 [README.md](README.md#system-architecture)：

| 环节 | 内容 |
|---|---|
| 数据构造 | 虚构制度语料 → Evidence-first 构造（必要事实 + 支持片段）→ 泄漏检查（按文档隔离、与 dev/test 的 n-gram 和哈希比对）→ 偏好对构造与质量控制 |
| 训练 | QLoRA SFT（S42 / S43）→ 以同种子 SFT 为初始策略和参考模型做 DPO（P42 / P43，`use_logits_to_keep`，8 GB） |
| 检查点选择 | 按预注册规则，只在 hard_dev 上选 |
| 生成 | 冻结的平台 Prompt（`compat/rag_pipeline.py`），贪心解码，最多 384 token |
| 评分与统计 | 规则评分器 4.0.3 → 源文档聚类配对 bootstrap → 7 条预注册晋级门槛 |

## 实验方法

- **预注册**：每次测试前，冻结协议、门槛、评分器版本和数据哈希（`freeze.json`、`EVALUATION_PROTOCOL.md`、`test_v5_final_freeze.json`）。修订都留有记录，从不默默生效。
- **双种子**：每个配置跑 42 / 43 两个种子，分别报告。两个种子在同一批题上评分，所以均值只作描述。
- **检查点选择**：只用 hard_dev。test_v5 在全部训练结束后只运行一次。
- **统计方法**：按源文档聚类的配对 bootstrap，给出 95% CI（test_v5 共 23 个簇）。
- **test_v5 评测组**：

  | 组 | 模型 |
  |---|---|
  | A | 3B Base |
  | B | v3 |
  | C42 / C43 | v4 |
  | S42 / S43 | v5 SFT |
  | P42 / P43 | v5 DPO |
  | D | 7B Q4_K_M（Ollama），参照组 |

  所有组使用相同的证据、Prompt 和评分器。

## Evidence-first 数据工程

- **做法**：先选定证据片段和必要事实，再构造问题，而不是让模型自由起草答案。
  - 自然样本中，161/212 条通过全部自动检查。
  - 早期让 7B 自由起草的做法，只有 14/259 条通过。
- **SFT 数据**：
  - 初始数据 476 条，即 v3 原有 315 条加 161 条证据优先样本。
  - **正式训练数据 472 条**：按修正 13 排除了 4 条超长样本。
- **DPO 数据**：最终 **580 对**，由 791 对候选筛选得到，所有排除都有记录。

## QLoRA / SFT / DPO

- **模型与量化**：Qwen2.5-3B-Instruct（revision `aa8e7253…`），NF4 双重量化，bf16 计算，LoRA r = 8。
- **SFT**：训练 3 个 epoch，两个种子都选中 epoch3。
- **DPO**：训练 2 个 epoch（146 步），两个种子都选中 checkpoint-108；每个种子约 1.8–1.9 小时。

## 最终评测结果（test_v5，125 题，评分器 4.0.3，provisional）

| 指标 | A | B（v3） | C42 | C43 | S42 | S43 | P42 | P43 | D（7B）\* |
|---|---|---|---|---|---|---|---|---|---|
| 多证据完整（/80，主指标） | 38 | 35 | 50 | 45 | 44 | 42 | 55 | **57** | 54 |
| 内容正确（/125） | 57 | 65 | 83 | 78 | 77 | 77 | 82 | **92** | 83 |
| 应回答题的整体拒答（/113，越低越好） | 15 | 11 | 4 | 3 | 9 | 6 | 3 | **0** | 1 |
| 正确拒答（/12） | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 11 | 12 |
| 部分回答（/8） | 2 | 3 | 5 | 5 | 6 | 7 | 3 | 5 | **8** |
| 真冲突识别（/8） | 0 | 0 | 2 | 2 | **6** | 3 | 2 | 5 | 2 |
| 注入抵抗（/6） | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| p50 延迟（秒） | 4.5 | 7.9 | 10.3 | 10.0 | 9.7 | 9.5 | 11.5 | 12.7 | 4.1\* |

\* D 的推理框架不同：D 用 Ollama / llama.cpp（GGUF Q4_K_M），3B 各组用 transformers + bitsandbytes NF4，LoRA 未合并。因此 D 的延迟与 3B 组**不可直接比较**。

**核心结果：DPO 相对同种子 SFT 的多证据完整率**

| 比较 | 变化 | 聚类 bootstrap 95% CI |
|---|---|---|
| P42 vs S42 | 44 → 55，**+13.8 个百分点** | [+3.6, +24.4] |
| P43 vs S43 | 42 → 57，**+18.8 个百分点** | [+8.2, +29.4] |

两个种子的整体拒答都减少、引用正确率都提高，CI 都不含 0。

**晋级门槛**（7 条全部满足，且两个种子都满足，才能晋级；比较对象为 v3，门槛 2 的比较对象为 C42）

| 候选 | 未通过的门槛 | 判定 |
|---|---|---|
| S42 | 2（内容正确率相对 v4） | 不晋级 |
| S43 | 1、2 | 不晋级 |
| P42 | 6c（证据外数字 7 题，上限 6）；7（延迟为 v3 的 1.47 倍） | 不晋级 |
| P43 | 7（延迟为 v3 的 1.61 倍） | 不晋级 |

**质量改善 ≠ 综合晋级**
- 在质量指标上，DPO 两个种子都显著优于 v3：多证据 +25.0 / +27.5 个百分点，内容正确 +13.6 / +21.6 个百分点。
- 但包含延迟和数字安全的综合门槛，两个种子都没有通过。

完整报告见 `rag_finetuning/experiments/p6_v5/final/`：

| 文件 | 内容 |
|---|---|
| [FINAL_EVALUATION_REPORT](rag_finetuning/experiments/p6_v5/final/FINAL_EVALUATION_REPORT.md) | 最终评测报告 |
| [PROMOTION_GATE_REPORT](rag_finetuning/experiments/p6_v5/final/PROMOTION_GATE_REPORT.md) | 晋级门槛报告 |
| [FINAL_FAILURE_ANALYSIS](rag_finetuning/experiments/p6_v5/final/FINAL_FAILURE_ANALYSIS.md) | 失败分析（公开版已脱敏） |
| [P6_RESEARCH_SUMMARY](rag_finetuning/experiments/p6_v5/final/P6_RESEARCH_SUMMARY.md) | P6-v1 至 v5 研究总结 |
| [FINAL_METRICS_TABLE.csv](rag_finetuning/experiments/p6_v5/final/FINAL_METRICS_TABLE.csv)、[STATISTICAL_COMPARISON.json](rag_finetuning/experiments/p6_v5/final/STATISTICAL_COMPARISON.json) | 聚合指标与统计比较 |

## 统计分析

| 比较 | 结论 |
|---|---|
| DPO vs v4 | 无法判断：与 C42 比，CI 含 0；与 C43 比，两个种子都显著更好。 |
| 3B DPO vs 7B Q4（内容正确率、多证据完整率） | **本次评测中未观察到统计显著差异**（P43 − D 内容正确率 +7.2 个百分点，CI [−3.4, +17.8]）。这**尚不能证明等效或非劣效**：没有预设界值，样本只有 125 题，评分也未经人工审核。 |
| v5 SFT vs v4 | 主指标没有超过 v4。 |
| P42 与 P43 的一致性 | 多证据 66/80 题结果相同，内容 101/125 题结果相同。 |
| 跨文档、3 个及以上片段子集 | 样本太小（各 17 题），无法判断。 |

## 显存优化

- **问题**：在 8 GB 显卡上，DPO 在 1024、768 以及 1024 + 预计算参考 logp 三种配置下都 OOM。
- **原因**：计算了全词表 logits。
- **解决**：改用 `use_logits_to_keep=True`（修正 3），只计算回答部分的 logits。
- **数值等价性验证**：logp 差 ≤ 1.5e-5，梯度余弦 ≥ 0.99991。
- **最坏情况验证**：`P6_MEM_FRACTION=0.74` 可行，0.80 会 OOM。
- **实测显存**：正式 DPO 峰值 reserved 为 5834 / 5798 MiB；3B 推理为 2160 MiB allocated / 3116 MiB reserved。

## 负向结果（有意保留）

- **v1**：1.5B QLoRA 的格式改善了，但内容没有改善（28 → 25/47），并且过度拒答。
- **v2**：Few-shot 对 7B 没有可靠的内容收益。
- **v3**：过度拒答的根因是训练目标中拒答句重复了 28 次；3B v3 的多证据完整率退化（6 → 3/19）；冲突识别所有模型都是 0/4。
- **v4**：内容正确率超过 v3，但多证据问题没有修复，两个种子都未通过门槛，不推荐使用。
- **Planning（先抽取事实清单再生成）**：低于原始 RAG（17 / 20 vs 26/43）。Oracle 事实清单能达到 41/43，说明瓶颈在"识别要写哪些事实"。
- **DPO**：
  - 部分回答退化（P42 为 3/8）；
  - 对冲突识别的影响，两个种子方向相反；
  - P43 新增 1 道拒答错误；
  - 回答更长，照抄证据原文更多。
- **注入抵抗**：所有方法都没有改善。6 道注入题中有 4 道，9 组全部失败，包括 7B。

## 局限

- **评分器**：规则评分器偏向照抄；会把正确的计算判为"证据外数字"；对部分回答的判定依赖固定的缺口表述。
- **人工审核未完成**：

  | 材料 | 已审核 |
  |---|---|
  | 偏好对 | 16/167 |
  | hard_dev | 15/26 |
  | test_v5 标注 | 0/125（全部由 AI 编写） |

- **样本量**：test_v5 只有 125 题、23 个源文档；部分子集只有 6–12 题，单题变化就可能改变门槛判定。
- **硬件与推理**：延迟只在一块笔记本 GPU 上测得；3B 与 7B 的推理框架不同。
- **执行偏离**：A_plan / D_plan 没有在 test_v5 上运行；探索组 X42 没有执行。

## 可复现性

**公开部分（只需 CPU）**：

```bash
python -m pip install -r requirements/ci.txt
python scripts/check_release.py
python scripts/verify_hashes.py
python -m unittest discover -s rag_finetuning/tests -t .   # 依赖未公开数据的测试会跳过
python -m examples.minimal.run_example
```

**完整实验无法仅凭本仓库复现**，还需要：
- 未公开的数据集、偏好对和模型输出（见 [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md)）；
- 固定 revision 的 Qwen2.5 权重；
- CUDA GPU。

训练环境见 `requirements/train-cu128.txt`。

## 许可证

- **原创代码与文档**：[Apache-2.0](LICENSE)，另见 [NOTICE](NOTICE)。
- **不包含的内容**：模型权重、适配器和模型输出不在本仓库中，也不受本仓库许可证覆盖。
- **Qwen 许可证**：
  - Qwen2.5-3B-Instruct 使用 **Qwen Research License**，**仅限非商业用途**。使用报告中的 3B 适配器或 3B 输出须遵守该协议。
  - Qwen2.5-1.5B / 7B 为 Apache-2.0。
- **Built with Qwen。** 详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [docs/RELEASE_REPORT.md](docs/RELEASE_REPORT.md)。
