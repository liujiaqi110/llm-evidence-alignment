# P6-v5 DPO 实验计划（预注册草案；尚未开始任何正式训练）

> 本计划在任何 v5 正式训练和任何 test_v5 运行之前写定。门槛、评分器（4.0.2）、冻结评测集哈希都沿用 `configs/freeze.json`，不因本计划改变。
> 本计划里的数据都是 **provisional**：偏好对和 hard_dev 都还没有人工审核。

## 1. 实验组

| 组 | 内容 | 初始化 | 数据 | 状态 |
|---|---|---|---|---|
| **C42 / C43** | v4 基线（已有） | — | P6-v4 训练集 | 已训练；hard_dev 已评 |
| **S42 / S43** | 针对性 SFT | 3B Base | `data/train_pref/sft_train.jsonl`：v3 315 条 + 证据优先多证据样本 161 条 | 未训练 |
| **P42 / P43** | SFT + DPO | 分别从 S42 / S43 的 hard_dev 选中适配器开始；参考 = 同一适配器的冻结副本 | `data/train_pref/dpo_pairs.jsonl`（791 对） | 未训练 |
| **X42**（探索组，单独预注册） | 直接在 v4 上做 DPO | v4 s42 适配器；参考 = v4 s42 冻结副本 | 同上 | 未训练 |

对照基准：B（3B v3）是晋级门槛的比较对象，A（3B Base）和 D（7B）照常报告。

### 1.1 探索组 X42 的规则

- 只跑种子 42，**不参与晋级判定**，不得写成“DPO 优于 SFT”或“v5 改善”的证据。
- 结果单独成节，标“探索性、单种子”。
- 无论结果好坏都报告。
- X42 的偏好对里有一部分 rejected 来自 v4 自己（78 对 `real:C42`），这对 X42 是“同策略”负例，对 P 组则是离策略负例。两组的差异不能只归因于初始化。

## 2. 训练配方（与 freeze.json 一致）

- **SFT**：与 P6-v3 / v4 完全相同。
  - Qwen2.5-3B-Instruct@aa8e7253，4-bit NF4。
  - LoRA r8 / α16 / dropout 0.05，7 个线性层。
  - lr 2e-4 cosine，warmup 0.1，3 epoch，有效批 8，max_seq_len 1024，只对回答计算损失。
  - 每个 epoch 保存一次。
- **DPO**：TRL 0.23.1 DPOTrainer。
  - sigmoid loss，β 0.1，lr 2e-5 cosine，warmup 0.1，2 epoch。
  - 批 1 × 梯度累积 8，max_length 1024，max_prompt_length 896，梯度检查点。
  - 每半个 epoch 保存一次。
  - 策略适配器名 `policy`，参考适配器名 `reference`。冒烟测试发现原来的名字 `train` 与 `nn.Module.train` 冲突，已改名。
- **检查点选择**：在 hard_dev 上依次比较 multi_complete_v5 最高 → content_ok_v4 最高 → whole_refusal 最少 → 更早的检查点。test_v5 不参与选择。
- **显存设置**：见 `DPO_MEMORY_SMOKE_REPORT.md`。冒烟测试结论：只有 1024 + `use_logits_to_keep` 能运行；1024、768、1024 + 预计算参考 logp 都 OOM。
  - 如果正式训练需要协议之外的显存选项（`use_logits_to_keep`），必须先在 freeze.json 写入修订 3，再开始训练。
  - 协议原本只允许调整序列长度和预计算参考 logp。
- **数据文件**：训练前记录 `dpo_pairs.jsonl` 和 `sft_train.jsonl` 的 SHA-256，并在 run_record 中复核。
  - 当前值见 `data/train_pref/pairs_report.json`：`204b40b2…`、`d60e7e9c…`。
  - 如果人工审核后数据有变化，记录新的哈希和变化原因，旧版本存档。

## 3. 正式训练前必须满足的条件（当前**均未满足**）

| # | 条件 | 当前状态 |
|---|---|---|
| 1 | 偏好对人工抽审：H 类至少 60 对、chosen 为参考答案的至少 40 对、C / I 人工构造对全部（59 对），填写 `PREFERENCE_PAIR_AUDIT.csv` 人工字段 | 未开始 |
| 2 | hard_dev 高优先级 24 题人工审核（`HUMAN_REVIEW_CHECKLIST.csv`）。用于确认检查点选择依据可靠，以及中文数字、同义表达、推算数字三类争议是否需要在 test_v5 前修订评分器 | 未开始 |
| 3 | DPO 冒烟测试在 1024 长度下完成 10 步、日志中有 `rewards/chosen`、`rewards/rejected`、`rewards/margins` | ✅ 已满足，但需要 `use_logits_to_keep`（修订 3 提议，待批准）；最长样本最坏情况未测 |
| 4 | 用户明确授权开始正式训练 | 未授权（本阶段要求不训练） |

### 3.1 人工审核结果的预设处理（在看到审核结果前写定）

- **规则 1（H 类）**：在抽审的 H 类中，如果“偏好不明确”超过 15%，正式 DPO 删除“rejected 事实覆盖率 ≥ chosen 的 H 类真实对”。按当前数据，这类对最多 135 对。
- **规则 2（参考答案作 chosen）**：在抽审的参考答案 chosen 中，如果“chosen 有错或不自然到不应学习”超过 10%，这些对改用模型自己的正确回答作 chosen；没有正确回答的题删除。
- **规则 3（人工构造对）**：人工审核判为“rejected 并无明确错误”的对，一律删除。
- **规则 4（新增审核）**：审核后剩余的对少于 600 时，不补造新对，按剩余数据训练并在报告中注明。
- 以上规则是按审核结果**删除**数据，不会为了指标增加或改写数据；所有删除都写入 `excluded.jsonl`。

## 4. 评测流程

1. **hard_dev**：S42 / S43、P42 / P43、X42 的检查点选择，以及与 C42 / C43 的开发集对比。开发集结果不作为泛化证据。
2. **test_v5**：
   - 选定检查点后，每组只生成一次；`run_eval_v5 gen` 对 test 输入拒绝覆盖。
   - 组别：A、B、C42、C43、S42、S43、P42、P43、D；X42 单独。
3. **晋级判定**：按 freeze.json 的 7 条门槛，与 B 比较。
   - 每个种子分别判定，聚类 bootstrap 按源文档。
   - 两个种子都通过才称“优于 v3（自动评测、provisional）”。
4. **DPO 有效性**：
   - 日志中必须有偏好指标，才能称为 DPO。
   - 另报告训练集上 rewards/accuracies 和 margins 的走势，以及 hard_dev 上 P 相对 S 的变化。
5. **额外监控（不作门槛，必须报告）**：
   - **回答长度**：P 组正文平均长度相对 S 组的比值。超过 1.3 时，标注“可能学到了长度偏好”。
   - **文风**：P 组回答中整句照抄证据原文的比例相对 S 组的变化，用来监测参考答案 chosen 带来的离策略文风偏移。
   - **按错误类型**：hard_dev 上 multi（A / B 类）、exception、conflict、no_conflict、partial、refuse 子集的分组结果。
   - **引用**：citation_valid、fact_attribution_ok。因为 H 类偏好对最多，需要确认引用改善没有以内容退步为代价。

## 5. 时间与资源估算（RTX 4060 Laptop 8 GB，本机）

| 步骤 | 估算 |
|---|---|
| SFT（476 条 × 3 epoch ≈ 179 步）× 2 种子 | 每个种子约 1.5～2 小时，参照 v4 的速度 |
| DPO（791 对 × 2 epoch ≈ 198 步，冒烟测速约 48 秒 / 步）× 3（P42、P43、X42） | 每组约 3 小时 |
| hard_dev 检查点评测（DPO 每组 5 个检查点 × 56 题） | 每组约 1 小时 |
| test_v5 生成（125 题 × 9 组） | 约 2～3 小时 |

不租 GPU，所有训练都在本机运行；权重和检查点留在 `<WORKSPACE>`，不提交。

## 6. 会使结论无效、必须如实报告的情况

- 任何组在 test_v5 上生成了两次，或用 test_v5 结果调整了任何设置；
- 正式训练用的数据哈希与记录不一致；
- DPO 日志里没有偏好指标；
- 检查点选择没有按预设顺序进行；
- 人工审核完成前，就把自动评测结果称为“可靠改善”。

## 7. 补充：2026-11-24 第二轮诊断后待确认的事项（提议，尚未生效）

下面这些来自本阶段的诊断。它们都要在人工审核结果出来**之前**由用户确认，确认后写入第 3.1 节和 `freeze.json`。现在只是提议。

| # | 提议 | 依据 |
|---|---|---|
| 7.1 | 训练前排除 [prompt + 回答] 超过 1024 token 的 15 对（10 道题） | `TRUNCATION_AUDIT.md`：截断会让 chosen 和 rejected 看到不同的系统提示，DPO 的前提被破坏；排除后最长样本才放得进显存 |
| 7.2 | 修订 3 `use_logits_to_keep=True` 维持 proposed，补充限定“前向等价、梯度在 bf16 舍入量级内近似等价” | `DPO_LOGITS_EQUIVALENCE_REPORT.md` |
| 7.3 | 正式 DPO 前，在 SFT 适配器上用同一批最坏情况样本复测显存 | `DPO_WORST_CASE_MEMORY_REPORT.md`：余量只有几十 MiB |
| 7.4 | 评分器 4.0.3（非十六进制畸形 ID）是否采纳；采纳后重新构造偏好对（预计排除 19 对） | `SCORER_AMENDMENT_PROPOSAL.md` |
