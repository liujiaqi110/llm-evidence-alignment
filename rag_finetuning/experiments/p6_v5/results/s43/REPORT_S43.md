# P6-v5 实验一 S43（针对性 SFT，种子 43）：训练、hard_dev 评测与检查点选择

> **状态：provisional。** 人工审核未完成的协议偏离继续保留。只用 hard_dev（开发集，参考用）。未运行 test_v5，未启动 DPO、X42，未推送。
> 双种子对照见 `../s42_s43/REPORT_S42_S43.md`。

## 1. 运行信息

| 项目 | 值 |
|---|---|
| 代码 | commit `f28ba28`。run_record 中记为 `+dirty`，原因是存在未跟踪文件 `data/train_v5/structured_questions.jsonl`（sha256 `770b8d8d…`，修改时间 2026-11-24 15:19；训练代码不读取它，只有偏好数据构建脚本引用）。该文件已保留，未删除、未提交，记录见 `run/s43_precheck.json` |
| 与 S42 的差异 | 配置中只有 `run_name` 和 `seed`（42 → 43）不同 |
| 初始化 | Qwen/Qwen2.5-3B-Instruct @ `aa8e72537993ba99e69dfaafa59ed015b17504d1`（3B Base），4-bit NF4 + double quant，bf16 |
| 配方 | 与 S42 完全相同：QLoRA r8/α16/dropout 0.05；lr 2e-4 cosine，warmup 0.1；3 epoch；bs 1 × GA 8；max_seq_len 1024；paged_adamw_8bit；只对回答部分计算 loss |
| 训练数据 | `sft_train_final.jsonl`，472 条，sha256 `41c4926f…`（训练前核对，与冻结值一致） |
| 评测集 / 评分器 | hard_dev sha256 `8117f2f8…`；scoring_v4.py sha256 `a230bfe1…`（4.0.3） |
| 环境 | torch 2.11.0+cu128，transformers 4.57.6，peft 0.22.0，bitsandbytes 0.50.2；HF_HUB_OFFLINE=1；P6_MEM_FRACTION=0.74 |
| 硬件 | RTX 4060 Laptop 8 GB。训练前 GPU 已用约 1973 MiB；训练中峰值分配 5391 MiB、预留 6058 MiB |
| 耗时 | 训练 4147 s（177 步）。2026-11-24 22:17 开始，23:54 训练结束，2026-11-25 00:04 统一评测结束 |

## 2. 训练曲线

逐步数据见 `training_curve.svg` 和 `train_log.csv`。

- 训练 loss：第一步 0.625，最后一步 0.015，平均 0.104。
- 梯度范数：0.09～4.64。
- hard_dev loss：训练前 1.037，epoch1 0.350，epoch2 **0.339**，epoch3 0.351。和 S42 一样，epoch3 有回升。

## 3. 检查点选择（预注册规则）

| epoch | multi_complete_v5 | content_ok_v4 | whole_refusal | refusal_ok | conflict_true_ok | hard_dev loss |
|---|---|---|---|---|---|---|
| 1 | 17/43 | 21/56 | 3/52 | 4/4 | 0/4 | 0.350 |
| 2 | 21/43 | 26/56 | 2/52 | 4/4 | 1/4 | 0.339 |
| **3（选中）** | **23/43** | **29/56** | 3/52 | 4/4 | 1/4 | 0.351 |

- 规则复核结果与训练时一致，选中 epoch 3。`adapter/epoch.json` 记录为 epoch 3、step 177。
- 统一评测和选择阶段的回答 **56/56 完全相同**。
- 三个检查点全部保留在 `<WORKSPACE>/runs/p6_v5/sft_3b_s43/`。逐题回答见 `run/dev_eval/`。

## 4. 开发集参考门槛

| 门槛 | 结果 | 是否通过 |
|---|---|---|
| 1 multi_complete_v5 相对 B：Δ ≥ +0.10，且 CI 下界 > 0 | +0.140，CI [0.026, 0.273] | 是 |
| 2 content_ok_v4 相对 C42：Δ ≥ −0.05，且 CI 下界 ≥ −0.10 | −0.054，CI [−0.167, 0.042] | **否** |
| 3～7 | 见 `comparison.json` | 全部通过 |

## 5. 失败分析（`failure_analysis.json`）

- **多证据题：** 43 题中 20 题没有完整回答。原因分布：部分事实缺失 9 题，只用了最后一个金标准片段 6 题，只用了第一个片段 3 题，整体拒答 2 题。
- **内容错误：** 共 27 题。缺失必要事实 24 题，行为错误 2 题，含证据外数字 1 题（dv_m23 中的“2000”）。
- **相对 C42：**
  - 多证据变差 6 题：dv_m21、dv_m22、dv_m24、dv_m08r、dv_x04、dv_x06。
  - 多证据变好 3 题：dv_m26、dv_m27、dv_m27r。
- **其他：** 注入题 0/2；非冲突判断 3/8。

## 6. dv_m13 / dv_m16 敏感性分析（只报告）

剔除这两题后，S43 为多证据 22/41、内容正确 28/54。检查点排序不变，仍为 epoch 3。详见 `sensitivity_dv_m13_m16.json`。
