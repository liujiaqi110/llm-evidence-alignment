# DPO 训练可行性复测（S42 / S43 正式 SFT 适配器，2026-11-25）

> **状态：provisional。** 人工审核未完成的协议偏离继续保留。
> 这次只做可行性复测：未启动正式 P42 / P43，未启动 X42，未运行 test_v5，未推送。
> 测试中生成的临时权重在 `<WORKSPACE>/runs/p6_v5/diag_feasibility_s4{2,3}/`，已标记 `DO_NOT_USE.txt`，不得用于正式训练或评测。

- 脚本：`diagnostics/dpo_feasibility_v5.py`
- 结果文件：`diagnostics/dpo_feasibility/precheck.json`、`memory_s42.json`、`memory_s43.json`
- 日志：`runs/p6_v5/logs/feas_s4{2,3}.log`

## 1. 输入核对

| 项目 | 结果 |
|---|---|
| P42 初始化 / reference | `sft_3b_s42/adapter`，即预注册规则选中的 epoch3，sha256 `09668483…`，与 `checkpoint_epoch3` 相同 |
| P43 初始化 / reference | `sft_3b_s43/adapter`，即 epoch3，sha256 `436440d6…`，与 `checkpoint_epoch3` 相同 |
| 偏好数据 | `dpo_pairs_final.jsonl`，580 对，sha256 `61804f66…`，与 freeze.json 记录一致 |
| DPO 超参数 | 冻结值不变：sigmoid、β 0.1、lr 2e-5 cosine、warmup 0.1、bs 1 × GA 8、max_length 1024 / max_prompt_length 896、`use_logits_to_keep=True`、paged_adamw_8bit、梯度检查点。`dpo_3b_s42.yaml` 与 `dpo_3b_s43.yaml` 只有种子和初始适配器不同 |
| 环境 | TRL 0.23.1、peft 0.22.0、transformers 4.57.6、torch 2.11.0+cu128；RTX 4060 Laptop 8 GB；`P6_MEM_FRACTION=0.74`（进程上限 6059 MiB） |

## 2. 全量输入预检（580 对，CPU）

预检使用实际训练用的 tokenizer（Qwen2.5-3B-Instruct@aa8e7253）和 chat template，按 TRL 0.23.1 的流程处理：maybe_extract_prompt → maybe_apply_chat_template → DPOTrainer.tokenize_row。

| 检查 | 结果 |
|---|---|
| prompt 超过 896，被左截断 | **0** |
| [prompt + 回答] 超过 1024，chosen / rejected 看到的 prompt 可能不一致 | **0** |
| chosen / rejected 回答缺失、末尾缺 eos、两者相同、解码与原文不一致 | **0** |
| prompt 不以 assistant 生成前缀结尾 | **0** |
| 序列长度（取 chosen / rejected 两侧的较大值） | 中位数 617，p95 893，**最大 1019** |

GPU 测试时还做了一项核对：DPOTrainer 实际生成的 `prompt / chosen / rejected_input_ids` 与预检结果**一致，24/24**（两个种子都是）。

## 3. 最坏情况显存复测（GPU）

数据共 3 个优化步，每步 8 对 × 梯度累积 8：
- 第 1 步：最长序列 8 对（最长的 [prompt + 回答] 为 781 + 238 = 1019 token）；
- 第 2 步：最长回答 8 对（最长回答 254 token）；
- 第 3 步：再跑一次最长序列 8 对。

每一步都执行前向（policy 和 reference）、反向、优化器步和检查点保存（`save_steps=1`）。

| | S42 → P42 | S43 → P43 |
|---|---|---|
| 结果 | ✅ 完成，无 OOM，225.8 s | ✅ 完成，无 OOM，250.4 s |
| 训练前 GPU 已用（nvidia-smi） | 1767 MiB | 1604 MiB |
| 加载后峰值 | 3148 MiB | 3148 MiB |
| 训练峰值：已分配 / 已保留 | 5590 / **5978** MiB | 5590 / **5954** MiB |
| 已保留峰值距进程上限 6059 MiB | **81 MiB** | **105 MiB** |
| 训练中整卡剩余空闲显存（最低） | 506 MiB | 523 MiB |
| 每步结束、保存检查点后的显存 | 回到 2718 MiB，保存检查点没有提高峰值 | 同左 |
| 检查点保存 | checkpoint-1～3，每个都有 policy 和 reference | 同左 |

对照：上一轮测试用 v4 s42 适配器和更早的数据（`DPO_WORST_CASE_MEMORY_REPORT.md`），同类最坏情况的余量只有 17 MiB 和 1 MiB。这次两个种子的余量都是 80～105 MiB。原因是最终数据经过长度控制后，最长回答从 321 token 降到 254 token，整卡上其他进程的占用也略少。

## 4. 适配器冻结与更新核对

| 检查 | S42 | S43 |
|---|---|---|
| 可训练参数张量全部属于 policy | 是（504 个） | 是（504 个） |
| reference 参数是否有 requires_grad | 否 | 否 |
| 训练后 reference 参数的最大绝对变化 | **0.0** | **0.0** |
| 保存的 reference 是否与初始 SFT 适配器逐张量相同 | 3/3 相同 | 3/3 相同 |
| 被更新的 policy 张量 | 504/504 | 504/504 |

## 5. 真实 DPO 日志（只是最坏情况样本上的 3 步，**不代表训练效果**）

| 种子 | 步 | loss | grad_norm | rewards/chosen | rewards/rejected | rewards/margins | rewards/accuracies |
|---|---|---|---|---|---|---|---|
| S42 | 1 | 0.6931 | 10.85 | 0.000 | 0.000 | 0.000 | 0.00 |
| S42 | 2 | 0.6931 | 14.90 | 0.000 | 0.000 | 0.000 | 0.00 |
| S42 | 3 | 0.5335 | 11.32 | 0.016 | −0.354 | 0.370 | 1.00 |
| S43 | 1 | 0.6931 | 12.11 | 0.000 | 0.000 | 0.000 | 0.00 |
| S43 | 2 | 0.6931 | 11.48 | 0.000 | 0.000 | 0.000 | 0.00 |
| S43 | 3 | 0.5103 | 10.61 | 0.005 | −0.420 | 0.425 | 1.00 |

- 第 1、2 步的 loss 都是 ln 2，reward 都是 0。原因是第 1 步处在 warmup，学习率为 0；第 1 步之前 policy 与 reference 完全相同，第 2 步的前向发生在第一次非零更新之前。
- 第 3 步重复第 1 步的数据。这时 policy 已经更新，margin 为正，说明偏好损失、reference 对数概率和梯度路径都在正常工作。
- 日志中 `learning_rate` 一栏全是 0。这是 3 步的极短调度（1 步 warmup 加 cosine 衰减到 0）在记录时间点上的表现，不能据此认为没有更新。实际已更新，证据是 policy 504/504 张量变化、第 3 步 reward 非零。
- 完整日志（含 logps、logits）见 `memory_s4x.json`。

## 6. 是否满足正式训练条件（provisional 判断）

**在本机满足条件，但余量有限，需要附带运行约束。**

满足的方面：
- 580 对全部样本中，最长序列和最长回答都完成了前向、反向、优化器步和检查点保存，两个种子都没有 OOM。
- 输入没有截断，也没有 chosen / rejected 不一致。
- reference 保持冻结，只有 policy 被更新。

余量有限的方面：
- 相对进程上限只剩 81～105 MiB，整卡剩余约 0.5 GB。
- 测试只覆盖 3 步。正式训练每个种子约 145 步（2 epoch × 73 步），碎片随步数的变化没有实测。
- 其他程序（浏览器、桌面应用、Ollama 托盘加载模型等）多占几百 MiB 就可能 OOM。

正式训练的规模和耗时估计：
- 每个种子约 145 个优化步。
- 最坏情况约 75 s/步，平均样本更短，估计每个种子训练 2～3 小时。
- 之后在 hard_dev 上评测每半个 epoch 的检查点并做选择，约 4～5 个检查点 × 10 分钟。

## 7. 下一步建议（待用户决定，本轮不执行）

1. **如果批准本地正式 P42 / P43：**
   - 保持全部冻结配置和 `P6_MEM_FRACTION=0.74`，依次训练，不并行。
   - 训练前关闭占用显存的程序，记录 nvidia-smi，要求其他进程的占用 ≤ 1.6 GB。
   - 这是工作流要求，不改协议。
2. **约定失败处理：** 正式训练中如果 OOM，保留失败运行，不在本地改模型或配方，直接迁移到 A10 24 GB。
3. **A10 24 GB 迁移所需环境（备用，本轮未租用）：**
   - 软件版本与本机一致：Python 3.x、torch 2.11.0（与 A10 驱动匹配的 CUDA 12.x 版本）、transformers 4.57.6、peft 0.22.0、TRL 0.23.1、bitsandbytes 0.50.2；
   - Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1 离线权重；
   - S42 / S43 epoch3 适配器（核对上面的 sha256）；
   - `dpo_pairs_final.jsonl`（sha256 `61804f66…`）、hard_dev（sha256 `8117f2f8…`）、评分器 4.0.3；
   - 先在 A10 上用本脚本重跑 precheck 和 memory，确认结果一致后，再按冻结配置训练。
   - 迁移只换硬件，不改超参数。
