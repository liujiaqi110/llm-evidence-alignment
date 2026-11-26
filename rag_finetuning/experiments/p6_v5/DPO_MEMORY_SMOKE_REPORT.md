# P6-v5 DPO 显存冒烟测试报告（不是正式结果）

> 冒烟测试只验证 DPO 能否在本机 8 GB 显卡上运行、偏好损失是否真的在起作用。
> - 只跑 10 个优化步（80 对），不保存适配器，不做检查点选择，不评测任何模型；
> - 策略和参考都从 **v4 s42 适配器**初始化，因为正式 SFT 还没有训练；
> - 结果不代表 P42 / P43 的效果。

## 1. 环境

| 项目 | 值 |
|---|---|
| GPU | RTX 4060 Laptop 8 GB（8188 MiB），Windows WDDM |
| 其他进程占用 | 测试前约 2034～2044 MiB（桌面与其他程序） |
| 进程显存上限 | `P6_MEM_FRACTION=0.74`，即 5.92 GiB。WDDM 下不设上限会溢出到共享内存，与 v3 / v4 训练一致 |
| 软件 | trl 0.23.1、peft 0.22.0、transformers 4.57.6、torch 2.11.0+cu128、bitsandbytes 0.50.2 |
| 模型 | Qwen2.5-3B-Instruct@aa8e7253，4-bit NF4 + double quant，bf16 计算；`prepare_model_for_kbit_training` + 梯度检查点（与 SFT 相同） |
| 参考模型策略 | 不加载第二份基座。同一基座上挂两个 LoRA 适配器：`policy`（可训练）、`reference`（冻结，`ref_adapter_name`），TRL 计算参考 logp 时切换到 `reference` 并关闭梯度 |
| 可训练参数 | 14,966,784（LoRA r8，7 个线性层） |
| 超参数 | sigmoid loss，β 0.1，lr 2e-5 cosine（warmup 0.1），批 1 × 梯度累积 8，max_prompt_length 896，paged_adamw_8bit，bf16 |
| 数据 | `dpo_pairs.jsonl`（sha256 `204b40b2…`），随机打乱（种子 42）后取前 80 对 |
| 初始化适配器 | v4 s42，sha256 `3f58fa75…` |

偏好对的 token 长度（prompt + 较长的回答）：中位数 640，95 分位 945，最大 1238。

| 长度上限 | 会被截断的对 |
|---|---|
| 1024 | 18 / 791（2.3%） |
| 768 | 196 / 791（24.8%） |

## 2. 全部尝试（没有一次被删除或覆盖）

| 次序 | 设置 | 结果 | 峰值显存（已分配 / 已保留） | 记录 |
|---|---|---|---|---|
| r1a | 1024 | ❌ 模型加载阶段 `KeyError: attribute 'train' already exists`：适配器名 `train` 与 `nn.Module.train` 冲突，是代码错误，不是显存问题 | — | `logs/dpo_smoke_1024.log` |
| r1b | 768 | ❌ 同上 | — | `logs/dpo_smoke_768.log` |
| — | 修复：适配器改名 `policy` | | | `scripts/train_dpo_v5.py` |
| r2 | 1024 | ❌ 第 1 步 OOM（申请 582 MiB 失败） | 5350 / 5518 MiB | `dpo_smoke_from_v4s42_smoke_r2_1024_oom/` |
| r3 | 768 | ❌ 第 1 步 OOM（同样申请 582 MiB） | 5350 / 5518 MiB | `…_smoke_r3_768/` |
| r4 | 1024 + `use_logits_to_keep` | ✅ 10 步完成，481.5 秒（约 48 秒 / 步） | **5577 / 5704 MiB** | `…_smoke_r4_1024_ltk/` |
| r6 | 1024 + `precompute_ref_log_probs` | ❌ 参考 logp 预计算完成（80 对，约 2.5 分钟），第 1 个训练步 OOM（申请 376 MiB 失败） | 5305 / 6058 MiB | `…_smoke_r6_1024_pre/` |

（r5 是 r4 失败时的备用组合，r4 已通过，所以没有运行。）

模型加载后显存峰值为 3148 MiB。`prepare_model_for_kbit_training` 会把非量化参数（含约 3 亿参数的词嵌入层）升为 fp32，这部分开销较大。SFT 脚本用的是同样的预处理，DPO 也保持一致。

## 3. 失败原因分析

- 768 和 1024 在第 1 步的失败位置、申请大小完全相同，说明瓶颈**不在序列长度上限**。
  - 前 80 对里大多数本来就短于 768。
  - 真正的瓶颈是完整词表（151,936）上的 logits：chosen 和 rejected 拼成一批，一次前向传播要按全词表算出所有位置的 logits。
- 预计算参考 logp 只省掉训练时参考模型那次前向传播，策略模型自己的全词表 logits 和反向传播仍然放不下。
- `use_logits_to_keep` 只计算回答部分的 logits，prompt 平均约 510 token，回答约 110～120 token，所以显存明显下降。
  - DPO 损失本来就只用回答 token 的对数概率，因此**损失定义不变**。

## 4. 偏好损失确实在起作用（r4 日志）

| 步 | loss | rewards/chosen | rewards/rejected | rewards/margins | rewards/accuracies |
|---|---|---|---|---|---|
| 1 | 0.6931 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 0.6931 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | 0.6089 | −0.041 | −0.237 | 0.196 | 0.875 |
| 8 | 0.7131 | −0.477 | −1.154 | 0.677 | 0.875 |
| 9 | 0.4119 | −0.146 | −1.234 | 1.088 | 0.750 |
| 10 | 0.7090 | −0.521 | −0.919 | 0.398 | 0.875 |

- 第 1、2 步 loss = ln2 = 0.6931、margins = 0：策略和参考在训练开始时完全相同。cosine warmup 的学习率从 0 开始，第 1 步的更新量为 0，所以第 2 步时策略仍与参考相同。这说明参考适配器设置正确。
- 之后 margins 为正、accuracies 0.75～0.875：日志里有 `rewards/chosen`、`rewards/rejected`、`rewards/margins`，满足 claim_rules 中“真正以偏好损失训练”的条件。
- 只有 10 步、80 对，loss 波动很大。**这不能说明 DPO 会改善评测指标。**

## 5. 对正式训练的影响

- 协议（freeze.json）原本只允许“调整序列长度与预计算参考 logp”。冒烟测试表明，这两种办法在本机都不够。
- 已在 `freeze.json → amendments[2]` 写入**修订 3 提议**：DPO 增加 `use_logits_to_keep=True`，其他超参数不变。
  - 状态为 `proposed`，在用户批准正式训练时一并确认，未批准前不生效。
  - 门槛和评测集哈希都不变。
- 按 r4 的速度：正式 DPO（791 对 × 2 epoch ≈ 198 步）每组约 2.6～3 小时，不含 hard_dev 检查点评测。
- 显存余量：r4 已保留 5704 MiB，上限约 6060 MiB，余量约 350 MiB。正式数据里最长的对有 1238 token（截断到 1024），比冒烟用的前 80 对（最长 1053）更长，**仍有 OOM 风险**。
  - 正式训练前建议先单独跑最长的 8 对作为最坏情况检查。
  - 如果 OOM，按协议降低序列长度，但长度上限必须在训练前确定。

## 6. 未验证的事项

- 正式 SFT 适配器（S42 / S43）还没训练，所以没有用它们做冒烟测试。它们的 LoRA 结构与 v4 相同，显存预计相同，但未验证。
- 没有测试保存检查点时的显存。`save_only_model=True` 只保存适配器，预计开销很小。
- 没有测试最长样本的最坏情况。
