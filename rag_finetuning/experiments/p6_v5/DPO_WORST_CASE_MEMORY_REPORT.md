# DPO 最坏情况显存测试

脚本：`diagnostics/dpo_worst_case_memory.py`。

结果：
- `diagnostics/dpo_worst_case/*.json`（每次一份）
- 日志 `<WORKSPACE>/runs/p6_v5/logs/diag_worst_*.log`
- 诊断用的适配器副本在 `runs/p6_v5/diag_worst_*`（不用于任何评测）

> 每次测试只跑 1 个优化步（8 对 × 梯度累积 8）：前向、反向、优化器步、保存检查点（`save_steps=1`）。这**不是正式训练**。
> 策略和参考都初始化自 v4 s42，因为正式 SFT 适配器尚未训练。

## 1. 测试集合

由 `diagnostics/truncation_audit.py` 按 TRL 的实际分词选出，见 `diagnostics/truncation/worst_case_sets.json`：

| 集合 | 定义 | 8 对的长度（prompt / chosen / rejected，token） |
|---|---|---|
| longest_seq_8 | [prompt + 回答] 最长的 8 对 | 832/223/386、844/220/254、728/156/367 … 全部超过 1024，会被截断 |
| longest_completion_8 | 回答 token 最长的 8 对 | 832/223/386、728/156/367、703/144/344 … 其中 6 对会被截断 |
| longest_untruncated_8 | 不会被截断（两侧都 ≤ 1024）的对中，chosen 与 rejected 序列长度之和最大的 8 对 | 781/238/177、798/182/188、734/208/242 … |
| longest_completion_untruncated_8 | 不会被截断的对中，回答 token 之和最长的 8 对 | 687/168/321、734/208/242、676/198/247 … |

后两组对应的是 `TRUNCATION_AUDIT.md` 提议的“排除 15 对超长样本”之后的数据。

## 2. 结果

共 14 次，失败的全部保留：

| 集合 | max_length | 设置 | 结果 | 峰值已分配 / 已保留（MiB） | 检查点保存 |
|---|---|---|---|---|---|
| longest_seq_8 | 1024 | 修订 3 提议配置（`use_logits_to_keep`） | ❌ OOM（申请 214 MiB） | 5680 / 5890 | — |
| longest_seq_8 | 1024 | + `expandable_segments` | ❌ OOM（申请 214 MiB） | 5680 / 5890 | — |
| longest_seq_8 | 1024 | + 预计算参考 logp | ❌ OOM（申请 450 MiB） | 5278 / 6034 | — |
| longest_seq_8 | 1024 | + 两者都开 | ❌ OOM（申请 450 MiB） | 5278 / 6034 | — |
| longest_seq_8 | 1024 | 进程上限 0.74 → 0.80（只作诊断） | ❌ OOM（申请 594 MiB） | 5792 / 6106 | — |
| longest_seq_8 | 1280 | 修订 3 提议配置 | ❌ OOM（申请 594 MiB） | 5510 / 6058 | — |
| longest_seq_8 | **896** | 修订 3 提议配置 | ✅ 完成 | 5674 / **6056** | ✅ policy + reference |
| longest_completion_8 | 1024 | 修订 3 提议配置 | ❌ OOM（申请 200 MiB） | 5485 / 6048 | — |
| longest_completion_8 | 1024 | + `expandable_segments` | ❌ OOM | 5485 / 6048 | — |
| longest_completion_8 | 1024 | + 预计算参考 logp | ❌ OOM（申请 428 MiB） | 5215 / 5938 | — |
| longest_completion_8 | 1024 | + 两者都开 | ❌ OOM | 5215 / 5938 | — |
| **longest_untruncated_8** | 1024 | 修订 3 提议配置 | ✅ 完成，71 秒 | 5561 / **6042** | ✅ |
| longest_untruncated_8 | 1024 | + `expandable_segments` | ✅ 完成 | 5561 / 6042 | ✅ |
| **longest_completion_untruncated_8** | 1024 | 修订 3 提议配置 | ✅ 完成，72 秒 | 5693 / **6058** | ✅ |

- **测试环境**：上限 `P6_MEM_FRACTION=0.74`，即 6059 MiB；GPU 总显存 8188 MiB；其他进程约占 1.5～2 GB。
- **成功的运行**：日志中都有 `rewards/chosen`、`rewards/rejected`、`rewards/margins`；保存检查点后显存回到约 2718 MiB；保存检查点本身没有增加峰值。
- **0.80 那一次的记录有误**：记录里的 `limit_mib` 仍写着 6059，因为脚本按 0.74 写死了，实际上限约 6550 MiB。它失败时 GPU 剩余空闲显存已不足，说明瓶颈已经是整块显卡，而不只是进程上限。
- **第一轮的 3 份记录**（`longest_seq_8_1024.json`、`longest_completion_8_1024.json`、`longest_seq_8_1280.json`）是加入 `--precompute-ref` 和 `--tag` 参数之前跑的，`precompute_ref_log_probs` 字段为空，实际都是“未开启”。

## 3. 结论（provisional）

1. **按修订 3 提议的配置（1024 + `use_logits_to_keep`）在全部 791 对上训练，会在最长的样本上 OOM。**
   - 冒烟测试 r4 能通过，只是因为前 80 对比较短。
   - `expandable_segments`、预计算参考 logp、提高进程上限都不能解决这个问题。
2. **排除截断审计中那 15 对超长样本后，1024 能运行。**
   - 剩余样本里最长的两组（按序列长度、按回答长度）都完成了前向、反向、优化器步和检查点保存。
   - 但已保留显存分别是 6042 和 6058 MiB，离上限只有 17 MiB 和 **1 MiB**。
3. **降到 896（协议允许）也能运行**，余量同样只有 3 MiB。
   - 而且 896 会截断 48 对，全部是 chosen 和 rejected 看到不同的 prompt，其中 1 对截到了证据。
   - 截断问题更严重，不建议采用。
4. **余量这么小，不能认为“正式训练一定不会 OOM”。** 以下情况都可能导致 OOM：
   - 其他程序多占一些显存：浏览器、桌面、用户的 Ollama 托盘应用加载了模型；
   - 显存碎片的分布不同；
   - 训练多步后，优化器状态或分页内存的行为变化。

## 4. 建议（待确认，本阶段不改配置）

- **数据**：采纳 `TRUNCATION_AUDIT.md` 选项 A，在训练前排除 15 对超长样本。它同时消除截断缺陷和最坏情况下的 OOM。
- **运行环境**：正式训练期间关闭其他占用显存的程序。训练开始前记录 `nvidia-smi`，要求其他进程的占用 ≤ 1.6 GB。这是工作流要求，不是协议修改。
- **复测**：正式 SFT 适配器训练完成后，在 S42 适配器上重新跑 `longest_untruncated_8` 和 `longest_completion_untruncated_8` 两组。
  - LoRA 结构相同，预计结果相同，但需要实测。
- **如果复测仍然 OOM**：剩下的办法都会改变协议或数值行为，必须先写修订、再训练，不能临时调整。例如：
  - 不把词嵌入层升为 fp32（与 SFT 配方不一致）；
  - 把梯度累积中的单条样本按 token 预算分组；
  - 进一步按长度排除样本。
- 修订 3 继续保持 proposed。
