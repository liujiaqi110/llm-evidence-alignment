# P6-v3 训练配置与实际运行记录

**配方说明**

训练配方与 P6-v1 的 `qlora_r8.yaml` 完全相同，只有数据和基础模型不同。这样，1.5B 上 B 组与 C 组之间的差异只来自数据。

**配置与记录位置**

- 配置文件：`configs/*.yaml`，冻结哈希见 `configs/freeze.json`。
- 运行记录：`<WORKSPACE>\runs\p6_v3\<run>\run_record.json`，摘要已收入 `results/training_runs.json`。

**代码版本**

- 训练代码与数据为提交 `79493af`。
- 运行记录里的 `+dirty` 表示训练时 `rag_finetuning/` 下还有未提交的文件，即本文档、数据卡、LLM 评审脚本和单元测试。这些文件都不参与训练。

## 1. 公共配置

| 项 | 值 |
|---|---|
| 量化 | bitsandbytes 0.50.2，4-bit NF4 + double quant，bf16 计算 |
| LoRA | r = 8，alpha = 16，dropout = 0.05；target：q / k / v / o / gate / up / down |
| 优化器 | paged_adamw_8bit，lr 2e-4，cosine，warmup 10%，weight decay 0，梯度裁剪 1.0 |
| Batch | per-device 1 × 梯度累积 8（等效 batch = 8） |
| 序列 | max_seq_len 1024，实测训练样本 595–881 token，不截断 |
| Loss | 只对 assistant 回答 + `<|im_end|>` 计算，已在 v1 审查中逐条验证 |
| Padding | 右侧 padding 到 128 的倍数，padding 位置 label = −100、attention = 0 |
| 显存优化 | gradient checkpointing（非 reentrant），不打包样本 |
| 轮数 | 最多 3 个 epoch，每个 epoch 存一次 Adapter |
| 停止点 | 在 dev_v3（31 条）上生成并用评分器 v2 打分。规则：content_ok 最高 → 错误拒答最少 → 更早的 epoch。测试集不参与 |
| 随机种子 | 42（`set_seed` 与 `data_seed`） |
| 软件 | torch 2.11.0+cu128、transformers 4.57.6、peft 0.22.0、Windows 原生 CUDA，RTX 4060 Laptop 8 GB |

## 2. 实际运行

| Run | 基础模型（Revision） | 数据（sha256 前 16 位） | 样本 | 步数 | 首步 / 末步 loss | 训练 loss 均值 | dev loss（训练前 → 每个 epoch） | dev content_ok（epoch 1 / 2 / 3） | 选定 epoch | 梯度范数范围 | 显存峰值 allocated / reserved | 耗时 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **v3_1p5b_full（C 组）** | Qwen2.5-1.5B-Instruct `989aa79…` | full `d6f93b67…` | 315 | 120 | 1.161 / 0.287 | 0.170 | 0.788 → 0.087 → 0.069 → 0.070 | 26 / 26 / 26（错误拒答 1 / 0 / 0） | **2** | 0.17–3.17 | 3814 / 5430 MiB | 709 s |
| **v3_3b_full（E 组）** | Qwen2.5-3B-Instruct `aa8e725…` | full `d6f93b67…` | 315 | 120 | 1.783 / 0.221 | 0.163 | 1.202 → 0.076 → 0.060 → 0.058 | 27 / 27 / 26（错误拒答 0 / 0 / 0） | **1** | 0.09–8.74（第 2 步尖峰，之后 < 2） | 5131 / 6762 MiB | 1263 s |
| abl_1p5b_*（4 个消融） | 同 C | 各变体 | 见 DATASET_CARD | 3 个 epoch 调度（与 C 相同） | | | | | 固定 2（C 的选定值） | | | 见 `results/training_runs.json` |

**3B 冒烟测试（训练前）**

- 前向 / 反向，最长样本 886 token：loss 0.978 为有限值；252 个 LoRA-B 有梯度（36 层 × 7），底座参数没有梯度；显存峰值 allocated 5.0 GB、reserved 5.9 GB。
- 10 步训练冒烟：loss 从 0.97 降到 0.34；显存峰值 5.1 / 6.75 GB；每步约 10 秒。

**可训练参数与 Adapter**

| | 可训练参数 | Adapter 大小 | 存放位置（不进入 Git） |
|---|---|---|---|
| 1.5B | 9.23M | 36 MB | `<WORKSPACE>\runs\p6_v3\<run>\adapter` |
| 3B | 14.97M | 58 MB | 同上 |

**失败与中止记录（保留，不删除）**

`ABORTED_v3_1p5b_full_data1.0_step3`：AI 抽检发现 data 1.0 中有 24% 的陈述是残句，于是在第 3 步主动中止，修正数据后用 1.1 重新训练。该次运行未产生 Adapter，也未接触测试集。

`ABORTED_abl_1p5b_no_partial_2epoch_schedule`：消融第一次启动时误用了 `--epochs 2`，余弦调度与 C 不同，会引入学习率调度这一混杂因素；在第 2 步中止，改为 3 个 epoch 调度并固定取 epoch 2 checkpoint 后重新训练。

## 3. 观察

- **dev loss 降得很低**（约 0.06–0.07），但 dev content_ok 在各 epoch 之间几乎不变（26–27/31）。原因是 dev_v3 与训练数据同出一条 7B 起草流水线、风格高度相似，dev 对停止点的区分力有限。所以停止点选择主要由并列规则决定，这一点会作为局限写进报告。
- **训练是否可行**：3B 在 8 GB 显存上可以稳定训练。reserved 峰值 6.8 GB，接近上限，但没有溢出到共享内存，每步约 10.5 秒。
