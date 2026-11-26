# `llm-evidence-alignment` 发布检查清单（草案；本阶段不发布）

每一项都要在**导出后的新目录**中检查，不在原仓库中检查。勾选人和日期由执行发布的人填写。AI 自动检查的结果只能作为参考，不能代替这里的确认。

## A. 用户决策（发布前必须先定）

- [ ] 仓库名：`llm-evidence-alignment`（已拟定）
- [ ] 许可证：Apache-2.0 / MIT / 其他（原仓库没有 LICENSE）
- [ ] Qwen2.5-3B-Instruct 许可证原文已核对：允许公开发布它的输出（候选回答、偏好对、评测输出），并写明了需要附带的声明
- [ ] Qwen2.5-7B（Ollama qwen2.5:7b）许可证已核对
- [ ] test_v5 的公开时间：独立评测完成后，单独发布一次
- [ ] 人工审核结果是否随首次发布公开；如果不公开，README 要写明“审核未完成，所有结果 provisional”
- [ ] 公开范围：私有 / 公开

## B. 不得包含的内容

用 `scripts/check_release.py` 自动检查；失败即阻止发布。

- [ ] 没有模型权重 / 适配器 / 检查点：`*.safetensors`、`*.bin`、`*.pt`、`*.pth`、`*.gguf`、`*.ckpt`、`adapter_model*`、`optimizer.pt`
- [ ] 没有 Hugging Face / Ollama 缓存，没有虚拟环境（`.venv/`、`venv/`），没有 `__pycache__`、数据库文件
- [ ] 没有 test_v5 标注：`data/test_v5/built/`、`data/test_v5/*.yaml`、`data/test_v5/corpus/`、`data/frozen/test_v5.jsonl`。只保留 `test_v5.manifest.json`
- [ ] 没有 test_v5 逐题结果（含参考答案和必要事实）：`results/test_v5/per_question.csv`、`results/test_v5/per_fact.csv`、`results/test_v5/generations_*.jsonl`、`final/per_item_test_v5.csv`、`final/failure_data.json`。最终报告只引用题号
- [ ] 全文检索 test_v5 的 125 道题题面和参考答案片段，命中 0 处
- [ ] 没有密钥 / Token / 密码 / Cookie / 真实的 `.env`：gitleaks 或同类工具扫描为 0，并人工复核扫描报告
- [ ] 没有个人信息：邮箱、Windows 用户名、用户主目录路径、真实姓名
- [ ] 没有本地绝对路径：Windows 盘符开头的本地绝对路径，全部改为环境变量或 `<WORKSPACE>`
- [ ] 没有私有研究仓库 `<PRIVATE_REPO>` 的任何内容或引用
- [ ] 没有 P1～P5 平台的代码。例外是 `compat/` 中原样复制的 `rag_pipeline.py`、`local_llm.py`，并已附来源说明
- [ ] 单个文件不超过 10 MB；整个仓库不超过 50 MB
- [ ] 没有 `data/train_v5/structured_questions.jsonl`（未跟踪的重复副本）

## C. 必须包含的内容

- [ ] README.md / README_CN.md：
  - 研究问题；
  - 各版本结论，负面结果写在前面；
  - 局限：自动评测、人工审核未完成；
  - 复现方法：CPU 重算、GPU 训练。
- [ ] 所有预注册协议、`freeze.json` 与修订记录：4.0.1、4.0.2；4.0.3 和修订 3 标为“提议”
- [ ] 所有负面实验与失败记录：
  - Planning v1 / v2 / v2.1；
  - v4 对 v3 的比较；
  - 旧 7B 草稿的审查结果；
  - v3 / v4 训练中的 OOM；
  - DPO 冒烟测试 r1～r6；
  - 最坏情况显存测试、等价性测试。
- [ ] 数据排除历史：`excluded.jsonl`、`assemble_report*.json`、v1.0 存档、`pairs_report.json`
- [ ] 人工审核材料：`HUMAN_REVIEW_DASHBOARD.html`、`HUMAN_REVIEW_SAMPLING_REPORT.md`、各审计 CSV，人工字段为空或保留真实的审核记录
- [ ] `reports/PROVENANCE.md`：每个文件的来源分支和 commit
- [ ] `reports/PATH_REDACTION.md`：路径替换规则
- [ ] 每个模型的 run_record 与适配器 sha256（权重本身不发布）
- [ ] 第三方许可证说明：`THIRD_PARTY_NOTICES.md`

## D. 一致性检查

- [ ] 83 项 P6 单元测试全部通过：纯 CPU，Python 3.12 + PyYAML
- [ ] 冻结哈希与 `freeze.json` 一致：hard_dev `3b79e057…`、test_v5 `396150ea…`（manifest）
- [ ] 训练数据哈希与 `pairs_report.json` 一致：`dpo_pairs.jsonl` `204b40b2…`、`sft_train.jsonl` `d60e7e9c…`
- [ ] 评分器 `scoring_v4.py` 的哈希与修订 2 记录一致：`9c56176a…`
- [ ] 用公开的模型输出重算 hard_dev 所有组的指标，与 `results/hard_dev_scorer402/summary.json` 完全一致
- [ ] 报告中的每个数字都能追到某个结果文件（抽查至少 20 处）
- [ ] 所有写着“人工审核”的地方确实有人工记录；AI 预审和程序判定都标明“非人工”

## E. 仓库与历史

- [ ] 新目录使用全新的 `git init`，不复制旧 `.git`
- [ ] 原仓库 `rag-pipeline` 的 main、各分支、remote 配置都没有变化。发布前后各执行一次 `git -C <SOURCE_REPO> remote -v` 和 `git branch -a`，结果相同
- [ ] `.gitignore`：`workspace/`、权重扩展名、缓存、`.venv/`、`.env`
- [ ] CI 只运行 CPU 测试和发布检查，不需要密钥（`permissions: contents: read`）
- [ ] 首次提交前，用户审阅导出文件清单 `git ls-files`
- [ ] 推送前用户再次确认。远程仓库已存在时不覆盖、不强推

## F. 发布后

- [ ] 在 GitHub 页面确认文件清单、README 显示、CI 通过
- [ ] 克隆到一个干净目录，按 README 运行 CPU 复现
- [ ] 记录发布的 commit、日期和检查人
