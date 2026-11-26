"""用本机 qwen2.5:7b 起草 P6-v3 训练 / 开发语料（虚构公司“示例科技”的制度文档）。

    python -m rag_finetuning.experiments.p6_v3.scripts.gen_corpus --split train
    python -m rag_finetuning.experiments.p6_v3.scripts.gen_corpus --split dev

* 文档就是“事实来源”：问答的真值由文档定义，所以起草文档不引入事实错误；但文档内部是否自洽、措辞是否清楚，
  由 check_doc 做格式检查，最终仍需人工抽检（generation_method = llm_drafted_doc）。
* 主题列表与 P6-v1 / P6-v2 / P6-v3 测试集主题均不重复（见 TOPICS 与 test 主题清单）。
* 温度 0、固定 seed，可重复；已存在的文档不会被覆盖。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rag_finetuning.compat.local_llm import OllamaClient

HERE = Path(__file__).resolve().parents[1]

TOPICS = {
    "train": [
        ("meeting_minutes", "会议纪要管理规定"), ("referral_bonus", "员工内部推荐奖励办法"),
        ("office_supplies", "办公用品领用与采购规定"), ("printing", "打印与复印管理规定"),
        ("email_usage", "企业邮箱使用规范"), ("software_license", "软件许可证管理办法"),
        ("mobile_device", "移动设备接入管理规定"), ("guest_wifi", "访客网络接入规定"),
        ("health_checkup", "员工年度健康体检管理办法"), ("team_building", "团队建设活动经费管理办法"),
        ("holiday_duty", "节假日值班管理规定"), ("job_transfer", "员工内部调岗管理办法"),
        ("performance_appeal", "绩效结果申诉办法"), ("non_compete", "竞业限制管理规定"),
        ("patent_reward", "专利申报与奖励办法"), ("open_source", "开源软件使用与合规规范"),
        ("db_change", "生产数据库变更操作规范"), ("log_retention", "系统日志保留与查询规定"),
        ("alert_levels", "监控告警分级与处理规范"), ("dr_drill", "灾备切换演练管理办法"),
        ("complaint_escalation", "客户投诉升级处理流程"), ("refund_approval", "客户退款审批规定"),
        ("invoice_issuing", "对外开具发票管理办法"), ("contract_payment", "合同付款审批流程"),
        ("budget_adjust", "年度预算调整管理办法"), ("fx_reimburse", "外币费用报销规定"),
        ("expat_allowance", "外派人员补贴管理办法"), ("trainer_incentive", "内部讲师激励办法"),
        ("personnel_files", "员工档案管理规定"), ("resign_handover", "离职工作交接规定"),
        ("product_launch", "新产品发布评审流程"), ("seal_usage", "印章使用登记规定"),
    ],
    "dev": [
        ("doc_archiving", "文件归档管理规定"), ("meal_allowance", "员工餐补发放办法"),
        ("parking", "停车位申请与管理规定"), ("forum_posting", "内部论坛发帖规范"),
        ("test_device_loan", "测试设备借用管理规定"), ("annual_party", "年会组织管理办法"),
    ],
}

SYSTEM = """你是企业制度文档撰写人。请为一家虚构公司“示例科技”撰写一份中文内部制度，用 Markdown 输出，严格遵守：
1. 第一行是一级标题“# <制度名称>”；第二行写“本文为虚构公司“示例科技”的原创合成制度，仅用于模型训练实验，不含任何真实信息。”
2. 正文恰好 6 个二级标题“## <小节名>”，每个小节 2–4 句话。
3. 每个小节都要有具体、可核对的规则：明确的数字（金额、天数、次数、比例、时间）、负责审批或执行的角色、适用条件，至少两个小节写出例外情况或特殊条件。
4. 各小节之间不得互相矛盾；不要写“视情况而定”“另行通知”之类的模糊表述；不要出现真实公司、真实人名或真实法规条款号。
5. 只输出文档本身，不要任何解释。"""


def check_doc(text: str) -> list:
    errs = []
    if not text.startswith("# "):
        errs.append("no_h1")
    n = len(re.findall(r"(?m)^## ", text))
    if n < 5 or n > 7:
        errs.append(f"sections={n}")
    for block in re.split(r"(?m)^## ", text)[1:]:
        body = block.partition("\n")[2].strip()
        if len(body) < 40:
            errs.append("short_section")
        if not re.search(r"\d", body):
            errs.append("section_without_number")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(TOPICS))
    a = ap.parse_args(argv)
    out = HERE / "data" / "corpus" / a.split
    out.mkdir(parents=True, exist_ok=True)
    c = OllamaClient(model="qwen2.5:7b", timeout_s=600)
    info = c.model_info()
    log = []
    for stem, title in TOPICS[a.split]:
        p = out / f"{stem}.md"
        if p.exists():
            continue
        text, errs = "", ["not_run"]
        for attempt in range(3):
            g = c.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"制度名称：{title}"}],
                       {"temperature": 0 if attempt == 0 else 0.3, "seed": 42 + attempt, "num_predict": 1400, "num_ctx": 4096})
            text = g.text.strip().strip("`").removeprefix("markdown").strip()
            errs = check_doc(text)
            if not errs:
                break
        p.write_text(text + "\n", encoding="utf-8", newline="\n")
        log.append({"doc": stem, "title": title, "attempts": attempt + 1, "format_errors": errs})
        print(stem, attempt + 1, errs, flush=True)
    (out / "_generation_log.json").write_text(json.dumps({"model": info, "system_prompt": SYSTEM, "docs": log},
                                                         ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
