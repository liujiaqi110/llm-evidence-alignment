"""构建 data_v4 = data_v3 full（315 条，原样）+ 程序化结构样本（由参数表生成文档与问答，事实由构造保证）。

    python -m rag_finetuning.experiments.p6_v4.scripts.build_v4

每个虚构主题有一张参数表（审批角色、提前天数、金额阈值、期限、次数、例外条件、费用、违规处理），渲染为制度文档，
并按需渲染修订通知 / 补充规定 / 条件通知 / 未注明日期的部门通知。由同一参数表推出以下样本（每题考查不同事实，不改写凑数）：
  multi2 两片段联合 | multi3 三片段联合 | exception 一般规则 + 例外 | threshold_trap 金额超阈值须加签（附 hard_negative_answer：
  看似完整但遗漏必要条件的错误答案，仅存档，不参与 SFT）| override 新规覆盖旧规 | old_still 旧规仍适用于未覆盖部分 |
  scope_in / scope_out 不同部门 | cond_in / cond_out 条件不同不矛盾 | conflict 同时有效且无先后的真冲突 |
  same_conclusion 取值不同但结论一致 | partial 证据只支持部分 | refusal 证据中确无该信息（同文档难负例，由参数表保证缺失）
所有新样本 generation_method = programmatic_structured，review_status = provisional，并须通过评分器 v3 自检。
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from rag_finetuning.common import REFUSAL_TEXT, chunk_document, read_jsonl, write_jsonl
from rag_finetuning.experiments.p6_v4.scripts.scoring_v3 import score_item_v3
from rag_finetuning.scripts.build_dataset import jaccard

V4 = Path(__file__).resolve().parents[1]
V3 = V4.parent / "p6_v3"
VERSION = "p6v4-data-1.0"
TOPICS = {
    "train": [("prototype_loan", "样机借用", "样机"), ("meeting_recording", "会议录音转写", "录音转写"), ("plant_care", "办公绿植维护", "绿植摆放"),
              ("parcel_service", "快递收发", "寄件"), ("library", "内部图书借阅", "图书"), ("gym", "员工健身房使用", "健身房预约"),
              ("shuttle_bus", "通勤班车乘坐", "班车座位"), ("locker", "储物柜使用", "储物柜"), ("whiteboard_supply", "白板耗材领用", "耗材"),
              ("monitor_loan", "显示器借用", "显示器"), ("tea_break", "会议茶歇", "茶歇"), ("tech_talk", "技术分享会", "分享会场地"),
              ("patent_search", "专利检索服务", "检索委托"), ("translation", "翻译服务", "翻译委托"), ("visa_help", "出差签证协助", "签证协助"),
              ("recheck", "体检复查", "复查预约"), ("counseling", "心理咨询服务", "咨询预约"), ("car_share", "车位共享", "共享车位"),
              ("training_room", "培训教室使用", "教室"), ("screen_tool", "录屏软件授权", "录屏授权"), ("test_account", "测试账号申请", "测试账号"),
              ("cloud_burst", "云服务器临时扩容", "临时扩容"), ("domain", "内部域名申请", "域名"), ("ssl_cert", "SSL 证书申请", "证书"),
              ("sms_template", "短信模板报备", "短信模板"), ("dashboard", "数据看板权限", "看板权限"), ("survey", "问卷发布", "问卷"),
              ("wechat_post", "公众号投稿", "投稿"), ("gift_purchase", "礼品采购", "礼品"), ("business_card", "名片印制", "名片"),
              ("uniform", "工服领用", "工服"), ("expo_material", "展会物料", "展会物料"), ("sample_shipping", "样品寄送", "样品"),
              ("drone", "航拍设备借用", "航拍设备"), ("mic_kit", "直播设备借用", "直播设备"), ("lab_bench", "实验台预约", "实验台")],
    "dev": [("pantry", "茶水间管理", "茶水间物资"), ("live_stream", "会议直播", "直播"), ("icon_design", "图标设计委托", "设计委托"),
            ("mirror_repo", "内网镜像源申请", "镜像源"), ("printer_fix", "打印机报修", "报修"), ("conf_call", "电话会议线路", "会议线路"),
            ("office_suite", "办公软件授权", "软件授权"), ("badge_photo", "证件照拍摄", "拍摄预约")],
}
ROLES = ["部门负责人", "行政部经理", "财务经理", "IT 主管", "人力资源部经理", "项目负责人", "安全组负责人", "运营总监"]
DEPTS = ["研发中心", "销售部", "客服中心", "财务部", "市场部", "华北分公司", "海外事业部", "设计部"]
CONDS = ["每年 12 月年度盘点期间", "法定节假日期间", "新品发布周", "每季度最后一周", "系统升级期间"]
EXC = ["因客户紧急需求申请", "跨部门联合项目申请", "外地办公室员工申请", "新员工入职首月申请", "高级别会议使用"]
PUN = [("书面警告", "暂停使用资格 1 个月"), ("口头提醒", "通报批评"), ("书面警告", "取消当年使用资格")]
UNSTATED = [("是否需要填写纸质登记表", "纸质登记表"), ("费用能否开具发票", "开具发票"), ("能否委托他人代办", "委托他人代办"),
            ("由哪个系统提交", "提交系统"), ("节假日是否照常受理", "节假日受理")]
Q2 = ["{s}要提前多久申请、找谁审批？每次能用多久？", "申请{s}需要提前几个工作日、由谁批？一次的期限是多少天？",
      "我想申请{s}，要提前多久、谁来审批？能用几天？"]
QX = ["{e}的{s}，使用期限是多少？需要谁同意？", "如果是{e}，{s}能用多久？要谁点头？", "{e}时，{s}期限有没有特殊规定？",
      "{e}的话，{s}最长可以用几天？由谁批准？"]
QT = ["一笔涉及 {b} 元的{s}申请，需要谁审批？", "{s}申请金额是 {b} 元，审批人有哪些？", "金额 {b} 元的{s}，谁来批？",
      "我的{s}申请涉及 {b} 元，要走哪些审批？"]
QO = ["现在{s}每次收费多少？", "{s}目前的收费标准是多少？", "按现行规定，{s}一次要交多少钱？"]
QA = ["{s}收费标准修订后，费用由谁承担？", "{s}费用调整之后，钱还是由谁出？", "{s}改了收费以后，承担方变了吗？"]
QS = ["我是{d}的员工，每次{s}能用多久？", "{d}员工申请{s}，期限是多少天？", "作为{d}的同事，{s}一次可以用几天？"]
QSO = ["我在{d}，每次{s}能用多久？", "{d}员工的{s}期限是多少天？", "我们{d}申请{s}，一次能用几天？"]
QC = ["{c}申请{s}，要提前几个工作日？", "在{c}，{s}需要提前多久申请？", "{c}想申请{s}，最晚提前几天？"]
QCO = ["平时（非{c}）申请{s}，要提前几个工作日？", "不在{c}的时候，{s}要提前多久申请？", "正常情况下（不是{c}），{s}提前几天申请？"]
QF = ["每人每年最多能申请几次{s}？", "{s}一年最多申请几次？", "一个人一年能用几次{s}？"]
QSC = ["今年已经申请了 {h} 次{s}，还能再申请吗？", "我今年用了 {h} 次{s}，还可以再申请吗？", "{s}今年已经申请 {h} 次了，还能申请吗？"]
QP = ["违反{t}第一次会怎样？{s}申请{u}？", "如果违反了{t}，第一次怎么处理？另外{s}申请{u}？", "违规第一次的处理是什么？{s}申请{u}？"]
QR = ["{s}申请{u}？", "请问{s}的申请{u}？", "想了解一下：{s}申请{u}？"]
Q3 = ["{s}的申请流程、每人每年次数上限和费用分别是什么？", "关于{s}：多久前申请、一年最多几次、要不要付费？",
      "申请{s}要提前多久？每年能申请几次？费用怎么算？"]


def nice(v: int, rng) -> int:
    c = [v * 2, v + max(1, v // 2)]
    return rng.choice([x for x in c if x != v])


def topic_spec(stem, name, subj, rng):
    r1, r2 = rng.sample(ROLES, 2)
    return {"stem": stem, "name": name, "title": f"{name}管理规定", "s": subj, "r1": r1, "r2": r2,
            "adv": rng.choice([1, 2, 3, 5]), "amt": rng.choice([500, 1000, 2000, 3000, 5000]), "dur": rng.choice([3, 5, 7, 14, 30]),
            "cnt": rng.choice([2, 3, 4, 6, 12]), "exc": rng.choice(EXC), "fee": rng.choice([20, 50, 80, 100, 200]),
            "pun": rng.choice(PUN), "dept": rng.choice(DEPTS), "cond": rng.choice(CONDS), "unst": rng.sample(UNSTATED, 2),
            "month": rng.choice([3, 4, 5, 6, 7])}


def render_doc(t):
    t["dur2"] = t["dur"] * 2
    sec = {
        "申请与审批": f"{t['s']}须提前 {t['adv']} 个工作日提交申请，由{t['r1']}审批；涉及金额超过 {t['amt']} 元的，还须经{t['r2']}审批。",
        "期限与次数": f"每次{t['s']}的使用期限为 {t['dur']} 天，每人每年最多申请 {t['cnt']} 次。",
        "例外情况": f"{t['exc']}的，使用期限可延长至 {t['dur2']} 天，但须经{t['r2']}同意。",
        "费用": f"{t['s']}每次收取 {t['fee']} 元服务费，由申请人所在部门承担。",
        "违规处理": f"违反本规定的，第一次给予{t['pun'][0]}，第二次{t['pun'][1]}。",
    }
    md = f"# {t['title']}\n\n本文为虚构公司“示例科技”的原创合成制度，由程序按参数表生成，仅用于 P6-v4 训练实验。\n\n"
    md += "".join(f"## {k}\n\n{v}\n\n" for k, v in sec.items())
    return md, sec


def render_notices(t, rng):
    t["adv_new"] = nice(t["adv"], rng)
    t["fee_new"] = nice(t["fee"], rng)
    t["dur_scope"] = nice(t["dur"], rng)
    t["adv_cond"] = nice(t["adv"], rng)
    t["cnt_conf"] = nice(t["cnt"], rng)
    old_fee = f"{t['s']}每次收取 {t['fee']} 元服务费"
    n = {
        "费用修订": f"通知：自 2026 年 {t['month']} 月 1 日起，《{t['title']}》中“{old_fee}”的规定修订为每次收取 {t['fee_new']} 元，原规定同时废止；费用承担方式不变。",
        "部门补充规定": f"通知：仅适用于{t['dept']}员工的补充规定——每次{t['s']}的使用期限为 {t['dur_scope']} 天；其他部门仍按《{t['title']}》执行。",
        "特殊时段": f"通知：{t['cond']}，{t['s']}须提前 {t['adv_cond']} 个工作日提交申请；其他时间仍按原规定执行。",
    }
    md = f"# {t['name']}修订与补充通知\n\n本文为虚构的原创合成资料，由程序生成，仅用于 P6-v4 训练实验。\n\n" + "".join(f"## {k}\n\n{v}\n\n" for k, v in n.items())
    ops = f"# {t['name']}运营通知（转存）\n\n本文为虚构的原创合成资料，未注明生效日期，也未说明与原制度的关系。\n\n## 次数上限\n\n通知：每人每年最多申请{t['s']} {t['cnt_conf']} 次。\n\n"
    return md, ops


def build_topic(t, split, outdir, rng):
    doc, _ = render_doc(t)
    nmd, ops = render_notices(t, rng)
    for suffix, text in (("", doc), ("__notice", nmd), ("__ops", ops)):
        (outdir / f"{t['stem']}{suffix}.md").write_text(text, encoding="utf-8", newline="\n")
    C = {}
    for suffix in ("", "__notice", "__ops"):
        for c in chunk_document(outdir / f"{t['stem']}{suffix}.md", split):
            C[(suffix, c.section)] = c
    base = [C[("", i)] for i in range(1, 6)]
    notice = [C[("__notice", i)] for i in range(1, 4)]
    opsc = C[("__ops", 1)]
    body = lambda c: c.text.split("\n", 1)[1]  # noqa: E731
    others = []

    def ev(chs, extra_n=1):
        pool = [c for c in base if c not in chs]
        rng.shuffle(pool)
        e = list(chs) + pool[:extra_n]
        rng.shuffle(e)
        return e

    s = t["s"]
    samples = []

    def add(kind, beh, q, chs, claims, facts, extra=None, n_extra=1):
        e = ev(chs, n_extra)
        ans = "".join(f"{txt}[{c.cid}]。" for txt, c in claims) + "\n\n引用：" + ", ".join(dict.fromkeys(c.cid for _, c in claims))
        samples.append({"kind": kind, "behavior": beh, "question": q, "evidence": e, "gold": [c.cid for c in chs],
                        "answer": ans, "facts": [f for f, _ in facts], "fact_cids": [c.cid for _, c in facts], **(extra or {})})

    a, d, x, f, p = base
    st = lambda c: body(c).rstrip("。")  # noqa: E731
    add("multi2", "answer", rng.choice(Q2).format(s=s), [a, d], [(st(a), a), (f"每次{s}的使用期限为 {t['dur']} 天", d)],
        [(f"{t['adv']}个工作日", a), (t["r1"], a), (f"{t['dur']}天", d)])
    add("multi3", "answer", rng.choice(Q3).format(s=s), [a, d, f],
        [(f"{s}须提前 {t['adv']} 个工作日提交申请，由{t['r1']}审批", a), (f"每人每年最多申请 {t['cnt']} 次", d), (st(f), f)],
        [(f"{t['adv']}个工作日", a), (f"{t['cnt']}次", d), (f"{t['fee']}元", f)], n_extra=1)
    add("exception", "answer", rng.choice(QX).format(e=t["exc"], s=s), [x, d],
        [(st(x), x), (f"一般情况下每次{s}的使用期限为 {t['dur']} 天", d)], [(f"{t['dur2']}天", x), (t["r2"], x)])
    big = t["amt"] * 2
    add("threshold_trap", "answer", rng.choice(QT).format(b=big, s=s), [a],
        [(f"{s}由{t['r1']}审批；涉及金额超过 {t['amt']} 元的，还须经{t['r2']}审批，{big} 元超过了这一标准", a)],
        [(t["r1"], a), (t["r2"], a)], {"hard_negative_answer": f"由{t['r1']}审批[{a.cid}]。\n\n引用：{a.cid}",
                                        "hard_negative_note": "看似完整但遗漏“超过阈值须加签”的必要条件；仅存档，不参与 SFT"})
    # 冲突族：每个主题按轮换取 3 种
    fam = rng.sample(["override", "old_still", "scope_in", "scope_out", "cond_in", "cond_out", "conflict", "same_conclusion"], 3)
    for k in fam:
        if k == "override":
            add(k, "answer", rng.choice(QO).format(s=s), [notice[0], f],
                [(f"自 2026 年 {t['month']} 月 1 日起，{s}每次收取 {t['fee_new']} 元，原规定同时废止", notice[0]),
                 (f"《{t['title']}》中的每次 {t['fee']} 元是修订前的旧规定", f)],
                [(f"{t['fee_new']}元", notice[0])], {"no_conflict": True, "conflict_kind": "version_override"})
        elif k == "old_still":
            add(k, "answer", rng.choice(QA).format(s=s), [f, notice[0]],
                [(f"费用仍由申请人所在部门承担", f), (f"修订通知只调整了收费金额，并写明费用承担方式不变", notice[0])],
                [("申请人所在部门", f)], {"no_conflict": True, "conflict_kind": "old_rule_still_applies"})
        elif k == "scope_in":
            add(k, "answer", rng.choice(QS).format(d=t["dept"], s=s), [notice[1], d],
                [(f"仅适用于{t['dept']}员工的补充规定：每次{s}的使用期限为 {t['dur_scope']} 天", notice[1]),
                 (f"其他部门每次{s}的使用期限仍为 {t['dur']} 天", d)], [(f"{t['dur_scope']}天", notice[1])],
                {"no_conflict": True, "conflict_kind": "scope_difference"})
        elif k == "scope_out":
            other = rng.choice([x2 for x2 in DEPTS if x2 != t["dept"]])
            add(k, "answer", rng.choice(QSO).format(d=other, s=s), [d, notice[1]],
                [(f"{other}员工按《{t['title']}》执行，每次{s}的使用期限为 {t['dur']} 天", d),
                 (f"{t['dur_scope']} 天的规定仅适用于{t['dept']}员工", notice[1])], [(f"{t['dur']}天", d)],
                {"no_conflict": True, "conflict_kind": "scope_difference"})
        elif k == "cond_in":
            add(k, "answer", rng.choice(QC).format(c=t["cond"], s=s), [notice[2], a],
                [(f"{t['cond']}，{s}须提前 {t['adv_cond']} 个工作日提交申请", notice[2]), (f"其他时间为提前 {t['adv']} 个工作日", a)],
                [(f"{t['adv_cond']}个工作日", notice[2])], {"no_conflict": True, "conflict_kind": "condition_difference"})
        elif k == "cond_out":
            add(k, "answer", rng.choice(QCO).format(c=t["cond"].replace("每年 ", ""), s=s), [a, notice[2]],
                [(f"{s}须提前 {t['adv']} 个工作日提交申请，由{t['r1']}审批", a),
                 (f"提前 {t['adv_cond']} 个工作日只适用于{t['cond']}", notice[2])], [(f"{t['adv']}个工作日", a)],
                {"no_conflict": True, "conflict_kind": "condition_difference"})
        elif k == "conflict":
            add(k, "conflict", rng.choice(QF).format(s=s), [d, opsc],
                [(f"《{t['title']}》规定每次{s}的使用期限为 {t['dur']} 天，每人每年最多申请 {t['cnt']} 次", d),
                 (f"运营通知写的是每人每年最多申请{s} {t['cnt_conf']} 次", opsc)],
                [(f"{t['cnt']}次", d), (f"{t['cnt_conf']}次", opsc)], {"conflict_kind": "conflict_true"})
            samples[-1]["answer"] = samples[-1]["answer"].replace(
                "\n\n引用：", "两份资料的次数上限存在冲突，运营通知未注明生效日期，资料没有说明以哪一份为准。\n\n引用：")
        elif k == "same_conclusion":
            hi = max(t["cnt"], t["cnt_conf"]) + 1
            add(k, "answer", rng.choice(QSC).format(h=hi, s=s), [d, opsc],
                [(f"不能再申请：《{t['title']}》规定每人每年最多申请 {t['cnt']} 次", d),
                 (f"运营通知写的是每人每年最多申请{s} {t['cnt_conf']} 次，{hi} 次同样已超过", opsc)],
                [("不能", d)], {"no_conflict": True, "conflict_kind": "same_conclusion_despite_values"})
    u1, u2 = t["unst"]
    add("partial", "partial", rng.choice(QP).format(t=t["title"], s=s, u=u1[0]), [p],
        [(f"违反本规定的，第一次给予{t['pun'][0]}，第二次{t['pun'][1]}", p)], [(t["pun"][0], p)])
    samples[-1]["answer"] = samples[-1]["answer"].replace("\n\n引用：", f"资料中没有说明{s}申请{u1[0]}。\n\n引用：")
    # 同文档难负例拒答：参数表中不存在该信息
    e = ev([a, d, f], 1)
    samples.append({"kind": "refusal", "behavior": "refuse", "question": rng.choice(QR).format(s=s, u=u2[0]), "evidence": e, "gold": [],
                    "answer": f"资料中没有关于{s}申请{u2[0].replace('是否', '').replace('能否', '')}的说明，{REFUSAL_TEXT}\n\n引用：无",
                    "facts": [], "fact_cids": [], "refusal_checked": "由参数表保证文档中不含该信息"})
    out = []
    for i, smp in enumerate(samples):
        rid = f"v4_{t['stem']}_{smp['kind']}"
        rec = {"id": rid, "sample_id": rid, "split": split, "type": "B", "subtype": smp["kind"], "task_category": smp["kind"],
               "behavior": smp["behavior"], "question": smp["question"],
               "evidence": [{"cid": c.cid, "source": c.source, "doc": c.doc, "section": c.section, "text": c.text} for c in smp["evidence"]],
               "gold_cids": smp["gold"], "citation_ids": smp["gold"], "reference_answer": smp["answer"], "answer": smp["answer"],
               "facts": smp["facts"], "required_facts": smp["facts"], "fact_cids": smp["fact_cids"],
               "evidence_spans": [{"fact": f_, "cid": c_} for f_, c_ in zip(smp["facts"], smp["fact_cids"])],
               "forbid": [], "format": None, "no_conflict": smp.get("no_conflict", False), "conflict_kind": smp.get("conflict_kind"),
               "source_docs": sorted({c.doc for c in smp["evidence"]}), "source_document_ids": sorted({c.doc for c in smp["evidence"]}),
               "answerability": {"answer": "full", "conflict": "full", "partial": "partial", "refuse": "none"}[smp["behavior"]],
               "generation_method": "programmatic_structured", "review_status": "provisional", "dataset_version": VERSION,
               "topic_docs": [t["stem"]]}
        for k in ("hard_negative_answer", "hard_negative_note", "refusal_checked"):
            if k in smp:
                rec[k] = smp[k]
        out.append(rec)
    return out


def main() -> int:
    rng = random.Random(4242)
    built, problems = {}, []
    for split, topics in TOPICS.items():
        outdir = V4 / "data" / "corpus" / split
        outdir.mkdir(parents=True, exist_ok=True)
        recs = []
        for stem, name, subj in topics:
            recs += build_topic(topic_spec(stem, name, subj, rng), split, outdir, rng)
        bad = []
        for r in recs:
            s = score_item_v3(r, r["reference_answer"])
            ok = s["content_ok"] and s["format_ok"] and s["fact_attribution_ok"] is not False
            if r["behavior"] != "refuse":
                ok = ok and s["claim_support_full"]
            if s["is_multi"]:
                ok = ok and s["multi_content_complete"] and s["multi_chunk_coverage"]
            if not ok:
                bad.append((r["id"], s["content_ok"], s["format_ok"], s["fact_attribution_ok"], s["claim_support_full"],
                            s["n_supported"], s["n_claims"]))
        problems += [f"{split}: {b}" for b in bad]
        built[split] = [r for r in recs if r["id"] not in {b[0] for b in bad}]
    v3_full = read_jsonl(V3 / "data" / "v3" / "full" / "train.jsonl")
    test_q = [r["question"] for r in read_jsonl(V4 / "data" / "test_v4" / "built" / "test_v4.jsonl")]
    test_docs = {d for r in read_jsonl(V4 / "data" / "test_v4" / "built" / "test_v4.jsonl") for d in r["source_docs"]}
    new_docs = {d for r in built["train"] + built["dev"] for d in r["source_docs"]}
    prev_docs = {d for r in v3_full for d in r["source_docs"]}
    checks = {"doc_overlap_with_test_v4": sorted(new_docs & test_docs), "doc_overlap_with_v3_train": sorted(new_docs & prev_docs),
              "train_dev_doc_overlap": sorted({d for r in built["train"] for d in r["source_docs"]} & {d for r in built["dev"] for d in r["source_docs"]}),
              "max_q_jaccard_new_vs_test_v4": round(max(jaccard(r["question"], q) for r in built["train"] for q in test_q), 3),
              "duplicate_questions_in_new": sum(n - 1 for n in Counter(r["question"] for r in built["train"]).values() if n > 1),
              "self_check_rejected": problems}
    near = [(a["question"], b["question"]) for i, a in enumerate(built["train"]) for b in built["train"][i + 1:] if jaccard(a["question"], b["question"]) > 0.8]
    checks["near_duplicate_pairs(>0.8)"] = len(near)
    full_v4 = v3_full + built["train"]
    out = V4 / "data" / "v4"
    write_jsonl(out / "full" / "train.jsonl", full_v4)
    write_jsonl(out / "structured_only" / "train.jsonl", built["train"])
    dev = read_jsonl(V3 / "data" / "v3" / "dev.jsonl") + built["dev"]
    write_jsonl(out / "dev.jsonl", dev)
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    man = {"version": VERSION, "n_full": len(full_v4), "n_v3_part": len(v3_full), "n_new_structured": len(built["train"]),
           "n_dev": len(dev), "n_dev_new": len(built["dev"]),
           "new_by_kind": dict(Counter(r["subtype"] for r in built["train"])),
           "full_by_behavior": dict(Counter(r["behavior"] for r in full_v4)),
           "refusal_share_full": round(sum(r["behavior"] == "refuse" for r in full_v4) / len(full_v4), 3),
           "sha256": {"full": h(out / "full" / "train.jsonl"), "structured_only": h(out / "structured_only" / "train.jsonl"), "dev": h(out / "dev.jsonl")},
           "checks": checks}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items()}, ensure_ascii=False, indent=1)[:3000])
    return 1 if (checks["doc_overlap_with_test_v4"] or checks["train_dev_doc_overlap"]) else 0


if __name__ == "__main__":
    sys.exit(main())
