"""审核页面的 AI 辅助判断（确定性程序规则，非人工、非大模型；只用于提示，不写入任何人工字段）。

为每个审核条目补充：
* evidence[*].sentences：证据逐句切分，标注含哪些必要事实、是否条件句、被 chosen / rejected 哪些陈述最匹配；
* claims 的最匹配证据句（字符 bigram 召回，沿用评分器 v2 的 bigram_recall）；
* ai：建议（偏好成立 / 可能不成立 / 不确定；hard_dev 为标注检查结果）、理由（每条带依据）、缺失事实、引用问题、适用条件、评分争议提示；
* risk：风险分与原因（页面按风险从高到低排序）。
"""
from __future__ import annotations

import itertools
import re

from rag_finetuning.evaluation.scoring import normalize, numbers
from rag_finetuning.experiments.p6_v3.scripts import scoring_v2 as v2
from rag_finetuning.experiments.p6_v5.scripts.scorer_403_proposal import malformed_403

COND = {"exception": "例外条件：证据中有“例外 / 但”条款，需判断问题是否落在例外里",
        "threshold_trap": "金额 / 数量门槛：需判断问题中的数值是否超过门槛",
        "cond_in": "条件内：问题满足某个条件，相应规定适用",
        "cond_out": "条件外：问题不满足条件，相应规定不适用",
        "scope_in": "适用范围：两条规定适用范围不同，不是冲突",
        "same_conclusion": "两条规定结论相同，不是冲突",
        "override": "新规覆盖旧规：应以新版本为准",
        "partial": "资料只能回答一部分：应回答能答的部分并说明缺失",
        "conflict": "资料之间存在真实冲突：应指出冲突并分别列出",
        "multi2": "需要整合 2 个片段", "multi3": "需要整合 3 个片段", "natural_multi_ef": "需要整合多个片段",
        "refusal": "资料不足：应拒答"}
COND_RE = re.compile(r"(超过|以上|以下|不足|例外|但|除|如遇|若|仅限|仅|不得|须经|还须|方可|期间|之前|之后|以内)")
STRONG = 0.5
AI_ASSIST_VERSION = "2"  # 1：首版；2：“引用可能可接受”要求陈述中的必要事实全部出现在所引片段中（2026-11-24 审核中发现误报后修正）


def split_sents(text: str):
    head, _, body = text.partition("\n")
    return head, [s.strip() for s in re.split(r"(?<=[。；])", body or head) if s.strip()]


def best_match(claim: str, ev):
    best = (0.0, None, None)
    for e in ev:
        for i, s in enumerate(e["sentences"]):
            r = v2.bigram_recall(claim, s["text"])
            if r > best[0]:
                best = (r, e["cid"], i)
    return best


def annotate_evidence(item):
    for e in item["evidence"]:
        head, sents = split_sents(e["text"])
        e["head"] = head
        e["sentences"] = []
        for s in sents:
            facts = [f["fact"] for f in item["facts"] if f["cid"] == e["cid"] and any(a and normalize(a) in normalize(s) for a in f["fact"].split("|"))]
            e["sentences"].append({"text": s, "facts": facts, "condition": bool(COND_RE.search(s)), "marks": []})


def annotate_claims(item, key, side, ev=None):
    ev = ev if ev is not None else item["evidence"]
    for k, c in enumerate(item[key]["claims"]):
        r, cid, idx = best_match(c["text"], ev)
        c["best"] = {"cid": cid, "idx": idx, "score": round(r, 2)} if cid else None
        if cid and r >= STRONG:
            e = next(e for e in ev if e["cid"] == cid)
            e["sentences"][idx]["marks"].append(f"{side}{k + 1}")


def fact_src(item, fact):
    f = next((x for x in item["facts"] if x["fact"] == fact), None)
    return f"[{f['cid']}]「{f['sentence']}」" if f and f["sentence"] else (f"[{f['cid']}]" if f else "")


def citation_issues(item, key):
    """逐句检查引用：最匹配证据句不在所引片段中 / 程序判定无支持 / 所引片段中也有相近句（引用可能可以接受）。"""
    out = []
    ev = {e["cid"]: e for e in item["evidence"]}
    for k, c in enumerate(item[key]["claims"]):
        b = c.get("best")
        if c.get("supported") is not False and (not b or b["score"] < STRONG or b["cid"] in c["cites"]):
            continue
        alt = max((v2.bigram_recall(c["text"], s["text"]) for cid in c["cites"] if cid in ev for s in ev[cid]["sentences"]), default=0.0)
        # 该句中出现的必要事实，是否也能在所引片段原文中找到
        ct = normalize(c["text"])
        facts_here = [f["fact"] for f in item.get("facts", []) if any(x and normalize(x) in ct for x in f["fact"].split("|"))]
        # v2（2026-11-24 修正）：该句中的必要事实必须**全部**出现在所引片段中才算“引用可能可接受”。
        # v1 只要求出现任意一个，导致误报：例如陈述含“60 天 / 项目负责人”，所引片段只在另一条规则里有“项目负责人”。
        in_cited = [f for f in facts_here if any(any(x and normalize(x) in normalize(ev[cid]["text"]) for x in f.split("|")) for cid in c["cites"] if cid in ev)]
        out.append({"claim": k + 1, "text": c["text"], "cites": c["cites"], "best": b, "alt_score_in_cited": round(alt, 2),
                    "facts_in_claim": facts_here, "facts_found_in_cited": in_cited,
                    "maybe_acceptable": alt >= STRONG or (bool(facts_here) and len(in_cited) == len(facts_here))})
    return out


def pair_assist(item, pending: dict):
    annotate_evidence(item)
    annotate_claims(item, "auto_chosen", "C")
    annotate_claims(item, "auto_rejected", "R")
    ac, ar = item["auto_chosen"], item["auto_rejected"]
    reasons, doubts = [], []
    miss_r = [f for f in ar["missing_facts"] if f not in ac["missing_facts"]]
    for f in miss_r:
        reasons.append({"kind": "缺失事实", "text": f"rejected 缺少必要事实「{f}」，chosen 有", "basis": fact_src(item, f)})
    if ac["missing_facts"]:
        doubts.append({"kind": "chosen 有缺失", "text": "chosen 也缺少：" + "、".join(ac["missing_facts"]), "basis": "；".join(fact_src(item, f) for f in ac["missing_facts"])})
    ci_r = citation_issues(item, "auto_rejected")
    ci_c = citation_issues(item, "auto_chosen")
    for x in ci_r:
        where = f"最匹配的证据句在 [{x['best']['cid']}] 第 {x['best']['idx'] + 1} 句（匹配度 {x['best']['score']}）" if x["best"] else "没有找到匹配的证据句"
        if x["maybe_acceptable"]:
            why_ok = (f"所引片段原文中也有必要事实「{'、'.join(x['facts_found_in_cited'])}」" if x["facts_found_in_cited"]
                      else f"所引片段中也有相近句（匹配度 {x['alt_score_in_cited']}）")
            doubts.append({"kind": "引用可能可接受", "text": f"rejected 第 {x['claim']} 句引用 {', '.join(x['cites']) or '（无）'}，程序判定不支持；但{why_ok}",
                           "basis": where})
        else:
            reasons.append({"kind": "引用错误", "text": f"rejected 第 {x['claim']} 句「{x['text'][:40]}」引用 {', '.join(x['cites']) or '（无）'}", "basis": where})
    for x in ci_c:
        doubts.append({"kind": "chosen 引用存疑", "text": f"chosen 第 {x['claim']} 句引用 {', '.join(x['cites']) or '（无）'}，程序判定不支持", "basis": f"[{x['best']['cid']}] 第 {x['best']['idx'] + 1} 句" if x["best"] else ""})
    bad_ids = [t for t, k in malformed_403(item["rejected"]) if not k.startswith("hex_wrong")] + ar["malformed_ids"] + ar["fabricated_ids"]
    if bad_ids:
        reasons.append({"kind": "引用 ID 写坏", "text": "rejected 的引用 ID 无效：" + "、".join(dict.fromkeys(bad_ids)), "basis": "评分器 4.0.2 / 4.0.3 提议规则"})
    if ar["whole_refusal"] and not ac["whole_refusal"]:
        reasons.append({"kind": "过度拒答", "text": "rejected 整体拒答，但证据可以回答（chosen 给出了答案）", "basis": "必要事实都在证据中"})
    if item["type"] == "I" and ar["conflict_claimed"]:
        reasons.append({"kind": "误判冲突", "text": "rejected 声称资料冲突，但两条规定适用范围 / 条件不同", "basis": COND.get(item.get("subtype"), "")})
    if item["type"] == "G" and not ar["conflict_claimed"]:
        reasons.append({"kind": "未指出冲突", "text": "rejected 没有指出资料之间的冲突，只采信了一方", "basis": "资料存在真实冲突"})
    if ar["unsupported_numbers"]:
        reasons.append({"kind": "证据外数字", "text": "rejected 出现证据中没有的数字：" + " ".join(ar["unsupported_numbers"]), "basis": "评分器 4.0.2"})
    if not reasons and not doubts:
        doubts.append({"kind": "差异不明显", "text": "程序没有找到 rejected 的明确错误，请人工比较", "basis": item["error_explanation"]})
    chosen_ok = ac["content_ok_v4"] and not ac["missing_facts"] and not ci_c
    if not chosen_ok and ar["content_ok_v4"] and not ar["missing_facts"]:
        sug = "可能不成立"
    elif doubts:
        sug = "不确定"
    else:
        sug = "偏好成立"
    cond = COND.get(item.get("subtype") or "", "")
    item["ai"] = {"suggestion": sug, "reasons": reasons, "doubts": doubts, "condition": cond,
                  "missing_in_rejected": miss_r, "missing_in_chosen": ac["missing_facts"], "pending_exclusion": pending.get(item["id"], "")}
    risk, why = {"H": 60, "C": 40, "I": 40}.get(item["type"], 50), []
    if "Q1_H_stratified" in item["quotas"] and item["rejected_kind"].startswith("真实"):
        risk += 10
        why.append("H 类真实错误")
    if sug == "不确定":
        risk += 30
        why.append("AI 辅助判断不确定")
    if sug == "可能不成立":
        risk += 35
        why.append("AI 辅助判断可能不成立")
    if not chosen_ok:
        risk += 15
        why.append("chosen 自动检查未全过")
    if item["chosen_source"] == "reference":
        why.append("chosen 为程序参考答案")
    if item["ai"]["pending_exclusion"]:
        risk, why = 5, ["已按决策排除（低优先级）"]
    item["risk"] = {"score": risk, "reasons": why}
    return item


def derive_formula(n: str, pool):
    try:
        x = float(n)
    except ValueError:
        return ""
    vals = sorted({float(p) for p in pool if re.fullmatch(r"\d+(\.\d+)?", p)})
    for a, b in itertools.permutations(vals, 2):
        for v, op in ((a + b, "+"), (a - b, "−"), (a * b, "×"), (a * b / 100, "×%")):
            if abs(v - x) < 1e-6:
                fa, fb = (f"{a:g}", f"{b:g}")
                return f"{fa} × {fb}% = {n}" if op == "×%" else f"{fa} {op} {fb} = {n}"
    return ""


def dev_assist(item, rec, drows, score_fn):
    """drows：disputes_by_category.csv 中该题的行（hard_dev 各组 + Oracle 条件）。"""
    annotate_evidence(item)
    pool = set(numbers(" ".join(e["text"] for e in rec["evidence"]))) | set(numbers(rec["question"]))
    reasons, doubts = [], []
    no_src = [f["fact"] for f in item["facts"] if not f["sentence"]]
    if no_src:
        doubts.append({"kind": "事实无原句", "text": "程序在来源片段中找不到这些必要事实的原句：" + "、".join(no_src), "basis": "请核对必要事实是否写对"})
    else:
        reasons.append({"kind": "标注自检", "text": f"{len(item['facts'])} 个必要事实都能在标注的来源片段中找到原句", "basis": "见“必要事实”表"})
    ref_ok = score_fn(rec, rec["reference_answer"])["content_ok_v4"]
    (reasons if ref_ok else doubts).append({"kind": "参考答案自检", "text": "参考答案通过评分器 4.0.2 自检" if ref_ok else "参考答案未通过评分器自检", "basis": "评分器 4.0.2"})
    for a in item["answers"]:
        annotate_claims(a, "auto", a["group"] + ":", item["evidence"])
    merged = {}
    for r in drows:
        k = (r["source"], r["group"])
        m = merged.setdefault(k, {"source": r["source"], "group": r["group"], "answer": r["answer"], "parts": []})
        for part in re.split(r"；", r["reason"]):
            if part and part not in m["parts"]:
                m["parts"].append(part)
    disputes = []
    for (src, g), m in sorted(merged.items()):
        au = score_fn(rec, m["answer"])
        for part in m["parts"]:
            hint = ""
            if part.startswith("同义表达"):
                mm = re.search(r"缺失“([^”]+)”.*（([^）]+)）", part)
                if mm and (au["whole_refusal_on_answerable"] or "无法回答" in m["answer"]):
                    hint = f"答案整体拒答，「{mm.group(2)}」很可能是“无法回答”的意思，不是「{mm.group(1)}」的同义表达"
                elif mm:
                    hint = f"答案用「{mm.group(2)}」表达，请判断是否等同于必要事实「{mm.group(1)}」"
            elif part.startswith("中文数字"):
                hint = "答案把数字写成中文（如“一张”），语义相同；需要决定评分是否接受这种写法"
            elif part.startswith("推算数字"):
                nums = re.findall(r"'([\d.]+)'", part)
                fs = [f for f in (derive_formula(n, pool) for n in nums) if f]
                hint = ("可能的算法：" + "；".join(fs) + "。请核对计算是否符合规则（例如有无上限、单位换算）") if fs else "请核对该数字的来由"
            disputes.append({"key": f"{src}|{g}|{len(disputes)}", "source": src, "group": g, "reason": part, "hint": hint, "answer": m["answer"],
                             "missing": au["missing_facts"], "unsupported": au["unsupported_numbers_v4"]})
    if disputes:
        doubts.append({"kind": "评分争议", "text": f"{len(disputes)} 处可能的评分器误判，需要逐条判断", "basis": "见“评分争议”区"})
    cond = COND.get(item.get("subtype") or "", "") or COND.get(item.get("behavior") or "", "")
    item["ai"] = {"suggestion": "标注未发现问题" if not doubts or (len(doubts) == 1 and disputes) else "标注需核对",
                  "reasons": reasons, "doubts": doubts, "condition": cond, "disputes": disputes, "pending_exclusion": ""}
    risk = 100 if disputes else (85 if item.get("behavior") in ("conflict", "partial") else 75)
    why = (["存在评分争议"] if disputes else []) + ([f"行为 {item['behavior']}"] if item.get("behavior") in ("conflict", "partial") else []) +           (["必要事实无原句"] if no_src else []) + ([] if ref_ok else ["参考答案自检未通过"])
    item["risk"] = {"score": risk + (10 if no_src or not ref_ok else 0), "reasons": why or ["hard_dev 高优先级"]}
    return item
