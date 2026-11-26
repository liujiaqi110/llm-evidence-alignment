"""P6-v4 两个独立工程基线（与模型权重无关；可复现、纯规则）。

方案一 coverage_check(question, evidence, answer)：事实覆盖检查
  * 相关片段：问题与片段的“内容 bigram”重叠率 ≥ THETA_REL（开发集标定）；
  * 对答案未引用、且其最相关句子未被答案覆盖（bigram 召回 < 0.5）的相关片段，在“引用：”行前追加
    “补充：<该片段中与问题最相关的原句>[cid]。”，并把 cid 加入引用行；
  * 拒答答案不改动（避免把真正的拒答改成作答），只返回标记。
方案二 detect_conflicts(evidence)：检索证据间的冲突预检测
  * 抽取每个片段中带数量单位的数值事实（值、单位、遮蔽数字后的上下文键）与标记：
    覆盖（“自 X 起 … 修订 / 调整为 … 废止”）、范围（“仅适用于 …”）、条件（“… 期间 / … 之后 / … 时，”）；
  * 不同文档、同单位、不同取值、上下文键 bigram Jaccard ≥ TAU 的事实对 → 按标记分类：
    version_override / scope_difference / condition_difference / conflict_true；TAU_LOW ≤ 相似度 < TAU → insufficient。
  * 相似度只用于给“同一属性”做候选配对，阈值在开发集上验证；分类依据是显式文字标记，而不是相似度。
  * conflict_hint(evidence) 把检测结果渲染成一段提示，作为“+工程方案二”变体追加到用户消息末尾（Prompt 其余部分不变）。
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Dict, List

from rag_finetuning.evaluation.scoring import CITE_RE, normalize

THETA_REL = 0.20          # 开发集标定（v3 test 上 D / E 后处理，平均 content_ok 最高），冻结
TAU, TAU_LOW = 0.35, 0.20   # 开发集标定（宏 F1 0.879，105 个无关对零误报），冻结
UNITS = r"(万元|元|%|个工作日|工作日|天|小时|分钟|次|人|个月|周|名|份|台|张|倍)"
QTY = re.compile(r"(?<![\d:：.])(\d+(?:[.,]\d+)?)\s*" + UNITS)
OVERRIDE = re.compile(r"自\s*\d{4}\s*年[^，。]{0,12}起[^。]*(修订|调整)为[^。]*(废止|替代|不再执行)")
SCOPE = re.compile(r"仅适用于([^，。—]{1,14}?)(员工|供应商|部门)")
COND = re.compile(r"([^，。：]{2,16}(期间|之后|之前|以后))[，,]|((法定)?节假日[^，。：]{0,10})[，,]|(不论[^，。]{1,16})[，,]")
OTHERWISE = re.compile(r"其他(时间|月份|时段|情况)仍按")
STOP = set("的了和与及或是在为由须应可将对于以中等之其该此并且但而则每次")


def bigrams(s: str) -> set:
    s = re.sub(r"[\W\d_]", "", s)
    s = "".join(ch for ch in s if ch not in STOP)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def body(text: str) -> str:
    return text.split("\n", 1)[-1]


def sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[。！？])", body(text)) if s.strip()]


# ---------------------------------------------------------------- 方案一

def relevance(question: str, text: str) -> float:
    q = bigrams(question)
    return len(q & bigrams(text)) / len(q) if q else 0.0


def coverage_check(question: str, evidence: List[dict], answer: str, theta: float = THETA_REL) -> Dict:
    from rag_finetuning.experiments.p6_v3.scripts.scoring_v2 import DECLINE_RE
    cited = set(CITE_RE.findall(answer or ""))
    body_ans = re.split(r"\n?\s*引用\s*[：:]", answer or "", maxsplit=1)[0]
    is_refusal = bool(DECLINE_RE.search(answer or "")) and not cited
    added = []
    for e in evidence:
        if relevance(question, e["text"]) < theta or e["cid"] in cited:
            continue
        best = max(sentences(e["text"]), key=lambda s: relevance(question, s), default="")
        if not best:
            continue
        g = bigrams(best)
        covered = len(g & bigrams(body_ans)) / len(g) if g else 1.0
        if covered < 0.5:
            added.append((e["cid"], best.rstrip("。")))
    if is_refusal or not added:
        return {"answer": answer, "added": [], "flag_refusal_with_relevant": is_refusal and bool(added)}
    supp = "".join(f"补充：{s}[{c}]。" for c, s in added)
    m = re.search(r"\n?\s*引用\s*[：:]\s*(.*)$", answer, flags=re.S)
    if m:
        ids = [x.strip() for x in re.split(r"[,，、\s]+", m.group(1)) if x.strip() and x.strip() != "无"]
        new = answer[:m.start()].rstrip() + supp + "\n\n引用：" + ", ".join(dict.fromkeys(ids + [c for c, _ in added]))
    else:
        new = answer.rstrip() + supp + "\n\n引用：" + ", ".join(dict.fromkeys(list(cited) + [c for c, _ in added]))
    return {"answer": new, "added": [c for c, _ in added], "flag_refusal_with_relevant": False}


# ---------------------------------------------------------------- 方案二

def chunk_facts(e: dict) -> List[dict]:
    out = []
    b = body(e["text"])
    marks = {"override": bool(OVERRIDE.search(b)), "scope": SCOPE.search(b).group(1) if SCOPE.search(b) else None,
             "condition": (next((g for g in COND.search(b).groups() if g), None) if COND.search(b) else None)
             or (re.sub(r"^通知：", "", b).split("，")[0][:16] if OTHERWISE.search(b) else None)}
    for s in sentences(e["text"]):
        s2 = re.sub(r"^通知：", "", s)
        if marks["override"] and "修订为" in s2:
            # 覆盖类通知：配对键用引号中的旧规定原文；取值用“修订为”之后的新值
            quoted = re.search(r"[“\"]([^”\"]+)[”\"]", s2)
            new_part = s2.split("修订为", 1)[1].split("，原规定")[0]
            m = QTY.search(new_part)
            if quoted and m:
                key = QTY.sub("#", quoted.group(1))
                out.append({"cid": e["cid"], "doc": e["doc"], "value": m.group(1).replace(",", ""), "unit": m.group(2),
                            "key": key, "sentence": s, "marks": marks})
            continue
        for clause in re.split(r"[；;]", s2):
            clause = re.sub(r"^.*?补充规定——", "", clause)
            clause = SCOPE.sub("", clause)
            clause = COND.sub("", clause)
            for m in QTY.finditer(clause):
                out.append({"cid": e["cid"], "doc": e["doc"], "value": m.group(1).replace(",", ""), "unit": m.group(2),
                            "key": QTY.sub("#", clause), "sentence": s, "marks": marks})
    return out


def detect_conflicts(evidence: List[dict], tau: float = TAU, tau_low: float = TAU_LOW) -> List[Dict]:
    facts = [f for e in evidence for f in chunk_facts(e)]
    found = {}
    for f1, f2 in combinations(facts, 2):
        if f1["doc"] == f2["doc"] or f1["unit"] != f2["unit"] or f1["value"] == f2["value"]:
            continue
        a, b = bigrams(f1["key"]), bigrams(f2["key"])
        sim = len(a & b) / len(a | b) if a | b else 0.0
        if sim < tau_low:
            continue
        pair = tuple(sorted([f1["cid"], f2["cid"]]))
        if sim < tau:
            kind = "insufficient"
        else:
            m1, m2 = f1["marks"], f2["marks"]
            if m1["override"] or m2["override"]:
                kind = "version_override"
            elif m1["scope"] or m2["scope"]:
                kind = "scope_difference"
            elif m1["condition"] or m2["condition"]:
                kind = "condition_difference"
            else:
                kind = "conflict_true"
        prev = found.get(pair)
        rank = ["insufficient", "conflict_true", "condition_difference", "scope_difference", "version_override"]
        if prev is None or (sim, rank.index(kind)) > (prev["sim"], rank.index(prev["kind"])):
            found[pair] = {"cids": list(pair), "kind": kind, "sim": round(sim, 3),
                           "values": [(f1["cid"], f1["value"] + f1["unit"]), (f2["cid"], f2["value"] + f2["unit"])],
                           "scope": f1["marks"]["scope"] or f2["marks"]["scope"], "condition": f1["marks"]["condition"] or f2["marks"]["condition"]}
    return list(found.values())


HINT = {"conflict_true": "资料 [{a}] 与 [{b}] 对同一事项的取值不同（{va} 与 {vb}），且未发现生效日期、适用范围或触发条件的说明，可能存在真实冲突：回答时应分别列出两种说法并指出冲突。",
        "version_override": "资料 [{a}] 与 [{b}] 取值不同（{va} 与 {vb}），其中一份是明确写明生效日期并废止原规定的修订通知：应以修订后的规定为准，并说明旧规定已被修订。",
        "scope_difference": "资料 [{a}] 与 [{b}] 取值不同（{va} 与 {vb}），其中一份仅适用于“{scope}”：应按提问者所属范围选择适用的规定，不属于冲突。",
        "condition_difference": "资料 [{a}] 与 [{b}] 取值不同（{va} 与 {vb}），其中一份只在“{cond}”条件下适用：应按问题中的条件选择适用的规定，不属于冲突。",
        "insufficient": "资料 [{a}] 与 [{b}] 中有相近事项的取值不同（{va} 与 {vb}），但无法判断是否为同一规定，请结合原文判断。"}


def conflict_hint(evidence: List[dict]) -> str:
    det = detect_conflicts(evidence)
    if not det:
        return ""
    lines = [HINT[d["kind"]].format(a=d["values"][0][0], b=d["values"][1][0], va=d["values"][0][1], vb=d["values"][1][1],
                                    scope=d["scope"] or "", cond=d["condition"] or "") for d in det]
    return "\n\n【资料一致性检查（系统自动生成，仅供参考，不是资料内容）】\n" + "\n".join(f"- {x}" for x in lines)
