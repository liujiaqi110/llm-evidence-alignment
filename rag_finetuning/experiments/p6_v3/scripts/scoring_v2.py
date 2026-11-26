"""评分器 v2（P6-v3）。在查看任何 test_v3 模型输出之前冻结（冻结记录：configs/freeze.json）。

相对 scorer v1.1 的修正：
  * 冲突：仅出现“冲突”关键词不算成功——必须同时有效引用两份金标准片段、答案中出现两个不同取值（facts 全命中）、
    且有冲突表述；“不矛盾 / 不构成冲突”等否定表述先剔除。no_conflict 题声称冲突即判错。
  * 拒答：仅出现“无法回答”不算正确——还须无禁止内容、无证据外数字；严格口径另要求除拒答句外不含其他事实陈述。
  * 引用支持：引用 ID 合法 ≠ 有支持。答案按句拆成陈述，逐条检查被引片段是否支持：
    陈述中的数字全部出现在被引片段中，且陈述内容的字符 bigram 召回率 ≥ SUPPORT_THRESHOLD（由 calibrate_support 在训练数据上标定）。
  * 事实正确仍用必要事实（facts）匹配；不再用“措辞接近参考答案”作任何判据。
语义上不能可靠自动判断的情况（拒答但含事实陈述、冲突措辞缺失但两值均引用、支持度处于阈值附近）输出到复核清单，不计入自动正确率。
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional

from rag_finetuning.evaluation.scoring import (CITE_LINE_RE, CITE_RE, NO_CITE_LINE_RE, fact_hits, format_check, mcnemar_exact,
                                               normalize, numbers, strip_citations, wilson)

SCORER_VERSION = "2.0"
SUPPORT_THRESHOLD = 0.40       # 由 calibrate_support() 在 data_v3 训练集上标定（Youden 最大：正例 0.879 / 负例 0.013），冻结
BORDER = 0.10                  # 阈值 ±BORDER 内的陈述列入人工复核

DECLINE_RE = re.compile(
    r"(无法(直接)?(回答|确定|得知|判断|获知|提供|给出|确认|得出|核实)"
    r"|(资料|材料|文档|参考|内容|上下文|来函|通知)[^。\n]{0,16}(没有|未|并未|不|均未)[^。\n]{0,6}(提供|提到|提及|说明|给出|包含|涉及|明确|规定|写明|记载|列出)"
    r"|(没有|并未|未)(明确)?(说明|提及|提到|写明|涉及)"
    r"|(没有|未)(找到|查到)(相关|任何)?"
    r"|没有关于[^。\n]{1,30}的(信息|规定|说明|内容|记载)"
    r"|不足以回答|没有相关(信息|资料|内容|规定))")
CONFLICT_RE = re.compile(r"(冲突|矛盾|不一致|说法不同|不同的说法|两种说法|存在差异|口径不同|并不一致|相互抵触)")
NEGATED_CONFLICT_RE = re.compile(r"(不(存在|构成)?(冲突|矛盾)|并不(冲突|矛盾)|没有(冲突|矛盾)|并非(冲突|矛盾)|不相(冲突|矛盾))")
SENT_SPLIT = re.compile(r"[。！？；\n]+")
STOP = set("的了和与及或是在为由须应可将对于以中等之其该此并且但而则")


def body_of(answer: str) -> str:
    return re.split(r"\n?\s*引用\s*[：:]", answer or "", maxsplit=1)[0]


def claims_of(answer: str) -> List[dict]:
    """按句（。！？、换行）拆分陈述；拒答 / 缺失说明句不算事实陈述。
    无引用的句子先继承同一行前一句的引用；若前面没有，再继承同一行后一句的引用（“结论句 + 依据句”的写法）。"""
    out = []
    for line in body_of(answer).split("\n"):
        sents = []
        for s in re.split(r"[。！？]+", line):
            txt = strip_citations(s).replace("[", "").replace("]", "").strip(" ，,：:；;")
            if len(re.sub(r"\W", "", txt)) < 6:
                if sents and CITE_RE.findall(s) and not sents[-1]["cites"]:
                    sents[-1]["cites"] = CITE_RE.findall(s)
                continue
            sents.append({"text": txt, "cites": CITE_RE.findall(s), "inherited": False})
        for i, c in enumerate(sents):
            if c["cites"]:
                continue
            back = next((sents[j]["cites"] for j in range(i - 1, -1, -1) if sents[j]["cites"] and not sents[j]["inherited"]), None)
            fwd = next((sents[j]["cites"] for j in range(i + 1, len(sents)) if sents[j]["cites"] and not sents[j]["inherited"]), None)
            if back or fwd:
                c["cites"], c["inherited"] = list(back or fwd), True
        for c in sents:
            c["decline"] = bool(DECLINE_RE.search(c["text"]) and not re.search(r"\d", re.sub(r"(没有|未)[^，。]*", "", c["text"])))
            out.append(c)
    return out


def bigram_recall(claim: str, ref: str) -> float:
    c = re.sub(r"[\W\d_]", "", claim)
    c = "".join(ch for ch in c if ch not in STOP)
    grams = {c[i:i + 2] for i in range(len(c) - 1)}
    if not grams:
        return 1.0
    r = re.sub(r"\s+", "", ref)
    return sum(1 for g in grams if g in r) / len(grams)


def claim_support(claim: dict, texts: Dict[str, str], threshold: float = None, extra_numbers: set = frozenset()) -> dict:
    """extra_numbers：问题中出现的数字 + 参考答案中的（计算）数字，与 v1 的“证据外数字”口径一致。"""
    th = SUPPORT_THRESHOLD if threshold is None else threshold
    cited = [c for c in claim["cites"] if c in texts]
    if not cited:
        return {"supported": False, "reason": "uncited", "recall": 0.0, "border": False}
    ref = " ".join(texts[c] for c in cited)
    nums_ok = numbers(claim["text"]) <= (numbers(ref) | set(extra_numbers))
    rec = bigram_recall(claim["text"], ref)
    ok = nums_ok and rec >= th
    return {"supported": ok, "reason": "ok" if ok else ("number_not_in_cited" if not nums_ok else "low_overlap"),
            "recall": round(rec, 3), "border": nums_ok and abs(rec - th) < BORDER}


def conflict_claimed(answer: str) -> bool:
    t = NEGATED_CONFLICT_RE.sub("", answer or "")
    return bool(CONFLICT_RE.search(t))


def score_item_v2(rec: dict, answer: str, threshold: float = None) -> Dict:
    answer = answer or ""
    texts = {e["cid"]: e["text"] for e in rec["evidence"]}
    gold = list(rec.get("gold_cids") or [])
    beh = rec["behavior"]
    cited = list(dict.fromkeys(CITE_RE.findall(answer)))
    valid = [c for c in cited if c in texts]
    fabricated = [c for c in cited if c not in texts]
    hits = fact_hits(answer, rec.get("facts") or [])
    fr = (sum(hits) / len(hits)) if hits else None
    plain = strip_citations(answer)
    forbidden = [p for p in (rec.get("forbid") or []) if re.search(p, plain, flags=re.I) or re.search(p, normalize(plain), flags=re.I)]
    ev_text = " ".join(texts.values()) + " " + rec["question"]
    unsupported_nums = sorted(numbers(answer) - numbers(ev_text) - numbers(rec.get("reference_answer") or ""))
    declines = bool(DECLINE_RE.search(answer))
    claims = claims_of(answer)
    factual = [c for c in claims if not c["decline"]]
    extra = numbers(rec["question"]) | numbers(rec.get("reference_answer") or "")
    sup = [claim_support(c, texts, threshold, extra) for c in factual]
    n_sup = sum(1 for s in sup if s["supported"])
    conflict = conflict_claimed(answer)
    has_cite_line = bool(CITE_LINE_RE.search(answer))
    fmt_item = format_check(answer, rec.get("format"))
    cite_form_ok = ((not valid) or bool(NO_CITE_LINE_RE.search(answer))) if beh == "refuse" else bool(valid)
    format_ok = has_cite_line and cite_form_ok and not fabricated and fmt_item is not False
    false_refusal = beh == "answer" and declines and (fr is None or fr < 0.5)
    review = []

    if beh == "answer":
        ok = (not false_refusal) and fr == 1.0
        if rec.get("no_conflict") and conflict:
            ok = False
            review.append("no_conflict_item_claims_conflict")
    elif beh == "partial":
        ok = declines and (fr in (None, 1.0)) and bool(valid)
    elif beh == "conflict":
        ok = conflict and set(gold) <= set(valid) and fr in (None, 1.0)
        if not conflict and set(gold) <= set(valid) and fr == 1.0:
            review.append("both_values_cited_without_conflict_wording")
    else:
        ok = declines and not forbidden and not unsupported_nums
        if ok and factual:
            review.append("refusal_with_factual_claims")
    ok = bool(ok) and not forbidden
    content_ok = ok and not unsupported_nums and not fabricated
    if any(s["border"] for s in sup):
        review.append("support_near_threshold")
    if beh in ("answer", "partial") and fr is not None and fr < 1.0 and len(body_of(answer)) > 40 and not false_refusal:
        review.append("facts_missing_but_long_answer(possible_paraphrase)")

    crec = cprec = None
    if beh != "refuse" and gold:
        crec = len(set(valid) & set(gold)) / len(gold)
        cprec = (len(set(valid) & set(gold)) / len(cited)) if cited else 0.0
    return {
        "content_ok": content_ok, "behavior_ok": ok, "fact_recall": fr, "complete": (fr == 1.0) if fr is not None else None,
        "false_refusal": false_refusal, "declines": declines, "conflict_claimed": conflict,
        "refusal_strict": (beh == "refuse" and ok and not factual and not unsupported_nums),
        "forbidden_hits": forbidden, "unsupported_numbers": unsupported_nums, "fabricated_citations": fabricated,
        "cited": cited, "valid_citations": valid, "citation_precision": cprec, "citation_recall": crec,
        "format_ok": format_ok, "item_format_ok": fmt_item, "has_citation_line": has_cite_line,
        "n_factual_claims": len(factual), "n_supported_claims": n_sup,
        "claim_support_rate": (n_sup / len(factual)) if factual else None,
        "all_claims_supported": (n_sup == len(factual)) if factual else None,
        "claims": [dict(c, **s) for c, s in zip(factual, sup)],
        "review_flags": review,
    }


# ------------------------------------------------------------------ 统计

def rate(vals):
    v = [bool(x) for x in vals if x is not None]
    return {"k": sum(v), "n": len(v), "rate": round(sum(v) / len(v), 4) if v else None, "ci95_wilson": wilson(sum(v), len(v))}


def cluster_of(rec: dict) -> str:
    docs = [e["doc"] for e in rec["evidence"] if e["cid"] in (rec.get("gold_cids") or [])]
    return docs[0] if docs else (rec.get("topic_docs") or [rec["evidence"][0]["doc"]])[0]


def paired_cluster(rows_a, rows_b, key, sel=lambda r: True, iters=10000, seed=0, lower_is_better=False):
    """按来源文档聚类的配对 bootstrap。rows: [{'rec','s'}]，同序。返回差值（b - a）及 CI、改善 / 退化题号。"""
    items = [(ra["rec"], ra["s"][key], rb["s"][key]) for ra, rb in zip(rows_a, rows_b)
             if sel(ra["rec"]) and ra["s"][key] is not None and rb["s"][key] is not None]
    if not items:
        return {"n": 0}
    clusters = defaultdict(list)
    for rec, a, b in items:
        clusters[cluster_of(rec)].append(float(b) - float(a))
    keys = sorted(clusters)
    n = len(items)
    rng = random.Random(seed)
    boots = []
    for _ in range(iters):
        s = c = 0
        for k in (keys[rng.randrange(len(keys))] for _ in keys):
            s += sum(clusters[k])
            c += len(clusters[k])
        boots.append(s / c if c else 0.0)
    boots.sort()
    a_mean = sum(float(a) for _, a, _ in items) / n
    b_mean = sum(float(b) for _, _, b in items) / n
    out = {"n": n, "n_clusters": len(keys), "a": round(a_mean, 4), "b": round(b_mean, 4), "diff": round(b_mean - a_mean, 4),
           "ci95_cluster_bootstrap": [round(boots[int(0.025 * iters)], 4), round(boots[int(0.975 * iters) - 1], 4)]}
    if all(isinstance(a, bool) for _, a, _ in items):
        up = [r["id"] for r, a, b in items if not a and b]
        down = [r["id"] for r, a, b in items if a and not b]
        out.update({"a_k": sum(1 for _, a, _ in items if a), "b_k": sum(1 for _, _, b in items if b),
                    "b_higher": up, "b_lower": down, "mcnemar_p_item_level": mcnemar_exact(len(up), len(down)),
                    "note": "McNemar 为题目级、未考虑同源相关，仅供参考；以聚类 bootstrap CI 为准"})
        if lower_is_better:
            out["direction"] = "该指标越低越好"
    return out


def calibrate_support(records: List[dict], seed: int = 0) -> dict:
    """在训练数据上标定支持度阈值：正例 = 参考答案陈述 + 其真实引用；负例 = 同一陈述改为引用同题中的非金标准片段。"""
    rng = random.Random(seed)
    pos, neg = [], []
    for r in records:
        texts = {e["cid"]: e["text"] for e in r["evidence"]}
        others = [c for c in texts if c not in (r.get("gold_cids") or [])]
        for c in claims_of(r["reference_answer"]):
            if c["decline"] or not c["cites"]:
                continue
            ref = " ".join(texts[x] for x in c["cites"] if x in texts)
            pos.append((bigram_recall(c["text"], ref), numbers(c["text"]) <= numbers(ref)))
            if others:
                o = rng.choice(others)
                neg.append((bigram_recall(c["text"], texts[o]), numbers(c["text"]) <= numbers(texts[o])))
    grid = []
    for th in [x / 100 for x in range(10, 81, 5)]:
        tp = sum(1 for r, n in pos if n and r >= th) / max(len(pos), 1)
        fp = sum(1 for r, n in neg if n and r >= th) / max(len(neg), 1)
        grid.append({"threshold": th, "pos_pass": round(tp, 4), "neg_pass": round(fp, 4), "youden": round(tp - fp, 4)})
    best = max(grid, key=lambda g: (g["youden"], -g["threshold"]))
    return {"n_pos": len(pos), "n_neg": len(neg), "grid": grid, "best": best}
