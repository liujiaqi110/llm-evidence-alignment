"""评分器 v3（P6-v4）。在查看任何 test_v4 模型输出之前冻结。修正 METRIC_AUDIT.md 中列出的问题：

1. 陈述支持使用全分母：应回答类题（answer / partial / conflict）全部计入，未作答 = 未支持。
2. 多证据拆为两个指标，并且不含冲突题：
   * multi_content_complete：每条必要事实都被答到（内容层面）；
   * multi_chunk_coverage：每个“承载必要事实的金标准片段”至少有一条事实被答到；
   * fact_attribution_ok：每条答到的事实所在的句子（含继承），引用了包含该事实的片段。
3. 应回答类题整体拒答率（分母 = answer + partial + conflict），与 answer 题错误拒答率并列报告。
4. 畸形 ID（8–11 位或 13–16 位十六进制）计为无效引用。
5. 禁止规则在句子级做否定感知：命中句中含“不 / 未 / 没有 / 是否 / 无法 / 并非 / 尚未 / 并未 / 忽略”则不算命中；
   以 "!" 开头的模式为严格匹配（用于假 ID、注入口令）。
6. 必要事实支持少量受控同义（可以 ⇔ 均可 / 可报销 / 能够 / 允许；不能 ⇔ 不得 / 不可以 / 禁止 / 不允许 / 不予）。
7. 冲突类按子类报告：conflict_true / version_override / old_rule_still_applies / scope_difference / condition_difference。
其余判定沿用 v2（scoring_v2.py），并对同一输出并行给出 v2 结果。
"""

from __future__ import annotations

import re
from typing import Dict, List

from rag_finetuning.evaluation.scoring import (CITE_LINE_RE, CITE_RE, NO_CITE_LINE_RE, format_check, normalize, numbers,
                                               strip_citations)
from rag_finetuning.experiments.p6_v3.scripts import scoring_v2 as v2

SCORER_VERSION = "3.0"
SUPPORT_THRESHOLD = v2.SUPPORT_THRESHOLD          # 0.40，沿用 v3 标定，不重新调
MALFORMED_RE = re.compile(r"(?<![0-9a-zA-Z])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)([0-9a-f]{8,11}|[0-9a-f]{13,16})(?![0-9a-zA-Z])")
NEG_RE = re.compile(r"(不|未|没有|是否|无法|并非|尚未|并未|忽略|非真实|不是事实|属于指令)")
SYNONYMS = {"可以": ["可以", "均可", "可报销", "能够", "允许", "能报"],
            "不能": ["不能", "不得", "不可以", "禁止", "不允许", "不予", "不可"]}
ANSWERABLE = ("answer", "partial", "conflict")


def expand(fact: str) -> List[str]:
    alts = []
    for a in fact.split("|"):
        alts += SYNONYMS.get(a, [a])
    return [normalize(a) for a in alts]


def fact_hit(answer_norm: str, fact: str) -> bool:
    return any(a and a in answer_norm for a in expand(fact))


def forbidden_hits(answer: str, patterns: List[str]) -> List[str]:
    plain = strip_citations(answer)
    sents = [s for s in re.split(r"[。！？\n；;]+", plain) if s.strip()]
    hits = []
    for p in patterns or []:
        strict = p.startswith("!")
        pat = p[1:] if strict else p
        if strict:
            if re.search(pat, answer, flags=re.I) or re.search(pat, normalize(plain), flags=re.I):
                hits.append(p)
            continue
        for s in sents:
            if re.search(pat, s, flags=re.I) and not NEG_RE.search(s):
                hits.append(p)
                break
    return hits


def sentence_cites(answer: str) -> List[dict]:
    return v2.claims_of(answer)          # 句子 + 继承后的引用（v2 规则）


def score_item_v3(rec: dict, answer: str) -> Dict:
    answer = answer or ""
    s2 = v2.score_item_v2(rec, answer)  # 并行保留 v2
    texts = {e["cid"]: e["text"] for e in rec["evidence"]}
    beh = rec["behavior"]
    gold = list(rec.get("gold_cids") or [])
    facts = list(rec.get("facts") or [])
    fact_cids = list(rec.get("fact_cids") or [None] * len(facts))
    an = normalize(strip_citations(answer))
    hits = [fact_hit(an, f) for f in facts]
    fr = sum(hits) / len(hits) if hits else None
    complete = (fr == 1.0) if hits else None
    cited = list(dict.fromkeys(CITE_RE.findall(answer)))
    valid = [c for c in cited if c in texts]
    fabricated = [c for c in cited if c not in texts]
    malformed = [m for m in MALFORMED_RE.findall(answer)]
    forbid = forbidden_hits(answer, rec.get("forbid") or [])
    declines = bool(v2.DECLINE_RE.search(answer))
    conflict = v2.conflict_claimed(answer)
    extra_nums = numbers(rec["question"]) | numbers(rec.get("reference_answer") or "")
    unsupported_nums = sorted(numbers(answer) - numbers(" ".join(texts.values())) - extra_nums)
    # 事实 → 片段（显式标注优先，否则在金标准片段中定位）
    fchunks = []
    for f, fc in zip(facts, fact_cids):
        if fc:
            fchunks.append([fc])
        else:
            fchunks.append([c for c in gold if any(a in normalize(texts.get(c, "")) for a in expand(f))] or
                           [c for c in texts if any(a in normalize(texts[c]) for a in expand(f))])
    sents = sentence_cites(answer)
    attrib = []
    for f, h, fc in zip(facts, hits, fchunks):
        if not h or not fc:
            continue
        ok = False
        for st in sents:
            if any(a in normalize(st["text"]) for a in expand(f)):
                if set(st["cites"]) & set(fc):
                    ok = True
                    break
        attrib.append(ok)
    carrying = sorted({c for fc in fchunks for c in fc if c in gold})
    covered = sorted({c for fc, h in zip(fchunks, hits) if h for c in fc if c in gold})
    # 行为
    whole_refusal = beh in ANSWERABLE and declines and (fr is None or fr < 0.5)
    sub = rec.get("conflict_kind") or (rec.get("subtype") if rec.get("no_conflict") else None)
    if beh == "answer":
        ok = (not whole_refusal) and complete is not False
        if rec.get("no_conflict") and conflict:
            ok = False
    elif beh == "partial":
        ok = declines and complete is not False and bool(valid)
    elif beh == "conflict":
        ok = conflict and complete is True and set(gold) <= set(valid)
    else:
        ok = declines and not forbid and not unsupported_nums
    ok = bool(ok) and not forbid
    claims = [c for c in s2["claims"]]
    n_fact = len(claims)
    n_sup = sum(1 for c in claims if c["supported"])
    has_line = bool(CITE_LINE_RE.search(answer))
    fmt_item = format_check(answer, rec.get("format"))
    cite_form = ((not valid) or bool(NO_CITE_LINE_RE.search(answer))) if beh == "refuse" else bool(valid)
    is_multi = beh in ("answer", "partial") and len(carrying) >= 2 and not rec.get("no_conflict")
    return {
        "content_ok": ok and not unsupported_nums and not fabricated and not malformed,
        "behavior_ok": ok, "fact_recall": fr, "complete": complete,
        "false_refusal": beh == "answer" and whole_refusal,
        "whole_refusal_on_answerable": whole_refusal if beh in ANSWERABLE else None,
        "refusal_ok": ok if beh == "refuse" else None,
        "refusal_strict": (ok and n_fact == 0) if beh == "refuse" else None,
        "partial_ok": ok if beh == "partial" else None,
        "conflict_true_ok": ok if beh == "conflict" else None,
        "no_conflict_ok": ok if rec.get("no_conflict") else None, "no_conflict_kind": sub if rec.get("no_conflict") else None,
        "is_multi": is_multi,
        "multi_content_complete": (complete is True) if is_multi else None,
        "multi_chunk_coverage": (set(carrying) <= set(covered)) if is_multi else None,
        "fact_attribution_ok": (all(attrib) if attrib else None),
        "claim_support_full": (n_fact > 0 and n_sup == n_fact) if beh in ANSWERABLE else None,
        "claim_support_and_complete": (n_fact > 0 and n_sup == n_fact and complete is True) if beh in ANSWERABLE else None,
        "n_claims": n_fact, "n_supported": n_sup,
        "citation_valid": not fabricated and not malformed, "fabricated": fabricated, "malformed": malformed,
        "format_ok": has_line and cite_form and not fabricated and not malformed and fmt_item is not False,
        "forbidden_hits": forbid, "unsupported_numbers": unsupported_nums, "declines": declines, "conflict_claimed": conflict,
        "v2": {k: s2[k] for k in ("content_ok", "behavior_ok", "complete", "false_refusal", "format_ok", "all_claims_supported")},
    }
