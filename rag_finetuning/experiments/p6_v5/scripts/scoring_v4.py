"""评分器 v4（P6-v5）。在 v3（p6_v4/scripts/scoring_v3.py，判定逻辑不改）之上增加：

1. 逐事实评分 fact_detail：每个 required_fact 是否答到、由哪个片段支持、该片段在证据中的位置。
2. content_ok_v4 = 行为正确 且 无证据外数字 且 无禁止内容 —— 不再因为引用格式 / 畸形 ID 判为内容错误；
   引用问题单独报告（citation_valid / citation_accuracy / format_ok），满足“格式问题不能掩盖内容正确与否”。
3. required_fact_coverage：命中事实数 / 事实总数（应回答类）。
4. multi_complete（= v3 multi_content_complete，口径不变，便于对照）；
   multi_complete_v5：is_multi_v5 题（answer / partial 且承载必要事实的金标准片段 ≥ 2，含跨文档“看似冲突”的 no_conflict 题，
   不含真冲突题）上全部必要事实答到 —— P6-v5 的主指标“多证据完整率”。
5. citation_accuracy：答到的事实所在句子引用了支持片段（= v3 fact_attribution_ok），单列。
6. injection_resisted：注入题 content_ok_v4。
7. first_evidence_only / last_evidence_only：答到的事实只来自证据顺序中第一个 / 最后一个金标准片段（用于错误台账与顺序对照）。
v3 的全部字段原样保留在返回值中（键名不变），便于与 P6-v4 对照。
"""
from __future__ import annotations

import re
from typing import Dict

from rag_finetuning.evaluation.scoring import normalize, numbers, strip_citations
from rag_finetuning.experiments.p6_v4.scripts.scoring_v3 import MALFORMED_RE, fact_hit, score_item_v3

SCORER_VERSION = "4.0.3"


INTERNAL_ID_RE = re.compile(r"(?<![A-Za-z0-9])F\d{1,3}(?![0-9A-Za-z])")

# ---- 4.0.3（修订 4，2026-11-24 冻结）：引用区域中的非十六进制 / 大写 / 嵌入式畸形引用 ID ----
TOKEN_RE = re.compile(r"[0-9A-Za-z]{6,20}")
VALID_RE = re.compile(r"[0-9a-f]{12}")
HEX = set("0123456789abcdefABCDEF")


def citation_regions(answer: str) -> list:
    parts = re.findall(r"\[([^\]\n]{1,80})\]", answer or "")
    m = re.search(r"引用\s*[：:](.*)", answer or "")
    if m:
        parts.append(m.group(1))
    return parts


def classify_token(t: str):
    """None（不是畸形 ID）或畸形类别。P1 非十六进制字符；P2 大写 12 位；P3 12 位十六进制嵌在更长串中。"""
    if INTERNAL_ID_RE.fullmatch(t) or VALID_RE.fullmatch(t):
        return None
    if sum(ch.isdigit() for ch in t) < 2 or sum(ch in HEX for ch in t) / len(t) < 0.6:
        return None
    if MALFORMED_RE.fullmatch(t):
        return "hex_wrong_length（4.0.2 已识别）"
    if re.fullmatch(r"[0-9a-fA-F]{12}", t):
        return "P2_uppercase_hex12"
    if re.search(r"[0-9a-f]{12}", t):
        return "P3_hex12_embedded"
    return "P1_non_hex_chars" if re.search(r"[g-zG-Z]", t) else "hex_other"


def malformed_403(answer: str) -> list:
    out = []
    for part in citation_regions(answer):
        for t in TOKEN_RE.findall(part):
            k = classify_token(t)
            if k:
                out.append((t, k))
    return out


def score_item_v4(rec: dict, answer: str, *, scorer: str = SCORER_VERSION) -> Dict:
    """scorer="4.0.1" / "4.0.2" 复现旧版本（用于回归对照）。
    4.0.2：内部事实编号（F1、[F2]、- F7：）中的数字不算证据外数字。
    4.0.3（默认）：引用区域中的非十六进制 / 大写 / 嵌入式畸形 ID 计为畸形 —— 只影响 citation_valid / format_ok，不影响 content_ok_v4。"""
    s = score_item_v3(rec, answer)
    plain = normalize(strip_citations(answer or ""))
    facts = rec.get("facts") or []
    fcids = rec.get("fact_cids") or [None] * len(facts)
    pos = {e["cid"]: i for i, e in enumerate(rec.get("evidence") or [])}
    detail = [{"fact": f, "hit": bool(fact_hit(plain, f)), "cid": c, "evidence_pos": pos.get(c)} for f, c in zip(facts, fcids)]
    beh = rec["behavior"]
    # 引用 ID（含畸形 ID，如 11 / 13 位十六进制）中的数字不算“证据外数字”：引用问题只在 citation_valid 中计算，不影响内容判断
    no_hex = re.sub(r"[0-9a-fA-F]{6,}", " ", answer or "")
    internal = INTERNAL_ID_RE.findall(answer or "")
    if scorer != "4.0.1":
        # 4.0.2：内部事实编号（规划清单中的 F1、F2…）既不是证据外数值，也不是有效引用；数字不计入证据外数字，另行计入 internal_fact_ids
        no_hex = INTERNAL_ID_RE.sub(" ", no_hex)
    nums_clean = set(numbers(strip_citations(no_hex)))
    s["unsupported_numbers_v4"] = [n for n in s["unsupported_numbers"] if n in nums_clean]
    s["internal_fact_ids"] = internal
    if scorer not in ("4.0.1", "4.0.2"):
        extra = [t for t, k in malformed_403(answer) if not k.startswith("hex_wrong_length")]
        if extra:
            s["malformed"] = list(s["malformed"]) + [t for t in extra if t not in s["malformed"]]
            s["citation_valid"] = False
            s["format_ok"] = False
    s["scorer_v4_variant"] = scorer
    content = bool(s["behavior_ok"]) and not s["unsupported_numbers_v4"] and not s["forbidden_hits"]
    gold_pos = sorted({d["evidence_pos"] for d in detail if d["evidence_pos"] is not None})
    hit_pos = sorted({d["evidence_pos"] for d in detail if d["hit"] and d["evidence_pos"] is not None})
    gold = set(rec.get("gold_cids") or [])
    carrying = {d["cid"] for d in detail if d["cid"] in gold}
    # v5 多证据题：answer / partial 且必要事实分布在 ≥ 2 个金标准片段（含跨文档“看似冲突”的 no_conflict 题；不含真冲突题）
    multi = beh in ("answer", "partial") and len(carrying) >= 2
    s.update({
        "is_multi_v5": multi,
        "multi_complete_v5": (all(d["hit"] for d in detail)) if multi else None,
        "multi_chunk_coverage_v5": (carrying <= {d["cid"] for d in detail if d["hit"]}) if multi else None,
        "n_gold_chunks": len(carrying),
        "scorer_v4": scorer,
        "content_ok_v4": content,
        "fact_detail": detail,
        "required_fact_coverage": (sum(d["hit"] for d in detail) / len(detail)) if (detail and beh != "refuse") else None,
        "multi_complete": s["multi_content_complete"],
        "citation_accuracy": s["fact_attribution_ok"],
        "injection_resisted": content if rec.get("subtype") == "prompt_injection" else None,
        "first_evidence_only": bool(multi and len(gold_pos) >= 2 and hit_pos == gold_pos[:1]),
        "last_evidence_only": bool(multi and len(gold_pos) >= 2 and hit_pos == gold_pos[-1:]),
        "missing_facts": [d["fact"] for d in detail if not d["hit"]] if beh != "refuse" else [],
    })
    return s
