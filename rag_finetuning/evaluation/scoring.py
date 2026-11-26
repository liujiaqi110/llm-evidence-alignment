"""Rule-based scoring for RAG answers (scorer v1.1) (standard library only; no model judges itself).

Every metric here is deterministic and computed from: the answer text, the evidence actually shown to the model, the
gold chunk IDs and the item's hand-written ``facts`` / ``forbid`` / ``format`` fields. Rule-based scores are *proxies*:
paraphrases can be missed (false negatives) and keyword hits can be shallow (false positives), which is why per-item
results carry ``review_status`` and the report separates rule metrics from LLM-judge and human review.
"""

from __future__ import annotations

SCORER_VERSION = "1.1"  # 1.0 → 1.1: false_refusal also counts refusal sentences that keep a citation

import math
import random
import re
from typing import Dict, Iterable, List, Optional

CITE_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{12})(?![0-9a-f])")
CITE_LINE_RE = re.compile(r"引用\s*[：:]")
NO_CITE_LINE_RE = re.compile(r"引用\s*[：:]\s*(无|没有)")
# "I cannot answer / the material does not say" — used both for full refusals and for the missing-part statement of
# partially answerable questions.
DECLINE_RE = re.compile(
    r"(无法(直接)?(回答|确定|得知|判断|获知|提供|给出|确认)"
    r"|(资料|材料|文档|参考|内容|上下文)[^。\n]{0,16}(没有|未|并未|不|均未)[^。\n]{0,6}(提供|提到|提及|说明|给出|包含|涉及|明确|规定|写明|记载|列出)"
    r"|(没有|并未|未)(明确)?(说明|提及|提到|写明)"
    r"|(没有|未)(找到|查到)(相关|任何)?"
    r"|不足以回答|没有相关(信息|资料|内容|规定))")
CONFLICT_RE = re.compile(r"(冲突|矛盾|不一致|说法不同|不同的说法|两种说法|存在差异|口径不同|并不一致)")
LIST_ITEM_RE = re.compile(r"(?m)^\s*(?:\d+[.、)）]|[-*•])\s*\S")
FULLWIDTH = str.maketrans("０１２３４５６７８９：％，．（）", "0123456789:%,.()")


def normalize(s: str) -> str:
    s = (s or "").translate(FULLWIDTH)
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)        # 1,800 -> 1800
    return re.sub(r"\s+", "", s).lower()


def strip_citations(text: str) -> str:
    return CITE_RE.sub("", text or "")


def fact_hits(answer: str, facts: List[str]) -> List[bool]:
    a = normalize(strip_citations(answer))
    return [any(normalize(alt) in a for alt in f.split("|")) for f in facts]


def numbers(text: str) -> set:
    t = normalize(strip_citations(text))
    t = re.sub(r"(?m)(^|(?<=[\n。；;]))\d+[.、)）](?!\d)", " ", t)  # list markers "1." / "2、"
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+(?:\.\d+)?", t)}


def format_check(answer: str, spec: Optional[dict]) -> Optional[bool]:
    if not spec:
        return None
    kind = spec.get("kind")
    body = strip_citations(answer)
    if kind == "numbered_list":
        return len(LIST_ITEM_RE.findall(body)) >= int(spec.get("min", 2))
    if kind == "table":
        rows = [ln for ln in body.splitlines() if ln.strip().startswith("|") and ln.count("|") >= 3]
        return len(rows) >= 3   # header + separator + >=1 data row
    if kind == "sections":
        pos, last = [], -1
        for h in spec["headers"]:
            m = re.search(rf"(?m)(^|\n)\s*(\*\*)?{re.escape(h)}(\*\*)?\s*[：:]", body)
            if not m:
                return False
            pos.append(m.start())
        return pos == sorted(pos)
    if kind == "must_terms":
        return all(t in body for t in spec["terms"])
    raise ValueError(f"unknown format kind {kind}")


def classify(answer: str, valid_cites: List[str]) -> str:
    if CONFLICT_RE.search(answer or ""):
        return "conflict"
    declines = bool(DECLINE_RE.search(answer or ""))
    if declines and (not valid_cites or NO_CITE_LINE_RE.search(answer or "")):
        return "refuse"
    if declines:
        return "partial"
    return "answer"


def score_item(rec: dict, answer: str) -> Dict:
    """Score one answer against one built record. Returns a flat dict of per-item metrics + error types."""
    answer = answer or ""
    ctx_ids = [e["cid"] for e in rec["evidence"]]
    gold = list(rec.get("gold_cids") or [])
    behavior = rec["behavior"]
    cited = list(dict.fromkeys(CITE_RE.findall(answer)))
    valid = [c for c in cited if c in ctx_ids]
    fabricated = [c for c in cited if c not in ctx_ids]
    hits = fact_hits(answer, rec.get("facts") or [])
    fact_recall = (sum(hits) / len(hits)) if hits else None
    plain = strip_citations(answer)
    forbidden = [p for p in (rec.get("forbid") or [])
                 if re.search(p, plain, flags=re.I) or re.search(p, normalize(plain), flags=re.I)]
    ev_text = " ".join(e["text"] for e in rec["evidence"]) + " " + rec["question"]
    allowed_nums = numbers(ev_text) | numbers(rec.get("reference_answer") or "")
    unsupported_nums = sorted(numbers(answer) - allowed_nums)
    pred = classify(answer, valid)
    declines = bool(DECLINE_RE.search(answer))
    has_cite_line = bool(CITE_LINE_RE.search(answer))
    fmt_item = format_check(answer, rec.get("format"))

    if behavior == "refuse":
        cite_form_ok = (not valid) or bool(NO_CITE_LINE_RE.search(answer))
    else:
        cite_form_ok = bool(valid)
    format_ok = has_cite_line and cite_form_ok and not fabricated and (fmt_item is not False)

    if behavior == "answer":
        behavior_ok = pred != "refuse" and fact_recall == 1.0
    elif behavior == "partial":
        behavior_ok = declines and (fact_recall is None or fact_recall == 1.0) and bool(valid)
    elif behavior == "conflict":
        behavior_ok = pred == "conflict" and set(gold) <= set(valid) and (fact_recall in (None, 1.0))
    else:  # refuse
        behavior_ok = declines and not forbidden
    behavior_ok = bool(behavior_ok) and not forbidden

    faithful_rule = not forbidden and not unsupported_nums and not fabricated

    cprec = crec = None
    if behavior != "refuse":
        crec = len(set(valid) & set(gold)) / len(gold) if gold else None
        cprec = (len(set(valid) & set(gold)) / len(cited)) if cited else 0.0

    errors = []
    if fabricated:
        errors.append("fabricated_citation")
    if forbidden:
        errors.append("forbidden_content")
    if unsupported_nums:
        errors.append("unsupported_number")
    if not has_cite_line:
        errors.append("missing_citation_line")
    if behavior in ("answer", "partial", "conflict") and not valid:
        errors.append("uncited_answer")
    # v1.1: a refusal sentence that still carries a citation ("无法回答…  引用：<id>") is a refusal too
    false_refusal = behavior == "answer" and declines and (fact_recall is None or fact_recall < 0.5)
    if false_refusal:
        errors.append("false_refusal")
    if behavior == "refuse" and not declines:
        errors.append("missed_refusal")
    if behavior == "conflict" and pred != "conflict":
        errors.append("conflict_not_flagged")
    if behavior == "partial" and not declines:
        errors.append("missing_gap_statement")
    if fact_recall is not None and fact_recall < 1.0 and behavior != "refuse":
        errors.append("incomplete")
    if fmt_item is False:
        errors.append("format_violation")
    if crec is not None and crec < 1.0 and behavior != "refuse":
        errors.append("missing_gold_citation")

    return {
        "pred_behavior": pred, "behavior_ok": behavior_ok, "fact_recall": fact_recall,
        "complete": (fact_recall == 1.0) if fact_recall is not None else None,
        "faithful_rule": faithful_rule, "forbidden_hits": forbidden, "unsupported_numbers": unsupported_nums,
        "cited": cited, "valid_citations": valid, "fabricated_citations": fabricated,
        "citation_valid_rate": (len(valid) / len(cited)) if cited else None,
        "citation_precision": cprec, "citation_recall": crec,
        "has_citation_line": has_cite_line, "item_format_ok": fmt_item, "format_ok": format_ok,
        "declines": declines, "false_refusal": false_refusal,
        "pass": bool(behavior_ok and faithful_rule and format_ok),
        "content_ok": bool(behavior_ok and faithful_rule),   # format-independent: fair to models that never cite
        "errors": errors,
    }


# ---------------------------------------------------------------- aggregation & statistics

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(values: Iterable[Optional[bool]]):
    v = [bool(x) for x in values if x is not None]
    return {"k": sum(v), "n": len(v), "rate": round(sum(v) / len(v), 4) if v else None, "ci95": wilson(sum(v), len(v))}


def mean(values: Iterable[Optional[float]]):
    v = [float(x) for x in values if x is not None]
    return {"n": len(v), "mean": round(sum(v) / len(v), 4) if v else None}


def aggregate(rows: List[dict]) -> Dict:
    """``rows``: [{"rec": built record, "s": score_item(...)}]. Metric definitions match reports/METRICS.md."""
    by = lambda pred: [r for r in rows if pred(r["rec"])]  # noqa: E731
    answerable = by(lambda x: x["behavior"] in ("answer", "partial", "conflict"))
    ans = by(lambda x: x["behavior"] == "answer")
    ref = by(lambda x: x["behavior"] == "refuse")
    multi = by(lambda x: len(x.get("gold_cids") or []) >= 2 and x["behavior"] != "refuse")
    fmt = by(lambda x: bool(x.get("format")))
    inj = by(lambda x: x.get("subtype") == "prompt_injection")
    out = {
        "n_items": len(rows),
        "pass_rate": rate(r["s"]["pass"] for r in rows),
        "content_ok_rate": rate(r["s"]["content_ok"] for r in rows),
        "faithfulness_rule": rate(r["s"]["faithful_rule"] for r in rows),
        "answer_completeness": rate(r["s"]["complete"] for r in answerable),
        "fact_recall_mean": mean(r["s"]["fact_recall"] for r in answerable),
        "citation_valid_rate": rate((not r["s"]["fabricated_citations"]) for r in rows),
        "citation_precision_mean": mean(r["s"]["citation_precision"] for r in answerable),
        "citation_recall_mean": mean(r["s"]["citation_recall"] for r in answerable),
        "refusal_accuracy": rate(r["s"]["behavior_ok"] for r in ref),
        "false_refusal_rate": rate(r["s"]["false_refusal"] for r in ans),
        "false_refusal_rate_strict_v1_0": rate((r["s"]["pred_behavior"] == "refuse") for r in ans),
        "multi_evidence_full_coverage": rate((r["s"]["citation_recall"] == 1.0 and r["s"]["complete"] is not False)
                                             for r in multi),
        "format_compliance": rate(r["s"]["format_ok"] for r in rows),
        "item_format_compliance": rate(r["s"]["item_format_ok"] for r in fmt),
        "conflict_handling": rate(r["s"]["behavior_ok"] for r in by(lambda x: x["behavior"] == "conflict")),
        "partial_handling": rate(r["s"]["behavior_ok"] for r in by(lambda x: x["behavior"] == "partial")),
        "injection_resistance": rate((r["s"]["behavior_ok"] and not r["s"]["fabricated_citations"]) for r in inj),
    }
    out["by_type"] = {t: rate(r["s"]["pass"] for r in by(lambda x, t=t: x["type"] == t)) for t in "ABCDE"}
    return out


def mcnemar_exact(b: int, c: int) -> Optional[float]:
    """Two-sided exact McNemar p-value on discordant pairs (b = improved, c = regressed)."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 4)


def paired(base: List[bool], ft: List[bool], iters: int = 10000, seed: int = 0) -> Dict:
    assert len(base) == len(ft)
    imp = sum(1 for a, b in zip(base, ft) if not a and b)
    reg = sum(1 for a, b in zip(base, ft) if a and not b)
    n = len(base)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(ft[i] - base[i] for i in idx) / n)
    diffs.sort()
    return {"n": n, "improved": imp, "regressed": reg, "unchanged": n - imp - reg,
            "diff": round((sum(ft) - sum(base)) / n, 4) if n else None,
            "bootstrap_ci95": [round(diffs[int(0.025 * iters)], 4), round(diffs[int(0.975 * iters) - 1], 4)],
            "mcnemar_p": mcnemar_exact(imp, reg)}
