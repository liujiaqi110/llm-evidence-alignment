"""P6-v4 指标审计：从 v3 的原始逐题输出重新计算指标，核对报告数字、分母与筛选规则，检查选择偏差与措辞偏好。

    python -m rag_finetuning.experiments.p6_v4.scripts.metric_audit

只读取 v3 的冻结输入、参考答案与各组 generations.jsonl；不修改任何 v3 文件。输出到 p6_v4/results/metric_audit/。
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.experiments.p6_v3.scripts.scoring_v2 import DECLINE_RE, CONFLICT_RE, conflict_claimed, score_item_v2

V3 = Path(__file__).resolve().parents[2] / "p6_v3"
OUT = Path(__file__).resolve().parents[1] / "results" / "metric_audit"
EVAL = Path("workspace/runs/p6_v3/eval/test_v3")
GROUPS = ["A", "B", "C", "D", "E", "F"]
ANS = ("answer", "partial", "conflict")
LOOSE_DECLINE = re.compile(r"(无法|不能确定|不确定|没有提供|未提供|没有给出|未给出|没有相关|并未说明|不清楚|没有具体|未具体|没有明确)")
TEMPLATE_DECLINE = re.compile(r"资料中没有(关于|说明|提到)")


def main() -> int:
    recs = {r["id"]: r for r in read_jsonl(V3 / "data" / "test_v3" / "built" / "test_v3.jsonl")}
    order = list(json.loads((V3 / "data" / "frozen" / "test_v3.manifest.json").read_text(encoding="utf-8"))["items"])
    stored = json.loads((V3 / "results" / "test_v3" / "summary.json").read_text(encoding="utf-8"))
    sc, ans = {}, {}
    for g in GROUPS:
        gens = {x["id"]: x["answer"] for x in read_jsonl(EVAL / g / "generations.jsonl")}
        ans[g] = gens
        sc[g] = {q: score_item_v2(recs[q], gens[q]) for q in order}
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"recomputed_vs_stored": {}, "claim_support": {}, "selection_bias": {}, "category_coverage": {},
           "wording_bias": {}, "valid_id_unsupported": {}}

    # 1) 重算与 v3 summary.json 的一致性
    for g in GROUPS:
        cok = sum(sc[g][q]["content_ok"] for q in order)
        rep["recomputed_vs_stored"][g] = {"content_ok_recomputed": cok, "content_ok_stored": stored["groups"][g]["metrics"]["content_ok"]["k"],
                                          "match": cok == stored["groups"][g]["metrics"]["content_ok"]["k"]}

    # 2) 陈述支持：三种口径
    answerable = [q for q in order if recs[q]["behavior"] in ANS]
    for g in GROUPS:
        own = [q for q in answerable if sc[g][q]["all_claims_supported"] is not None]
        rep["claim_support"][g] = {
            "per_group_denominator(REPORT 口径)": {"k": sum(sc[g][q]["all_claims_supported"] for q in own), "n": len(own),
                                                  "excluded_no_claims": [q for q in answerable if q not in own]},
            "full_denominator(v4 口径：68 道应回答类题，未作答=未支持)": {
                "k": sum(bool(sc[g][q]["all_claims_supported"]) for q in answerable), "n": len(answerable)},
            "answered_supported_and_complete": {"k": sum(bool(sc[g][q]["all_claims_supported"]) and sc[g][q]["complete"] is True
                                                         for q in answerable), "n": len(answerable)},
            "claim_level": {"k": sum(sc[g][q]["n_supported_claims"] for q in answerable),
                            "n": sum(sc[g][q]["n_factual_claims"] for q in answerable)}}
    for a, b in (("D", "E"), ("A", "C"), ("A", "B")):
        inter = [q for q in answerable if sc[a][q]["all_claims_supported"] is not None and sc[b][q]["all_claims_supported"] is not None]
        rep["claim_support"][f"paired_{a}_vs_{b}(MODEL_DECISION 口径：两组都有陈述的交集)"] = {
            "n": len(inter), a: sum(sc[a][q]["all_claims_supported"] for q in inter), b: sum(sc[b][q]["all_claims_supported"] for q in inter),
            "dropped_items": [q for q in answerable if q not in inter]}

    # 3) 选择偏差：少答 / 拒答是否让“全部有支持”更容易
    for g in GROUPS:
        own = [q for q in answerable if sc[g][q]["all_claims_supported"] is not None]
        full = [q for q in own if sc[g][q]["complete"] is True]
        part = [q for q in own if sc[g][q]["complete"] is False]
        nclaims = [sc[g][q]["n_factual_claims"] for q in own]
        rep["selection_bias"][g] = {
            "support_rate_when_complete": f"{sum(sc[g][q]['all_claims_supported'] for q in full)}/{len(full)}",
            "support_rate_when_incomplete": f"{sum(sc[g][q]['all_claims_supported'] for q in part)}/{len(part)}",
            "mean_claims_per_answered_item": round(sum(nclaims) / len(nclaims), 2) if nclaims else None,
            "answerable_items_with_no_claims(拒答或空)": len(answerable) - len(own)}

    # 4) 类别覆盖：每题只属于一个行为指标；检查遗漏
    beh = Counter(recs[q]["behavior"] for q in order)
    for g in GROUPS:
        refused_partial = [q for q in order if recs[q]["behavior"] == "partial" and sc[g][q]["declines"] and not sc[g][q]["valid_citations"]]
        refused_conflict = [q for q in order if recs[q]["behavior"] == "conflict" and sc[g][q]["declines"] and (sc[g][q]["fact_recall"] or 0) < 0.5]
        rep["category_coverage"][g] = {"behavior_counts": dict(beh),
                                       "false_refusal_denominator": beh["answer"],
                                       "full_refusals_on_partial_items(不计入错误拒答，只计入部分回答失败)": refused_partial,
                                       "refusals_on_conflict_items(不计入错误拒答)": refused_conflict}

    # 5) 措辞偏好
    for g in GROUPS:
        refuse_items = [q for q in order if recs[q]["behavior"] == "refuse"]
        loose_not_strict = [q for q in order if LOOSE_DECLINE.search(ans[g][q]) and not DECLINE_RE.search(ans[g][q])]
        template_hits = sum(1 for q in order if TEMPLATE_DECLINE.search(ans[g][q]))
        nc = [q for q in order if recs[q].get("no_conflict")]
        conflict_raw = [q for q in order if CONFLICT_RE.search(ans[g][q] or "")]
        rep["wording_bias"][g] = {
            "answers_with_v3_template_phrase(资料中没有关于/说明/提到)": template_hits,
            "decline_like_but_not_detected_by_DECLINE_RE": loose_not_strict,
            "refuse_items_correct": sum(sc[g][q]["behavior_ok"] for q in refuse_items),
            "conflict_keyword_answers": len(conflict_raw),
            "conflict_keyword_on_no_conflict_items": [q for q in nc if conflict_claimed(ans[g][q])]}

    # 6) 引用 ID 有效但陈述无支持
    for g in GROUPS:
        rep["valid_id_unsupported"][g] = [{"id": q, "claims": [c["text"][:60] for c in sc[g][q]["claims"] if not c["supported"] and c["reason"] != "uncited"],
                                           "reasons": [c["reason"] for c in sc[g][q]["claims"] if not c["supported"]]}
                                          for q in answerable if sc[g][q]["format_ok"] and sc[g][q]["all_claims_supported"] is False]
    (OUT / "metric_audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT / "per_item_claim_support.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "behavior"] + [f"{g}_{k}" for g in GROUPS for k in ("n_claims", "n_supported", "all_supported", "complete", "declines")])
        for q in order:
            w.writerow([q, recs[q]["behavior"]] + [x for g in GROUPS for x in (sc[g][q]["n_factual_claims"], sc[g][q]["n_supported_claims"],
                                                                                sc[g][q]["all_claims_supported"], sc[g][q]["complete"], sc[g][q]["declines"])])
    slim = json.loads(json.dumps(rep))
    for g in GROUPS:
        slim["claim_support"][g]["per_group_denominator(REPORT 口径)"].pop("excluded_no_claims")
        slim["valid_id_unsupported"][g] = len(slim["valid_id_unsupported"][g])
    for k in list(slim["claim_support"]):
        if k.startswith("paired"):
            slim["claim_support"][k]["dropped_items"] = len(slim["claim_support"][k]["dropped_items"])
    print(json.dumps(slim, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
