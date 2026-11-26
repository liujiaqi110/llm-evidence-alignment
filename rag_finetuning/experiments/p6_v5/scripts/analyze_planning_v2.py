"""P6-v5 Planning v2 开发集分析（只在模型输出落盘后读取标注做评分；不参与推理）。

对照：A = C42 原始 RAG（第一轮）  B = C42 + 现有证据覆盖生成（v1）  C = C42 + Planning v2
      D = 7B 原始 RAG（第一轮）    E = 7B + Planning v2
输出：PLANNING_V2_PER_QUESTION.csv、results/planning_v2/summary.json（提取召回 / 精确、失败归因、成本、集中度）

    python -m rag_finetuning.experiments.p6_v5.scripts.analyze_planning_v2
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.evaluation.scoring import normalize
from rag_finetuning.experiments.p6_v4.scripts.scoring_v3 import fact_hit
from rag_finetuning.experiments.p6_v5.diagnostics.scorer_disputes import disputes_for
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import score_item_v4

V5 = Path(__file__).resolve().parents[1]
RUNS = Path("workspace/runs/p6_v5/eval/hard_dev")
COND = {"A": ("C42", "3B v4 s42 原始 RAG"), "B": ("C42_plan", "3B v4 s42 + 现有证据覆盖生成"), "C": ("C42_pv2", "3B v4 s42 + Planning v2"),
        "D": ("D", "7B 原始 RAG"), "E": ("D_pv2", "7B + Planning v2"),
        "F": ("C42_pv21", "3B v4 s42 + Planning v2.1（开发集诊断后修复）"), "G": ("D_pv21", "7B + Planning v2.1（开发集诊断后修复）")}
PLAN_CONDS = ("B", "C", "E", "F", "G")
OUT = V5 / "results" / "planning_v2"


def extracted_text(facts):
    return normalize(" ".join(f["fact_text"] + " " + (f.get("support_span") or "") for f in facts))


def cause_of(rec, s, g, plan):
    """多证据题失败的主因（按优先级取一个）；同时返回逐事实明细。"""
    if s["multi_complete_v5"] and s["content_ok_v4"]:
        return "正确", []
    if disputes_for(rec, g["answer"]):
        return "评分器误判（疑似，见争议预审）", []
    if rec.get("no_conflict") and s.get("conflict_claimed"):
        return "条件 / 版本 / 范围理解错误", []
    if not s["missing_facts"] and not s["content_ok_v4"]:
        return "条件 / 版本 / 范围理解错误" if not (s["unsupported_numbers_v4"] or s["forbidden_hits"]) else "生成器错误组合事实", []
    if s["unsupported_numbers_v4"] or s["forbidden_hits"]:
        return "生成器错误组合事实", []
    if plan is None:
        return "事实遗漏（无规划器，无法区分）", []
    ext = extracted_text(plan["facts"])
    det = [{"fact": f, "extracted": bool(fact_hit(ext, f))} for f in s["missing_facts"]]
    if any(not d["extracted"] for d in det):
        return "原文存在，但规划器未提取", det
    return "规划器提取正确，但生成器遗漏", det


def main() -> int:
    recs = {r["id"]: r for r in read_jsonl(V5 / "data/hard_dev/built/hard_dev.jsonl")}
    gens = {}
    for c, (g, _) in list(COND.items()):
        if not (RUNS / g / "generations.jsonl").exists():
            COND.pop(c)
            continue
        gens[c] = {x["id"]: x for x in read_jsonl(RUNS / g / "generations.jsonl")}
    multi = [q for q, r in recs.items() if score_item_v4(r, r["reference_answer"])["is_multi_v5"]]
    rows, S = [], defaultdict(dict)
    for q, r in recs.items():
        for c in COND:
            S[c][q] = score_item_v4(r, gens[c][q]["answer"])
    ext_stats = {}
    for c in [x for x in PLAN_CONDS if x in COND]:
        rec_hits, n_gold, n_ext, n_ext_gold_chunk, n_ext_match, n_kept, n_drop, irrelevant_used = 0, 0, 0, 0, 0, 0, 0, 0
        per_item_recall = []
        for q, r in recs.items():
            if r["behavior"] == "refuse":
                continue
            plan = gens[c][q].get("plan") or {}
            facts, dropped = plan.get("facts", []), plan.get("dropped", [])
            ext = extracted_text(facts)
            hits = sum(1 for f in r["facts"] if fact_hit(ext, f))
            rec_hits += hits
            n_gold += len(r["facts"])
            per_item_recall.append(hits / len(r["facts"]) if r["facts"] else 1.0)
            gold = set(r["gold_cids"])
            n_ext += len(facts)
            n_ext_gold_chunk += sum(1 for f in facts if f["source_chunk_id"] in gold)
            n_ext_match += sum(1 for f in facts if any(fact_hit(normalize(f["fact_text"] + " " + (f.get("support_span") or "")), g) for g in r["facts"]))
            n_kept += len(facts)
            n_drop += len(dropped)
        ext_stats[c] = {"required_fact_extraction_recall": round(rec_hits / n_gold, 3), "n_required_facts": n_gold,
                        "items_with_full_extraction": sum(1 for x in per_item_recall if x >= 0.999), "n_answerable_items": len(per_item_recall),
                        "extracted_facts": n_ext, "precision_chunk_level(来源为金标准片段)": round(n_ext_gold_chunk / n_ext, 3) if n_ext else None,
                        "precision_fact_level(含某条必要事实)": round(n_ext_match / n_ext, 3) if n_ext else None,
                        "grounded_ratio(通过原文核对的比例)": round(n_kept / (n_kept + n_drop), 3) if (n_kept + n_drop) else None,
                        "dropped_by_reason": dict(Counter(d.get("drop_reason") for q in recs for d in (gens[c][q].get("plan") or {}).get("dropped", [])))}
        if c in ("C", "E", "F", "G"):
            m_rec = [sum(1 for f in recs[q]["facts"] if fact_hit(extracted_text(gens[c][q]["plan"]["facts"]), f)) / len(recs[q]["facts"]) for q in multi]
            ext_stats[c]["multi_items_extraction_recall_mean"] = round(statistics.mean(m_rec), 3)
            ext_stats[c]["multi_items_full_extraction"] = sum(1 for x in m_rec if x >= 0.999)
    causes = {c: Counter() for c in COND}
    for q in multi:
        r = recs[q]
        row = {"id": q, "question": r["question"], "tags": ",".join(r.get("hard_tags") or []), "subtype": r.get("subtype") or "",
               "source_docs": ",".join(sorted({e["doc"] for e in r["evidence"] if e["cid"] in set(r["gold_cids"])})),
               "required_facts": " / ".join(r["facts"])}
        for c in COND:
            s, g = S[c][q], gens[c][q]
            plan = g.get("plan") if c in ("C", "E", "F", "G") else None
            cause, det = cause_of(r, s, g, plan)
            causes[c][cause] += 1
            row.update({f"{c}_multi": int(bool(s["multi_complete_v5"])), f"{c}_content": int(s["content_ok_v4"]),
                        f"{c}_missing": " / ".join(s["missing_facts"]), f"{c}_cause": cause,
                        f"{c}_missing_detail": "；".join(f"{d['fact']}:{'已提取' if d['extracted'] else '未提取'}" for d in det),
                        f"{c}_support": f"{s['n_supported']}/{s['n_claims']}", f"{c}_latency_ms": g.get("latency_ms"), f"{c}_answer": g["answer"]})
            if plan is not None:
                row[f"{c}_extract_recall"] = round(sum(1 for f in r["facts"] if fact_hit(extracted_text(plan["facts"]), f)) / len(r["facts"]), 3)
                row[f"{c}_n_facts"] = len(plan["facts"])
                row[f"{c}_revised"] = plan["check"].get("revised")
        row.update({"review_status": "provisional（自动评分）", "human_verdict": "", "human_note": ""})
        rows.append(row)
    with open(V5 / "PLANNING_V2_PER_QUESTION.csv", "w", encoding="utf-8-sig", newline="") as f:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # 汇总
    allq = list(recs)
    summ = {"note": "自动规则评分（v4.0.1），provisional；hard_dev 为开发集，不作为泛化证据", "conditions": {c: d for c, (g, d) in COND.items()}, "metrics": {},
            "extraction": ext_stats, "causes_on_multi": {c: dict(v) for c, v in causes.items()}, "transitions": {}, "concentration": {}, "cost": {}}
    for c in COND:
        s = S[c]
        ans = [q for q in allq if recs[q]["behavior"] in ("answer", "partial", "conflict")]
        summ["metrics"][c] = {
            "multi_complete_v5": f"{sum(bool(s[q]['multi_complete_v5']) for q in multi)}/{len(multi)}",
            "content_ok_v4": f"{sum(s[q]['content_ok_v4'] for q in allq)}/{len(allq)}",
            "required_fact_coverage_mean": round(statistics.mean(s[q]["required_fact_coverage"] or 0 for q in allq if recs[q]["behavior"] != "refuse"), 3),
            "claim_support_full": f"{sum(bool(s[q]['claim_support_full']) for q in ans)}/{len(ans)}",
            "citation_accuracy": f"{sum(s[q]['citation_accuracy'] is True for q in ans)}/{sum(s[q]['citation_accuracy'] is not None for q in ans)}",
            "citation_valid": f"{sum(bool(s[q]['citation_valid']) for q in allq)}/{len(allq)}",
            "unsupported_inference_items(证据外数字或禁止内容)": sum(bool(s[q]["unsupported_numbers_v4"] or s[q]["forbidden_hits"]) for q in allq),
            "whole_refusal_on_answerable": f"{sum(bool(s[q]['whole_refusal_on_answerable']) for q in ans)}/{len(ans)}",
            "refusal_ok": f"{sum(bool(s[q]['refusal_ok']) for q in allq if recs[q]['behavior'] == 'refuse')}/{sum(1 for q in allq if recs[q]['behavior'] == 'refuse')}",
            "conflict_true_ok": f"{sum(bool(s[q]['conflict_true_ok']) for q in allq if recs[q]['behavior'] == 'conflict')}/{sum(1 for q in allq if recs[q]['behavior'] == 'conflict')}",
            "no_conflict_ok": f"{sum(bool(s[q]['no_conflict_ok']) for q in allq if recs[q].get('no_conflict'))}/{sum(1 for q in allq if recs[q].get('no_conflict'))}",
            "injection_resisted": f"{sum(bool(s[q]['injection_resisted']) for q in allq if recs[q].get('subtype') == 'prompt_injection')}/2"}
        if c in ("F", "G"):     # v2.1 保存了修订前答案：分别计算“生成阶段”与“最终（核验修订后）”
            st2 = {q: score_item_v4(recs[q], (gens[c][q]["plan"]["check"].get("stage2_answer") or gens[c][q]["answer"])) for q in allq}
            summ["metrics"][c]["stage2_multi_complete_v5(修订前)"] = f"{sum(bool(st2[q]['multi_complete_v5']) for q in multi)}/{len(multi)}"
            summ["metrics"][c]["stage2_content_ok_v4(修订前)"] = f"{sum(st2[q]['content_ok_v4'] for q in allq)}/{len(allq)}"
        g = gens[c]
        lat = sorted(g[q].get("latency_ms") or 0 for q in allq)
        stage = defaultdict(list)
        for q in allq:
            for x in (g[q].get("plan") or {}).get("calls", []):
                stage[x["stage"]].append(x["latency_ms"])
        summ["cost"][c] = {"latency_ms_p50": statistics.median(lat), "latency_ms_p95": lat[int(0.95 * (len(lat) - 1))],
                           "latency_ms_mean": round(statistics.mean(lat), 1), "calls_mean": round(statistics.mean(g[q].get("n_calls") or 1 for q in allq), 2),
                           "input_tokens_mean": round(statistics.mean(g[q].get("input_tokens") or 0 for q in allq), 1),
                           "output_tokens_mean": round(statistics.mean(g[q].get("output_tokens") or 0 for q in allq), 1),
                           "stage_latency_ms_total_mean": {k: round(sum(v) / len(allq), 1) for k, v in stage.items()},
                           "revised_items": sum(1 for q in allq if ((g[q].get("plan") or {}).get("check") or {}).get("revised"))}
    for x, y in [p for p in (("A", "B"), ("A", "C"), ("B", "C"), ("D", "E"), ("C", "E"), ("A", "F"), ("C", "F"), ("D", "G"), ("E", "G"), ("F", "G"))
                 if p[0] in COND and p[1] in COND]:
        okx = {q for q in multi if S[x][q]["multi_complete_v5"]}
        oky = {q for q in multi if S[y][q]["multi_complete_v5"]}
        summ["transitions"][f"{x}→{y}"] = {"fixed": sorted(oky - okx), "broken": sorted(okx - oky), "net": len(oky) - len(okx)}
    for x, y in [p for p in (("A", "C"), ("D", "E"), ("A", "F"), ("D", "G")) if p[0] in COND and p[1] in COND]:
        by_doc, by_tag = Counter(), Counter()
        for q in summ["transitions"][f"{x}→{y}"]["fixed"]:
            for d in next(r for r in rows if r["id"] == q)["source_docs"].split(","):
                by_doc[d] += 1
            for t in (recs[q].get("hard_tags") or ["(无)"]):
                by_tag[t] += 1
        summ["concentration"][f"{x}→{y}"] = {"fixed_by_doc": dict(by_doc.most_common()), "fixed_by_tag": dict(by_tag.most_common())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("metrics", "extraction", "causes_on_multi", "transitions", "cost", "concentration")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
