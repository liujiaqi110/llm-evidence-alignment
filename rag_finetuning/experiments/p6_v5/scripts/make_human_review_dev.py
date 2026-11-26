"""生成 hard_dev 人工审核清单 HUMAN_REVIEW_CHECKLIST.csv（AI 预审单列，所有人工字段留空；human_approved 必须由人工填写）。

    python -m rag_finetuning.experiments.p6_v5.scripts.make_human_review_dev
"""
from __future__ import annotations

import csv
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.experiments.p6_v5.diagnostics.scorer_disputes import disputes_for
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import SCORER_VERSION, score_item_v4

V5 = Path(__file__).resolve().parents[1]
RUNS = Path("workspace/runs/p6_v5/eval/hard_dev")
SHOW = [("C42", "3B v4 s42 原始 RAG"), ("D", "7B 原始 RAG"), ("A", "3B Base 原始 RAG")]
ALL = ["A", "B", "C42", "C43", "D", "A_plan", "D_plan", "C42_plan", "C42_pv2", "D_pv2", "C42_pv21", "D_pv21"]


def fact_sentence(fact, text):
    import re
    from rag_finetuning.evaluation.scoring import normalize
    for s in re.split(r"(?<=[。；])", text.split("\n", 1)[-1]):
        if any(a and normalize(a) in normalize(s) for a in fact.split("|")):
            return s.strip()
    return ""


def main() -> int:
    recs = read_jsonl(V5 / "data/hard_dev/built/hard_dev.jsonl")
    gens = {g: {x["id"]: x for x in read_jsonl(RUNS / g / "generations.jsonl")} for g in ALL if (RUNS / g / "generations.jsonl").exists()}
    rows = []
    for r in recs:
        gold = set(r["gold_cids"])
        by = {e["cid"]: e["text"] for e in r["evidence"]}
        flags, disputed_groups = [], []
        for g, G in gens.items():
            d = disputes_for(r, G[r["id"]]["answer"])
            if d:
                disputed_groups.append(g)
                flags.append(f"{g}：{d['reasons']}")
        row = {"priority": "", "id": r["id"], "behavior": r["behavior"], "subtype": r.get("subtype") or "", "conflict_kind": r.get("conflict_kind") or "",
               "tags": ",".join(r.get("hard_tags") or []), "question": r["question"],
               "evidence_full_text": "\n\n".join(f"[{e['cid']}]{'（金标准）' if e['cid'] in gold else ''} {e['text']}" for e in r["evidence"]),
               "required_facts_with_source": "\n".join(f"{f} ← [{c}] {fact_sentence(f, by.get(c, ''))}" for f, c in zip(r["facts"], r.get("fact_cids") or [])),
               "reference_answer": r["reference_answer"], "forbid_rules": " / ".join(r.get("forbid") or [])}
        for g, label in SHOW:
            s = score_item_v4(r, gens[g][r["id"]]["answer"])
            row[f"{g}_answer（{label}）"] = gens[g][r["id"]]["answer"]
            row[f"{g}_auto"] = (f"内容={int(s['content_ok_v4'])} 多证据={s['multi_complete_v5']} 缺失={'/'.join(s['missing_facts']) or '无'} "
                                f"证据外数字={' '.join(s['unsupported_numbers_v4']) or '无'} 支持={s['n_supported']}/{s['n_claims']}")
        row.update({
            "ai_prereview_flags（非人工）": "\n".join(flags) or "",
            "ai_prereview_note（非人工）": ("疑似评分器误判：答案可能已用同义 / 中文数字 / 推算方式表达了必要事实；需人工确认" if flags else ""),
            "groups_with_dispute": ",".join(disputed_groups),
            "human_reference_answer_correct": "", "human_required_facts_complete_and_necessary": "", "human_question_unambiguous": "",
            "human_verdict_C42": "", "human_verdict_D": "", "human_scorer_error": "", "human_note": "", "human_reviewer": "", "human_review_date": "",
            "human_approved": "", "review_status": "pending_human_review（AI 预审 ≠ 人工审核）", "scorer_version": SCORER_VERSION})
        row["priority"] = "高" if flags or r["behavior"] in ("conflict", "partial") or r.get("no_conflict") else "中"
        rows.append(row)
    rows.sort(key=lambda x: (x["priority"] != "高", x["id"]))
    with open(V5 / "HUMAN_REVIEW_CHECKLIST.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "行；高优先级", sum(r["priority"] == "高" for r in rows), "；有争议标记", sum(bool(r["groups_with_dispute"]) for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
