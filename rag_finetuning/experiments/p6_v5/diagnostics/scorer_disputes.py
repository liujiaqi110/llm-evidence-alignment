"""P6-v5 诊断：找出评分器可能误判的题（AI 预审，人工字段留空）。不修改评分器，不改变任何已报告的数字。

候选规则（只用于挑出需要人工看的题，不自动改判）：
  1 中文数字写法：缺失事实含阿拉伯数字，答案中有对应中文数字写法（1张 ↔ 一张、2次 ↔ 两次）；
  2 同义表达：缺失事实为“不变 / 不得 / 不能 / 不予”等，答案中出现等价说法（没有变化、仍然、不可以、不能、禁止）；
  3 推算数字：content 因证据外数字判错，但该数字可由证据和问题中的数字做加减乘得到（例如 8000×30%=2400）；
  4 单一失败事实且答案已给出同义结论的其他情形，由 AI 预审说明。
来源：hard_dev 第一轮 7 组输出 + Oracle 消融 3 模型 × 4 条件。
"""
from __future__ import annotations

import csv
import itertools
import json
import re
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.evaluation.scoring import normalize, numbers
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import score_item_v4

V5 = Path(__file__).resolve().parents[1]
OUT = V5 / "diagnostics" / "oracle_ablation"
CN = {"1": "一", "2": "两|二", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七", "8": "八", "9": "九", "10": "十"}
SYN = {"不变": r"(没有变|未变|没变|仍然|仍为|仍是|保持)", "不得": r"(不可以|不能|禁止|不允许)", "不能": r"(不可以|不得|无法|不允许)",
       "不予": r"(不能|不可以|不会)", "不再重复报销": r"(不能|不可以).{0,6}报"}


def derived(n: str, pool) -> bool:
    try:
        x = float(n)
    except ValueError:
        return False
    vals = [float(p) for p in pool if re.fullmatch(r"\d+(\.\d+)?", p)]
    for a, b in itertools.permutations(vals, 2):
        for v in (a + b, a - b, a * b, a * b / 100):
            if abs(v - x) < 1e-6:
                return True
    return False


def disputes_for(rec, ans):
    s = score_item_v4(rec, ans)
    if s["content_ok_v4"] and s["multi_complete_v5"] is not False:
        return None
    reasons = []
    plain = normalize(ans)
    for f in s["missing_facts"]:
        for alt in f.split("|"):
            m = re.match(r"^(\d+)(.*)$", normalize(alt))
            if m and m.group(1) in CN and re.search(f"({CN[m.group(1)]}){re.escape(m.group(2))}", plain):
                reasons.append(f"中文数字写法：“{alt}”在答案中写成中文数字")
            for k, pat in SYN.items():
                if k in alt and re.search(pat, ans):
                    reasons.append(f"同义表达：缺失“{alt}”，答案有等价说法（{re.search(pat, ans).group(0)}）")
    if s["unsupported_numbers_v4"]:
        pool = set(numbers(" ".join(e["text"] for e in rec["evidence"]))) | set(numbers(rec["question"]))
        der = [n for n in s["unsupported_numbers_v4"] if derived(n, pool)]
        if der and len(der) == len(s["unsupported_numbers_v4"]):
            reasons.append(f"推算数字：{der} 可由证据 / 问题中的数字计算得到")
    if not reasons:
        return None
    return {"missing": " / ".join(s["missing_facts"]), "unsupported_numbers": " ".join(s["unsupported_numbers_v4"]), "reasons": "；".join(dict.fromkeys(reasons))}


def main() -> int:
    recs = {r["id"]: r for r in read_jsonl(V5 / "data/hard_dev/built/hard_dev.jsonl")}
    rows = []
    for g in ("A", "B", "C42", "C43", "D", "A_plan", "D_plan"):
        for x in read_jsonl(V5 / f"results/hard_dev_round1/generations_{g}.jsonl"):
            d = disputes_for(recs[x["id"]], x["answer"])
            if d:
                rows.append({"source": "hard_dev 第一轮", "group_or_model": g, "condition": "原始 RAG", "id": x["id"], **d, "answer": x["answer"]})
    runs = Path("workspace/runs/p6_v5/diagnostics/oracle_ablation")
    for m in ("A", "C42", "D"):
        for c in ("B", "C", "D", "Brev"):
            p = runs / m / c / "generations.jsonl"
            if not p.exists():
                continue
            for x in read_jsonl(p):
                r = recs[x["id"]]
                ev = [e for e in r["evidence"] if e["cid"] in set(x["evidence_cids"])]
                d = disputes_for(dict(r, evidence=ev), x["answer"])
                if d:
                    rows.append({"source": "Oracle 消融", "group_or_model": m, "condition": c, "id": x["id"], **d, "answer": x["answer"]})
    for r in rows:
        rec = recs[r["id"]]
        r.update({"question": rec["question"], "required_facts": " / ".join(rec["facts"]), "reference_answer": rec["reference_answer"],
                  "ai_prereview": "AI 预审（非人工）：疑似评分器误判，答案可能已正确表达该事实；需人工确认",
                  "human_verdict": "", "human_note": "", "review_status": "待人工审核"})
    cols = ["source", "group_or_model", "condition", "id", "question", "required_facts", "reference_answer", "answer", "missing",
            "unsupported_numbers", "reasons", "ai_prereview", "human_verdict", "human_note", "review_status"]
    with open(OUT / "scorer_disputes_review.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    summ = {"n_rows": len(rows), "n_items": len({r["id"] for r in rows}),
            "by_reason_kind": {k: sum(1 for r in rows if k in r["reasons"]) for k in ("中文数字写法", "同义表达", "推算数字")},
            "items": sorted({r["id"] for r in rows})}
    (OUT / "scorer_disputes_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
