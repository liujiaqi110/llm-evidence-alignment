"""汇总评分器争议案例（中文数字 / 同义表达 / 证据推算数值），只读；不改评分器，不改判。

    python -m rag_finetuning.experiments.p6_v5.diagnostics.collect_disputes

来源：hard_dev 全部 12 组现有输出（评分器 4.0.2）+ Oracle 消融的争议清单（scorer_disputes_review.csv）。
对每个争议给出“如果该规则被采纳”的**上限影响**（该题该组的 multi_complete_v5 / content_ok_v4 可能由 0 变 1），
只用于说明修订的利害，不作为修订依据——是否采纳须由人工审核决定。
输出：diagnostics/scorer_review/disputes_by_category.csv、disputes_summary.json
"""
from __future__ import annotations

import csv
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.experiments.p6_v5.diagnostics.scorer_disputes import disputes_for
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import score_item_v4

V5 = Path(__file__).resolve().parents[1]
OUT = V5 / "diagnostics" / "scorer_review"
CAT = [("中文数字", "中文数字写法"), ("同义表达", "同义表达"), ("推算数值", "推算数字")]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hd = {r["id"]: r for r in read_jsonl(V5 / "data/hard_dev/built/hard_dev.jsonl")}
    rows = []
    for p in sorted(glob.glob("workspace/runs/p6_v5/eval/hard_dev/*/generations.jsonl")):
        g = Path(p).parent.name
        for x in read_jsonl(p):
            r = hd[x["id"]]
            d = disputes_for(r, x["answer"])
            if not d:
                continue
            s = score_item_v4(r, x["answer"])
            reasons = d["reasons"] if isinstance(d["reasons"], list) else [d["reasons"]]
            for name, key in CAT:
                hits = [z for z in reasons if key in z]
                if hits:
                    rows.append({"category": name, "source": "hard_dev", "group": g, "id": x["id"], "behavior": r["behavior"],
                                 "question": r["question"], "required_facts": " / ".join(r["facts"]), "missing_facts": " / ".join(s["missing_facts"]),
                                 "unsupported_numbers": " ".join(s["unsupported_numbers_v4"]), "reason": "；".join(hits),
                                 "multi_complete_v5": s["multi_complete_v5"], "content_ok_v4": s["content_ok_v4"],
                                 "only_this_blocks_multi": (s["multi_complete_v5"] is False and len(s["missing_facts"]) == len(hits) and name != "推算数值"),
                                 "only_this_blocks_content": (not s["content_ok_v4"] and name == "推算数值" and not s["missing_facts"]),
                                 "answer": x["answer"], "reference_answer": r["reference_answer"],
                                 "human_verdict": "", "human_rule_should_apply": "", "human_note": ""})
    orc = V5 / "diagnostics/oracle_ablation/scorer_disputes_review.csv"
    if orc.exists():
        with open(orc, encoding="utf-8-sig") as f:
            for o in csv.DictReader(f):
                if o["source"].startswith("hard_dev 第一轮"):
                    continue  # 已由上面的 12 组全量重算覆盖
                for name, key in CAT:
                    if key in o["reasons"]:
                        rows.append({"category": name, "source": o["source"], "group": f"{o['group_or_model']}|{o['condition']}", "id": o["id"],
                                     "behavior": "", "question": o["question"], "required_facts": o["required_facts"], "missing_facts": o["missing"],
                                     "unsupported_numbers": o["unsupported_numbers"], "reason": o["reasons"], "multi_complete_v5": "", "content_ok_v4": "",
                                     "only_this_blocks_multi": "", "only_this_blocks_content": "", "answer": o["answer"],
                                     "reference_answer": o["reference_answer"], "human_verdict": "", "human_rule_should_apply": "", "human_note": ""})
    with open(OUT / "disputes_by_category.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {"rows": len(rows), "by_category": dict(Counter(r["category"] for r in rows)),
            "by_category_unique_items": {c: len({r["id"] for r in rows if r["category"] == c}) for c, _ in CAT},
            "hard_dev_by_group_and_category": dict(Counter(f"{r['group']}|{r['category']}" for r in rows if r["source"] == "hard_dev")),
            "upper_bound_if_rule_accepted（仅 hard_dev）": {
                "multi_complete_v5_0_to_1": dict(Counter(r["group"] for r in rows if r["only_this_blocks_multi"] is True)),
                "content_ok_v4_0_to_1": dict(Counter(r["group"] for r in rows if r["only_this_blocks_content"] is True))}}
    ex = defaultdict(list)
    for r in rows:
        if len(ex[r["category"]]) < 3:
            ex[r["category"]].append({k: r[k] for k in ("source", "group", "id", "required_facts", "missing_facts", "unsupported_numbers", "reason")})
    summ["examples"] = ex
    (OUT / "disputes_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
