"""生成 reports/HUMAN_REVIEW.csv：test_v3 每题一行（参考答案审核）+ 主要组别的自动判定与复核标记。

    python -m rag_finetuning.experiments.p6_v3.scripts.make_review --summary-dir rag_finetuning/experiments/p6_v3/results/test_v3

优先级（high 优先审）：
  * 拒答题（确认证据确实不足、参考行为正确）；冲突 / 非冲突题（确认是否真冲突）；部分回答题；
  * 任一主要组被评分器标记 review_flags（拒答但含陈述、两值均引用但无冲突措辞、支持度接近阈值、疑似同义改写）；
  * 主要配对（A/C、D/E）自动判定不一致的题。
人工审核前所有结果为 provisional；审核人填写 human_* 列，不得由 AI 代填。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rag_finetuning.common import read_jsonl

V3 = Path(__file__).resolve().parents[1]
GROUPS = ["A", "B", "C", "D", "E", "F"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", required=True)
    a = ap.parse_args(argv)
    recs = {r["id"]: r for r in read_jsonl(V3 / "data" / "test_v3" / "built" / "test_v3.jsonl")}
    with open(Path(a.summary_dir) / "per_question.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        r = recs[row["question_id"]]
        reasons = []
        if r["behavior"] in ("refuse", "conflict", "partial") or r.get("no_conflict"):
            reasons.append(f"边界题型:{r['behavior']}{'/no_conflict' if r.get('no_conflict') else ''}")
        flags = {g: row.get(f"{g}_review_flags", "") for g in GROUPS if f"{g}_review_flags" in row}
        if any(flags.values()):
            reasons.append("评分器复核标记:" + ",".join(f"{g}={v}" for g, v in flags.items() if v))
        for x, y in (("A", "C"), ("D", "E")):
            if f"{x}_content_ok" in row and f"{y}_content_ok" in row and row[f"{x}_content_ok"] != row[f"{y}_content_ok"]:
                reasons.append(f"{x}/{y} 判定不一致")
        rec = {"question_id": r["id"], "behavior": r["behavior"], "subtype": r["subtype"], "question": r["question"],
               "gold_sources": "; ".join(f"{e['doc']}#{e['section']}" for e in r["evidence"] if e["cid"] in r["gold_cids"]),
               "reference_answer": r["reference_answer"], "required_facts": " / ".join(r["facts"]),
               "review_priority": "high" if reasons else "normal", "priority_reasons": " | ".join(reasons),
               "review_status": "provisional"}
        for g in GROUPS:
            if f"{g}_answer" in row:
                rec[f"{g}_answer"] = row[f"{g}_answer"]
                rec[f"{g}_auto_content_ok"] = row[f"{g}_content_ok"]
                rec[f"{g}_auto_claims_supported"] = row[f"{g}_claims_supported"]
        rec.update({"human_reference_ok(Y/N)": "", "human_key_facts_ok(Y/N)": "", "human_evidence_support_ok(Y/N)": "",
                    "human_refusal_boundary_ok(Y/N)": "", "human_verdict_per_group": "", "reviewer": "", "review_date": "", "notes": ""})
        out.append(rec)
    out.sort(key=lambda x: (x["review_priority"] != "high", x["question_id"]))
    dst = V3 / "reports" / "HUMAN_REVIEW.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("rows", len(out), "high", sum(x["review_priority"] == "high" for x in out))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
