"""用统一的评分器 4.0.2 重新评估 hard_dev 上的全部 P6-v5 实验组，并与 4.0.1 逐题对照（只读已有输出，不重新生成）。

    python -m rag_finetuning.experiments.p6_v5.scripts.rescore_402
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import score_item_v4

V5 = Path(__file__).resolve().parents[1]
RUNS = Path("workspace/runs/p6_v5")
OUT = V5 / "results" / "scorer_402"
DEV_GROUPS = ["A", "B", "C42", "C43", "D", "A_plan", "D_plan", "C42_plan", "C42_pv2", "D_pv2", "C42_pv21", "D_pv21"]


def main() -> int:
    recs = {r["id"]: r for r in read_jsonl(V5 / "data/hard_dev/built/hard_dev.jsonl")}
    sets = []
    for g in DEV_GROUPS:
        p = RUNS / "eval/hard_dev" / g / "generations.jsonl"
        if p.exists():
            sets.append((f"hard_dev:{g}", [(recs[x["id"]], x["answer"]) for x in read_jsonl(p)]))
    for m in ("A", "C42", "D"):
        for c in ("B", "C", "D", "Brev"):
            p = RUNS / "diagnostics/oracle_ablation" / m / c / "generations.jsonl"
            if p.exists():
                rows = []
                for x in read_jsonl(p):
                    r = recs[x["id"]]
                    ev = [e for e in r["evidence"] if e["cid"] in set(x["evidence_cids"])]
                    rows.append((dict(r, evidence=ev), x["answer"]))
                sets.append((f"oracle:{m}|{c}", rows))
    summ, changes = {}, []
    for name, rows in sets:
        agg = {}
        for ver in ("4.0.1", "4.0.2"):
            S = [score_item_v4(r, a, scorer=ver) for r, a in rows]
            agg[ver] = {"n": len(S), "multi_complete_v5": sum(bool(s["multi_complete_v5"]) for s in S),
                        "multi_n": sum(bool(s["is_multi_v5"]) for s in S), "content_ok_v4": sum(s["content_ok_v4"] for s in S),
                        "unsupported_number_items": sum(bool(s["unsupported_numbers_v4"]) for s in S),
                        "items_with_internal_fact_ids": sum(bool(s.get("internal_fact_ids")) for s in S)}
            if ver == "4.0.1":
                old = S
            else:
                for (r, a), so, sn in zip(rows, old, S):
                    if so["content_ok_v4"] != sn["content_ok_v4"] or so["multi_complete_v5"] != sn["multi_complete_v5"]:
                        changes.append({"set": name, "id": r["id"], "content_401": int(so["content_ok_v4"]), "content_402": int(sn["content_ok_v4"]),
                                        "multi_401": so["multi_complete_v5"], "multi_402": sn["multi_complete_v5"],
                                        "unsupported_401": " ".join(so["unsupported_numbers_v4"]), "unsupported_402": " ".join(sn["unsupported_numbers_v4"]),
                                        "internal_ids": " ".join(sn["internal_fact_ids"]), "answer": a})
        summ[name] = agg
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"note": "同一批已有输出，分别用 4.0.1 与 4.0.2 评分；只读，不重新生成", "sets": summ},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT / "changed_items.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(changes[0]) if changes else ["set"])
        w.writeheader()
        w.writerows(changes)
    for k, v in summ.items():
        a, b = v["4.0.1"], v["4.0.2"]
        print(f"{k:22s} 多证据 {a['multi_complete_v5']}→{b['multi_complete_v5']}/{b['multi_n']}  内容 {a['content_ok_v4']}→{b['content_ok_v4']}/{b['n']}  "
              f"证据外数字题 {a['unsupported_number_items']}→{b['unsupported_number_items']}  含内部编号 {b['items_with_internal_fact_ids']}")
    print("变化记录", len(changes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
