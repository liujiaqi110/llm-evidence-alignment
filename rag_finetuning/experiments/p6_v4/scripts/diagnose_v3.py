"""v3 多证据 / 拒答 / 冲突 / 非冲突失败的逐题诊断证据（3B Base = D，3B QLoRA v3 = E；F 7B 作参考）。

    python -m rag_finetuning.experiments.p6_v4.scripts.diagnose_v3

对每道题保存：问题、完整证据片段、金标准、必要事实及其支持片段（逐字定位）、各组回答、各组命中的事实与引用，
以及“自动可验证的诊断依据”（auto_basis）。最终根因类别（A–H）由 AI 助手在此基础上逐条标注到
results/diagnosis/diagnosis_labels.json —— 属于 AI 诊断，不是人工结论。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.evaluation.scoring import fact_hits, normalize
from rag_finetuning.experiments.p6_v3.scripts.scoring_v2 import score_item_v2

V3 = Path(__file__).resolve().parents[2] / "p6_v3"
OUT = Path(__file__).resolve().parents[1] / "results" / "diagnosis"
EVAL = Path("workspace/runs/p6_v3/eval/test_v3")
GROUPS = ["D", "E", "F"]


def fact_chunks(rec):
    out = []
    for f in rec["facts"]:
        alts = [normalize(a) for a in f.split("|")]
        where = [e["cid"] for e in rec["evidence"] if any(a in normalize(e["text"]) for a in alts)]
        out.append({"fact": f, "in_chunks": where, "in_gold": [c for c in where if c in rec["gold_cids"]]})
    return out


def main() -> int:
    recs = {r["id"]: r for r in read_jsonl(V3 / "data" / "test_v3" / "built" / "test_v3.jsonl")}
    G = {g: {x["id"]: x["answer"] for x in read_jsonl(EVAL / g / "generations.jsonl")} for g in GROUPS}
    sel = {}
    for q, r in recs.items():
        if len(r["gold_cids"]) >= 2 and r["behavior"] != "refuse" and not r.get("no_conflict"):
            sel[q] = "multi_evidence"
        if r["behavior"] == "refuse":
            sel.setdefault(q, "refusal")
        if r["behavior"] == "conflict":
            sel[q] = "conflict"
        if r.get("no_conflict"):
            sel[q] = "no_conflict"
    items = []
    for q, kind in sorted(sel.items()):
        r = recs[q]
        fc = fact_chunks(r)
        row = {"id": q, "kind": kind, "behavior": r["behavior"], "subtype": r["subtype"], "question": r["question"],
               "evidence": [{"cid": e["cid"], "doc": e["doc"], "section": e["section"], "is_gold": e["cid"] in r["gold_cids"],
                             "text": e["text"]} for e in r["evidence"]],
               "reference_answer": r["reference_answer"], "required_facts": fc, "groups": {}}
        for g in GROUPS:
            a = G[g][q]
            s = score_item_v2(r, a)
            hits = fact_hits(a, r["facts"])
            gold_cov = {c: [f["fact"] for f, h in zip(fc, hits) if h and c in f["in_gold"]] for c in r["gold_cids"]}
            basis = []
            if any(not f["in_chunks"] for f in fc) and r["behavior"] != "refuse":
                basis.append("某必要事实不在任何证据片段中（计算值或参考答案问题）")
            missing_chunks = [c for c in r["gold_cids"] if not any(c in f["in_gold"] for f, h in zip(fc, hits) if h)]
            if r["behavior"] in ("answer", "partial") and missing_chunks and any(gold_cov.values()):
                basis.append(f"答到了部分金标准片段的事实，但金标准片段 {missing_chunks} 的事实一个也没答到")
            if s["complete"] and (s["citation_recall"] or 0) < 1:
                basis.append("必要事实全部命中，但漏引了金标准片段 → 失败来自引用召回而非内容")
            if s["declines"] and r["behavior"] != "refuse" and (s["fact_recall"] or 0) < 1:
                basis.append("出现拒答 / 缺失声明且事实不完整")
            if r["behavior"] == "refuse" and not s["behavior_ok"]:
                basis.append("应拒答题未拒答" + (f"；出现事实陈述 {len(s['claims'])} 条" if s["claims"] else ""))
            row["groups"][g] = {"answer": a, "content_ok": s["content_ok"], "fact_hits": dict(zip(r["facts"], hits)),
                                "cited": s["valid_citations"], "citation_recall": s["citation_recall"], "declines": s["declines"],
                                "conflict_claimed": s["conflict_claimed"], "auto_basis": basis}
        items.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnosis_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    for it in items:
        if it["kind"] == "refusal" and all(it["groups"][g]["content_ok"] for g in "DE"):
            continue
        d, e = it["groups"]["D"], it["groups"]["E"]
        print(f"## {it['id']} [{it['kind']}/{it['subtype']}] D={int(d['content_ok'])} E={int(e['content_ok'])} | {it['question']}")
        print("   facts:", [(f['fact'], f['in_gold']) for f in it["required_facts"]])
        for g in "DE":
            x = it["groups"][g]
            print(f"   {g}: hits={[k for k, v in x['fact_hits'].items() if v]} cited={x['cited']} basis={x['auto_basis']}")
            print(f"      {x['answer'][:150]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
