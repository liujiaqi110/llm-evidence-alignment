"""生成 HUMAN_REVIEW_V4.csv：test_v3 的可人工审核材料包（可用 Excel 打开，UTF-8 BOM）。

    python -m rag_finetuning.experiments.p6_v4.scripts.make_review_v4

每题包含：原始问题、完整证据片段（含 Chunk ID、文档标题、小节、版本信息）、参考答案、required_facts、
每个事实的 supporting_evidence（逐字定位到的原句）、contradictory_evidence（同属性不同取值的金标准句）、
answerability、类别、3B Base（D）与 3B QLoRA v3（E）的回答（附 7B 参考）、自动评分（v2）、AI 预审标记，
以及留空的人工审核字段。排序：v3 HUMAN_REVIEW.csv 中的 70 条 high 在前。
人工审核字段不得由 AI 代填；未审核前所有结果为 provisional。
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.evaluation.scoring import fact_hits, normalize
from rag_finetuning.experiments.p6_v3.scripts.scoring_v2 import score_item_v2

V3 = Path(__file__).resolve().parents[2] / "p6_v3"
V4 = Path(__file__).resolve().parents[1]
EVAL = Path("workspace/runs/p6_v3/eval/test_v3")
SENT = re.compile(r"[^。！？\n]+[。！？]?")


def version_of(text: str) -> str:
    m = re.search(r"自\s*(\d{4}\s*年\s*\d+\s*月\s*\d+\s*日)\s*起", text)
    if m:
        return f"修订通知，生效 {m.group(1)}" + ("；原规定同时废止" if "废止" in text else "")
    if "仅适用于" in text:
        return "补充规定（限定适用范围）"
    if text.split("\n", 1)[-1].startswith("通知"):
        return "通知（未注明生效日期 / 优先级）"
    return "现行制度（未注明版本）"


def sentence_with(text: str, fact: str) -> str:
    body = text.split("\n", 1)[-1]
    for s in SENT.findall(body):
        if any(normalize(a) in normalize(s) for a in fact.split("|")):
            return s.strip()
    return ""


def main() -> int:
    recs = {r["id"]: r for r in read_jsonl(V3 / "data" / "test_v3" / "built" / "test_v3.jsonl")}
    with open(V3 / "reports" / "HUMAN_REVIEW.csv", encoding="utf-8-sig") as f:
        old = {r["question_id"]: r for r in csv.DictReader(f)}
    labels = json.loads((V4 / "results" / "diagnosis" / "diagnosis_labels.json").read_text(encoding="utf-8"))
    diag = {}
    for it in labels["items"]:
        diag.setdefault(it["id"], []).append(f"[{it['group']}:{it['cat']}] {it['basis']}")
    G = {g: {x["id"]: x["answer"] for x in read_jsonl(EVAL / g / "generations.jsonl")} for g in "DEF"}
    rows = []
    for q, r in recs.items():
        ev_txt = "\n\n".join(f"[{e['cid']}]{'（金标准）' if e['cid'] in r['gold_cids'] else ''} 《{e['text'].split(' > ')[0]}》"
                             f" {e['doc']}#{e['section']}｜版本：{version_of(e['text'])}\n{e['text']}" for e in r["evidence"])
        support, flags = [], []
        for f in r["facts"]:
            locs = [(e["cid"], sentence_with(e["text"], f)) for e in r["evidence"] if sentence_with(e["text"], f)]
            gold_locs = [x for x in locs if x[0] in r["gold_cids"]]
            if not locs:
                flags.append(f"必要事实“{f}”不在任何证据原文中（计算值，或参考答案需复核）")
            support.append(f"{f} ⇐ " + ("；".join(f"[{c}] {s}" for c, s in (gold_locs or locs)) or "（未定位）"))
        contra = "无"
        if r["behavior"] == "conflict" or r.get("no_conflict"):
            golds = [e for e in r["evidence"] if e["cid"] in r["gold_cids"]]
            contra = "\n".join(f"[{e['cid']}] {e['text'].split(chr(10), 1)[-1]}" for e in golds)
        sc = {g: score_item_v2(r, G[g][q]) for g in "DE"}
        if r["behavior"] == "refuse":
            for e in r["evidence"]:
                for p in r.get("forbid") or []:
                    if not p.startswith("(?") and re.search(p, e["text"]):
                        flags.append(f"禁止模式 {p} 出现在证据 {e['cid']} 中，需确认是否真的无答案")
        flags += diag.get(q, [])
        prio = old.get(q, {}).get("review_priority", "normal")
        rows.append({
            "review_order_priority": "1-high(v3)" if prio == "high" else "2-normal",
            "question_id": q, "question": r["question"], "category": f"{r['type']}/{r['subtype'] or 'plain'}",
            "behavior_expected": r["behavior"], "answerability": r["answerability"], "no_conflict": r.get("no_conflict", False),
            "evidence_full_text": ev_txt, "gold_chunk_ids": " ".join(r["gold_cids"]), "reference_answer": r["reference_answer"],
            "required_facts": " / ".join(r["facts"]), "supporting_evidence_per_fact": "\n".join(support),
            "contradictory_evidence": contra,
            "base_3b_answer(D)": G["D"][q], "qlora_3b_v3_answer(E)": G["E"][q], "reference_7b_answer(F)": G["F"][q],
            "auto_v2_D_content_ok": int(sc["D"]["content_ok"]), "auto_v2_E_content_ok": int(sc["E"]["content_ok"]),
            "auto_v2_D_fact_hits": "".join("1" if h else "0" for h in fact_hits(G["D"][q], r["facts"])),
            "auto_v2_E_fact_hits": "".join("1" if h else "0" for h in fact_hits(G["E"][q], r["facts"])),
            "auto_v2_E_claims_supported": f"{sc['E']['n_supported_claims']}/{sc['E']['n_factual_claims']}",
            "ai_prereview_flags(非人工)": "\n".join(flags),
            "review_status": "provisional",
            "human_reference_correct(Y/N)": "", "human_facts_complete_and_supported(Y/N)": "",
            "human_answerability_correct(Y/N)": "", "human_conflict_label_correct(Y/N/NA)": "",
            "human_D_correct(Y/N/Partial)": "", "human_E_correct(Y/N/Partial)": "",
            "human_reviewer": "", "human_review_date": "", "human_comments": ""})
    rows.sort(key=lambda x: (x["review_order_priority"], x["question_id"]))
    dst = V4 / "HUMAN_REVIEW_V4.csv"
    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_flag = sum(1 for x in rows if x["ai_prereview_flags(非人工)"])
    print("rows", len(rows), "high", sum(x["review_order_priority"].startswith("1") for x in rows), "with_ai_flags", n_flag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
