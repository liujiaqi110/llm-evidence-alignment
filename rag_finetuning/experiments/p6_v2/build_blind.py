"""构建 P6-v2 独立盲测集：blind/blind.yaml + blind/corpus/ → blind/built/blind.jsonl + 人工审核表。

    python -m rag_finetuning.experiments.p6_v2.build_blind

复用 P6-v1 的 build_record / derive_refusal / self_check（参考答案必须通过评分器自检），不修改其代码。
额外校验：盲测文档名、片段文本与 train / val / test / Few-shot 示例全部不重叠；问题与训练、诊断集问题的相似度。
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from rag_finetuning.common import DATASET_VERSION, chunk_document, load_corpus, write_jsonl
from rag_finetuning.experiments.p6_v2.prompts import demo_docs
from rag_finetuning.scripts import build_dataset as bd

HERE = Path(__file__).resolve().parent / "blind"
BLIND_EXPLICIT_ONLY = {"admin_bulletin", "partner_quote_email"}


def main() -> int:
    old = load_corpus()
    chunks = {}
    for p in sorted((HERE / "corpus").glob("*.md")):
        for c in chunk_document(p, "blind"):
            chunks[f"{c.doc}#{c.section}"] = c
    blind_docs = {c.doc for c in chunks.values()}
    old_docs = {c.doc for c in old.values()}
    problems = []
    if blind_docs & old_docs:
        problems.append(f"盲测文档与已有文档重名：{sorted(blind_docs & old_docs)}")
    if blind_docs & demo_docs():
        problems.append("盲测文档与 Few-shot 文档重叠")
    if {c.text for c in chunks.values()} & {c.text for c in old.values()}:
        problems.append("盲测片段文本与已有片段重复")
    bd.EXPLICIT_ONLY = set(bd.EXPLICIT_ONLY) | BLIND_EXPLICIT_ONLY      # 冲突 / 注入来源不作随机干扰片段
    items = yaml.safe_load((HERE / "blind.yaml").read_text(encoding="utf-8"))
    rows = [bd.build_record(it, "blind", chunks) for it in items]
    rows += [bd.derive_refusal(r, chunks, "blind") for r, it in zip(list(rows), items) if it.get("derive_refusal")]
    for r in rows:
        r["dataset_version"] = DATASET_VERSION + "+p6v2-blind-1.0"
    problems += bd.self_check({"blind": rows})
    # 问题相似度：与训练集、诊断集（test）问题
    prev_q = []
    for s in ("train", "val", "test"):
        prev_q += [json.loads(ln)["question"] for ln in open(bd.BUILT / f"{s}.jsonl", encoding="utf-8")]
    sims = sorted(((max(bd.jaccard(r["question"], q) for q in prev_q), r["id"], r["question"])
                   for r in rows if r["generation_method"] == "claude_authored"), reverse=True)
    out = HERE / "built"
    write_jsonl(out / "blind.jsonl", rows)
    stats = {"n": len(rows), "unique_questions": len({r["question"] for r in rows}), "docs": len(blind_docs),
             "by_type": dict(Counter(r["type"] for r in rows)), "by_behavior": dict(Counter(r["behavior"] for r in rows)),
             "by_subtype": dict(Counter(r["subtype"] for r in rows if r["subtype"])),
             "by_generation": dict(Counter(r["generation_method"] for r in rows)),
             "max_question_jaccard_vs_previous": round(sims[0][0], 3), "top_similar": [list(s) for s in sims[:5]],
             "sha256": hashlib.sha256((out / "blind.jsonl").read_bytes()).hexdigest(), "problems": problems}
    (out / "manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    sheet = HERE / "blind_review_sheet.csv"
    prev = {}
    if sheet.exists():
        with open(sheet, encoding="utf-8-sig") as f:
            prev = {r["id"]: r for r in csv.DictReader(f)}
    cols = ["id", "type", "subtype", "behavior", "question", "reference_answer", "facts", "forbid", "gold_sources",
            "review_status", "reviewer", "review_note"]
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            p = prev.get(r["id"], {})
            w.writerow({"id": r["id"], "type": r["type"], "subtype": r["subtype"], "behavior": r["behavior"],
                        "question": r["question"], "reference_answer": r["reference_answer"],
                        "facts": " / ".join(r["facts"]), "forbid": " / ".join(r["forbid"]),
                        "gold_sources": "; ".join(f"{e['doc']}#{e['section']}" for e in r["evidence"] if e["cid"] in r["gold_cids"]),
                        "review_status": p.get("review_status") or "provisional", "reviewer": p.get("reviewer", ""),
                        "review_note": p.get("review_note", "")})
    print(json.dumps({k: v for k, v in stats.items() if k != "top_similar"}, ensure_ascii=False, indent=1))
    for p in problems:
        print("问题：", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
