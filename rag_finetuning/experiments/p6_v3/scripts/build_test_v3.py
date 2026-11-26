"""构建并封存 P6-v3 独立测试集 test_v3（手写题目），生成人工复核表。

    python -m rag_finetuning.experiments.p6_v3.scripts.build_test_v3

校验：文档与 P6-v1/v2、P6-v3 训练 / 开发语料无重名、无片段重复；问题与以往全部问题的相似度；
每条参考答案必须通过评分器 v2：content_ok、format_ok、全部陈述有支持（同时检验支持度阈值不会误伤正确答案）。
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from rag_finetuning.common import BUILT, DATASET_VERSION, chunk_document, load_corpus, read_jsonl, write_jsonl
from rag_finetuning.experiments.p6_v3.scripts.scoring_v2 import score_item_v2
from rag_finetuning.scripts import build_dataset as bd

V3 = Path(__file__).resolve().parents[1]
TEST = V3 / "data" / "test_v3"
EXPLICIT = {"policy_updates", "supplier_mail"}


def load_chunks(dirpath: Path, split: str) -> dict:
    out = {}
    for p in sorted(dirpath.glob("*.md")):
        for c in chunk_document(p, split):
            out[f"{c.doc}#{c.section}"] = c
    return out


def main() -> int:
    chunks = load_chunks(TEST / "corpus", "test_v3")
    prev_docs = {c.doc for c in load_corpus().values()}
    prev_texts = {c.text for c in load_corpus().values()}
    for d in ("train", "dev"):
        for c in load_chunks(V3 / "data" / "corpus" / d, d).values():
            prev_docs.add(c.doc)
            prev_texts.add(c.text)
    blind = V3.parent / "p6_v2" / "blind" / "corpus"
    for c in load_chunks(blind, "blind").values():
        prev_docs.add(c.doc)
        prev_texts.add(c.text)
    problems = []
    docs = {c.doc for c in chunks.values()}
    if docs & prev_docs:
        problems.append(f"测试文档与已有文档重名：{sorted(docs & prev_docs)}")
    if {c.text for c in chunks.values()} & prev_texts:
        problems.append("测试片段与已有片段文本重复")
    bd.EXPLICIT_ONLY = set(bd.EXPLICIT_ONLY) | EXPLICIT
    items = yaml.safe_load((TEST / "test_v3.yaml").read_text(encoding="utf-8"))
    rows = []
    for it in items:
        r = bd.build_record(it, "test_v3", chunks)
        r["no_conflict"] = bool(it.get("no_conflict"))
        r["topic_docs"] = list(it.get("topic") or [])
        r["answerability"] = {"answer": "full", "conflict": "full", "partial": "partial", "refuse": "none"}[r["behavior"]]
        r["dataset_version"] = "p6v3-test-1.0"
        rows.append(r)
        s = score_item_v2(r, r["reference_answer"])
        if not (s["content_ok"] and s["format_ok"] and s["all_claims_supported"] in (None, True)):
            problems.append(f"{r['id']}: 参考答案未通过 v2 自检 content={s['content_ok']} format={s['format_ok']} "
                            f"support={[ (c['text'][:20], c['reason'], c['recall']) for c in s['claims'] if not c['supported']]}")
    ids = Counter(r["id"] for r in rows)
    problems += [f"重复题号 {i}" for i, n in ids.items() if n > 1]
    prev_q = [r["question"] for s in ("train", "val", "test") for r in read_jsonl(BUILT / f"{s}.jsonl")]
    prev_q += [r["question"] for r in read_jsonl(V3.parent / "p6_v2" / "blind" / "built" / "blind.jsonl")]
    for d in ("train", "dev"):
        p = V3 / "data" / "built" / f"{d}.jsonl"
        if p.exists():
            prev_q += [r["question"] for r in read_jsonl(p)]
    sims = sorted(((max(bd.jaccard(r["question"], q) for q in prev_q), r["id"], r["question"]) for r in rows), reverse=True)
    out = TEST / "built"
    write_jsonl(out / "test_v3.jsonl", rows)
    stats = {"n": len(rows), "unique_questions": len({r["question"] for r in rows}), "docs": len(docs),
             "by_behavior": dict(Counter(r["behavior"] for r in rows)), "by_type": dict(Counter(r["type"] for r in rows)),
             "by_subtype": dict(Counter(r["subtype"] or "plain" for r in rows)),
             "no_conflict_items": sum(r["no_conflict"] for r in rows),
             "max_question_jaccard_vs_all_previous": round(sims[0][0], 3), "top_similar": [list(x) for x in sims[:5]],
             "sha256": hashlib.sha256((out / "test_v3.jsonl").read_bytes()).hexdigest(), "problems": problems}
    (out / "manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "top_similar"}, ensure_ascii=False, indent=1))
    for p in problems:
        print("问题：", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
