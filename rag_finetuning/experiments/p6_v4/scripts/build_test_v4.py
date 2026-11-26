"""构建 P6-v4 独立测试集 test_v4，并做独立性与参考答案自检。

    python -m rag_finetuning.experiments.p6_v4.scripts.build_test_v4

* facts 写作 "事实@文档#小节"：解析为 facts + fact_cids（每个必要事实的支持片段）。
* 复用 v1 的 build_record 组装证据；冲突 / 通知 / 来函文档只在题目显式引用时出现。
* 独立性：与 P6-v1/v2/v3 全部语料、P6-v4 训练 / 开发语料无重名、无片段重复；报告问题相似度。
* 每条参考答案必须通过评分器 v3：content_ok、format_ok、（多证据题）覆盖与归属、陈述支持（全分母）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from rag_finetuning.common import BUILT, chunk_document, load_corpus, read_jsonl, write_jsonl
from rag_finetuning.experiments.p6_v4.scripts.scoring_v3 import score_item_v3
from rag_finetuning.scripts import build_dataset as bd

V4 = Path(__file__).resolve().parents[1]
EXP = V4.parent
TEST = V4 / "data" / "test_v4"
EXPLICIT = {"notices_2026", "ops_dept_notice", "partner_quote_letter"}


def load_dir(d: Path, split: str) -> dict:
    out = {}
    for p in sorted(d.glob("*.md")):
        for c in chunk_document(p, split):
            out[f"{c.doc}#{c.section}"] = c
    return out


def previous_corpora() -> dict:
    prev = dict(load_corpus())
    for d, s in ((EXP / "p6_v2" / "blind" / "corpus", "blind"), (EXP / "p6_v3" / "data" / "corpus" / "train", "v3train"),
                 (EXP / "p6_v3" / "data" / "corpus" / "dev", "v3dev"), (EXP / "p6_v3" / "data" / "test_v3" / "corpus", "v3test"),
                 (V4 / "data" / "corpus" / "train", "v4train"), (V4 / "data" / "corpus" / "dev", "v4dev")):
        if d.exists():
            prev.update({f"{s}:{k}": v for k, v in load_dir(d, s).items()})
    return prev


def main() -> int:
    chunks = load_dir(TEST / "corpus", "test_v4")
    prev = previous_corpora()
    problems = []
    if {c.doc for c in chunks.values()} & {c.doc for c in prev.values()}:
        problems.append(f"文档重名：{sorted({c.doc for c in chunks.values()} & {c.doc for c in prev.values()})}")
    if {c.text for c in chunks.values()} & {c.text for c in prev.values()}:
        problems.append("片段文本与已有语料重复")
    bd.EXPLICIT_ONLY = set(bd.EXPLICIT_ONLY) | EXPLICIT
    items = yaml.safe_load((TEST / "test_v4.yaml").read_text(encoding="utf-8"))
    rows = []
    for it in items:
        raw = it.get("facts") or []
        facts, fkeys = [], []
        for f in raw:
            fact, _, key = str(f).rpartition("@") if "@" in str(f) else (str(f), "", "")
            facts.append(fact)
            fkeys.append(key or None)
        it2 = dict(it, facts=facts)
        r = bd.build_record(it2, "test_v4", chunks)
        r["fact_cids"] = [chunks[k].cid if k else None for k in fkeys]
        for k in fkeys:
            if k and chunks[k].cid not in {e["cid"] for e in r["evidence"]}:
                problems.append(f"{r['id']}: 事实出处 {k} 不在证据中")
        r["no_conflict"] = bool(it.get("no_conflict"))
        r["conflict_kind"] = it.get("conflict_kind")
        r["topic_docs"] = list(it.get("topic") or [])
        r["answerability"] = {"answer": "full", "conflict": "full", "partial": "partial", "refuse": "none"}[r["behavior"]]
        r["dataset_version"] = "p6v4-test-1.0"
        r["generation_method"] = "claude_authored"
        s = score_item_v3(r, r["reference_answer"])
        bad = []
        if not (s["content_ok"] and s["format_ok"]):
            bad.append(f"content={s['content_ok']} format={s['format_ok']} forbid={s['forbidden_hits']} nums={s['unsupported_numbers']}")
        if s["is_multi"] and not (s["multi_content_complete"] and s["multi_chunk_coverage"]):
            bad.append("多证据覆盖未通过")
        if s["fact_attribution_ok"] is False:
            bad.append("事实归属未通过")
        if r["behavior"] != "refuse" and not s["claim_support_full"]:
            bad.append(f"陈述支持未通过 {s['n_supported']}/{s['n_claims']}")
        if bad:
            problems.append(f"{r['id']}: " + "；".join(bad))
        rows.append(r)
    ids = Counter(r["id"] for r in rows)
    problems += [f"重复题号 {i}" for i, n in ids.items() if n > 1]
    prev_q = [r["question"] for s in ("train", "val", "test") for r in read_jsonl(BUILT / f"{s}.jsonl")]
    for p in (EXP / "p6_v2" / "blind" / "built" / "blind.jsonl", EXP / "p6_v3" / "data" / "test_v3" / "built" / "test_v3.jsonl",
              EXP / "p6_v3" / "data" / "v3" / "full" / "train.jsonl", EXP / "p6_v3" / "data" / "v3" / "dev.jsonl",
              V4 / "data" / "v4" / "full" / "train.jsonl"):
        if p.exists():
            prev_q += [r["question"] for r in read_jsonl(p)]
    sims = sorted(((max(bd.jaccard(r["question"], q) for q in prev_q), r["id"], r["question"]) for r in rows), reverse=True)
    out = TEST / "built"
    write_jsonl(out / "test_v4.jsonl", rows)
    stats = {"n": len(rows), "docs": len({c.doc for c in chunks.values()}), "by_behavior": dict(Counter(r["behavior"] for r in rows)),
             "by_subtype": dict(Counter(r["subtype"] or "plain" for r in rows)),
             "by_conflict_kind": dict(Counter(r["conflict_kind"] for r in rows if r["conflict_kind"])),
             "multi_items(承载事实的金标准片段 ≥ 2)": sum(1 for r in rows if score_item_v3(r, r["reference_answer"])["is_multi"]),
             "max_question_jaccard_vs_previous": round(sims[0][0], 3), "top_similar": [list(x) for x in sims[:5]],
             "sha256": hashlib.sha256((out / "test_v4.jsonl").read_bytes()).hexdigest(), "problems": problems}
    (out / "manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "top_similar"}, ensure_ascii=False, indent=1))
    for p in problems:
        print("问题：", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
