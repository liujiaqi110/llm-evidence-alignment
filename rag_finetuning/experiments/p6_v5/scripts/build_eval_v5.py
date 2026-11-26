"""构建 P6-v5 困难开发集（hard_dev）或独立测试集（test_v5），并做独立性与参考答案自检。

    python -m rag_finetuning.experiments.p6_v5.scripts.build_eval_v5 --split hard_dev
    python -m rag_finetuning.experiments.p6_v5.scripts.build_eval_v5 --split test_v5

* 文档写在 data/<split>/corpus_bundle.md：以 "=== 文档名 ===" 分隔，构建时拆成 corpus/<文档名>.md（再分块）。
* 题目：data/<split>/<split>.yaml。facts 写作 "事实@文档#小节序号"；"a|b" 为备选写法。
* 证据组装复用 v1 的 build_record（金标准 + 同文档难负例 + 其他文档干扰片段，随机打乱）；额外支持：
    order: gold_first | gold_last    —— 把金标准片段整体放到最前 / 最后（其余顺序不变）；
    reverse_of: <题号>               —— 与另一题相同的问题、答案、事实，证据顺序完全颠倒（顺序对照）。
* 独立性：与 P6-v1～v4 全部语料、以及另一个 v5 评测集无重名、无片段重复；报告问题字符 bigram 相似度。
* 每条参考答案必须通过评分器 v4：content_ok_v4、format_ok、多证据完整与覆盖、事实归属、陈述支持（全分母）。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

from rag_finetuning.common import BUILT, read_jsonl, write_jsonl
from rag_finetuning.experiments.p6_v4.scripts.build_test_v4 import load_dir, previous_corpora
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import score_item_v4
from rag_finetuning.scripts import build_dataset as bd

V5 = Path(__file__).resolve().parents[1]
EXP = V5.parent
SPLITS = ("hard_dev", "test_v5")


def explode_bundle(d: Path) -> None:
    raw = (d / "corpus_bundle.md").read_text(encoding="utf-8")
    out = d / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.md"):
        old.unlink()
    for m in re.finditer(r"(?ms)^=== *(\w+) *===\s*\n(.*?)(?=^=== |\Z)", raw):
        (out / f"{m.group(1)}.md").write_text(m.group(2).strip() + "\n", encoding="utf-8", newline="\n")


def reorder(rec: dict, mode: str) -> None:
    gold = set(rec["gold_cids"])
    g = [e for e in rec["evidence"] if e["cid"] in gold]
    o = [e for e in rec["evidence"] if e["cid"] not in gold]
    rec["evidence"] = g + o if mode == "gold_first" else o + g


def build(split: str) -> int:
    d = V5 / "data" / split
    explode_bundle(d)
    chunks = load_dir(d / "corpus", split)
    prev = previous_corpora()
    other = [s for s in SPLITS if s != split][0]
    if (V5 / "data" / other / "corpus").exists():
        prev.update({f"{other}:{k}": v for k, v in load_dir(V5 / "data" / other / "corpus", other).items()})
    problems = []
    if {c.doc for c in chunks.values()} & {c.doc for c in prev.values()}:
        problems.append(f"文档重名：{sorted({c.doc for c in chunks.values()} & {c.doc for c in prev.values()})}")
    if {c.text for c in chunks.values()} & {c.text for c in prev.values()}:
        problems.append("片段文本与已有语料重复")
    spec = yaml.safe_load((d / f"{split}.yaml").read_text(encoding="utf-8"))
    bd.EXPLICIT_ONLY = set(bd.EXPLICIT_ONLY) | set(spec["explicit_docs"])
    items = spec["items"]
    by_id = {it["id"]: it for it in items}
    rows, built = [], {}
    for it in items:
        base = by_id[it["reverse_of"]] if it.get("reverse_of") else it
        raw = base.get("facts") or []
        facts, fkeys = [], []
        for f in raw:
            fact, _, key = str(f).rpartition("@") if "@" in str(f) else (str(f), "", "")
            facts.append(fact)
            fkeys.append(key or None)
        if it.get("reverse_of"):
            r = copy.deepcopy(built[it["reverse_of"]])
            r["id"] = it["id"]
            r["evidence"] = list(reversed(r["evidence"]))
            r["order_pair"] = it["reverse_of"]
            built[it["reverse_of"]]["order_pair"] = it["id"]
        else:
            r = bd.build_record(dict(it, facts=facts), split, chunks)
            r["fact_cids"] = [chunks[k].cid if k else None for k in fkeys]
            for k in fkeys:
                if k and chunks[k].cid not in {e["cid"] for e in r["evidence"]}:
                    problems.append(f"{r['id']}: 事实出处 {k} 不在证据中")
            if it.get("order"):
                reorder(r, it["order"])
            r["no_conflict"] = bool(it.get("no_conflict"))
            r["conflict_kind"] = it.get("conflict_kind")
            r["hard_tags"] = list(it.get("tags") or [])
            r["answerability"] = {"answer": "full", "conflict": "full", "partial": "partial", "refuse": "none"}[r["behavior"]]
            r["dataset_version"] = f"p6v5-{split}-1.0"
            r["generation_method"] = "claude_authored"
            r["review_status"] = "provisional"
        gpos = [i for i, e in enumerate(r["evidence"]) if e["cid"] in set(r["gold_cids"])]
        r["gold_positions"] = gpos
        s = score_item_v4(r, r["reference_answer"])
        bad = []
        if not (s["content_ok_v4"] and s["content_ok"] and s["format_ok"]):
            bad.append(f"content={s['content_ok_v4']}/{s['content_ok']} format={s['format_ok']} forbid={s['forbidden_hits']} "
                       f"nums={s['unsupported_numbers']} missing={s['missing_facts']}")
        if s["is_multi"] and not (s["multi_complete"] and s["multi_chunk_coverage"]):
            bad.append("多证据覆盖未通过")
        if s["fact_attribution_ok"] is False:
            bad.append("事实归属未通过")
        if r["behavior"] != "refuse" and not s["claim_support_full"]:
            bad.append(f"陈述支持未通过 {s['n_supported']}/{s['n_claims']}")
        if bad:
            problems.append(f"{r['id']}: " + "；".join(bad))
        built[r["id"]] = r
        rows.append(r)
    ids = Counter(r["id"] for r in rows)
    problems += [f"重复题号 {i}" for i, n in ids.items() if n > 1]
    prev_q = [r["question"] for s in ("train", "val", "test") for r in read_jsonl(BUILT / f"{s}.jsonl")]
    for p in (EXP / "p6_v2/blind/built/blind.jsonl", EXP / "p6_v3/data/test_v3/built/test_v3.jsonl", EXP / "p6_v3/data/v3/full/train.jsonl",
              EXP / "p6_v3/data/v3/dev.jsonl", EXP / "p6_v4/data/v4/full/train.jsonl", EXP / "p6_v4/data/v4/dev.jsonl",
              EXP / "p6_v4/data/test_v4/built/test_v4.jsonl", V5 / "data" / other / "built" / f"{other}.jsonl"):
        if p.exists():
            prev_q += [r["question"] for r in read_jsonl(p)]
    sims = sorted(((max(bd.jaccard(r["question"], q) for q in prev_q), r["id"], r["question"]) for r in rows), reverse=True)
    out = d / "built"
    write_jsonl(out / f"{split}.jsonl", rows)
    sc = [score_item_v4(r, r["reference_answer"]) for r in rows]
    stats = {"n": len(rows), "docs": len({c.doc for c in chunks.values()}), "chunks": len(chunks),
             "by_behavior": dict(Counter(r["behavior"] for r in rows)),
             "by_subtype": dict(Counter(r["subtype"] or "plain" for r in rows)),
             "by_conflict_kind": dict(Counter(r["conflict_kind"] for r in rows if r.get("conflict_kind"))),
             "multi_items(v3口径)": sum(1 for s in sc if s["is_multi"]), "multi_items_v5": sum(1 for s in sc if s["is_multi_v5"]),
             "multi_3plus_gold": sum(1 for r, s in zip(rows, sc) if s["is_multi_v5"] and s["n_gold_chunks"] >= 3),
             "cross_doc_multi": sum(1 for r, s in zip(rows, sc) if s["is_multi_v5"] and len({e["doc"] for e in r["evidence"] if e["cid"] in set(r["gold_cids"])}) >= 2),
             "gold_not_first": sum(1 for r in rows if r["gold_positions"] and r["gold_positions"][0] > 0),
             "order_pairs": sum(1 for r in rows if r.get("order_pair")) // 2,
             "max_question_jaccard_vs_previous": round(sims[0][0], 3), "top_similar": [list(x) for x in sims[:5]],
             "sha256": hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest(), "problems": problems}
    (out / "manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "top_similar"}, ensure_ascii=False, indent=1))
    for p in problems:
        print("问题：", p)
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, required=True)
    return build(ap.parse_args(argv).split)


if __name__ == "__main__":
    sys.exit(main())
