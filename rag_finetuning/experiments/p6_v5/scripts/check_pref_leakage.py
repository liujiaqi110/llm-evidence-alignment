"""P6-v5 偏好数据隔离与泄漏检查（只读）。

    python -m rag_finetuning.experiments.p6_v5.scripts.check_pref_leakage

检查：
1 文档隔离：训练用语料（P6-v1 train、P6-v3 train、P6-v4 train）与全部评测语料（hard_dev、test_v5、P6-v1 test/val、P6-v2 盲测、test_v3、test_v4）
  无文档重名、无片段文本重复；
2 偏好对 / SFT 新样本的每个证据片段 ID 都属于训练语料；
3 问题相似度：与全部评测集问题的最大字符 bigram Jaccard，以及 > 0.5 的数量；训练集内部近重复（> 0.8）与完全重复；
4 冻结评测集哈希未变：hard_dev / test_v5 冻结输入与 freeze.json、manifest 一致；
5 构造代码静态检查：偏好数据构造脚本不读取评测集参考答案 / 必要事实 / Oracle 输出（评测集问题只用于相似度检查）。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from rag_finetuning.common import REPO, chunk_document, read_jsonl
from rag_finetuning.scripts import build_dataset as bd

V5 = Path(__file__).resolve().parents[1]
EXP = V5.parent
OUT = V5 / "data" / "train_pref"
TRAIN_DIRS = [REPO / "rag_finetuning/data/corpus/train", EXP / "p6_v3/data/corpus/train", EXP / "p6_v4/data/corpus/train"]
EVAL_DIRS = [V5 / "data/hard_dev/corpus", V5 / "data/test_v5/corpus", REPO / "rag_finetuning/data/corpus/test", REPO / "rag_finetuning/data/corpus/val",
             EXP / "p6_v2/blind/corpus", EXP / "p6_v3/data/test_v3/corpus", EXP / "p6_v4/data/test_v4/corpus"]
EVAL_Q = [V5 / "data/hard_dev/built/hard_dev.jsonl", V5 / "data/test_v5/built/test_v5.jsonl", REPO / "rag_finetuning/data/built/test.jsonl",
          REPO / "rag_finetuning/data/built/val.jsonl", EXP / "p6_v2/blind/built/blind.jsonl", EXP / "p6_v3/data/test_v3/built/test_v3.jsonl",
          EXP / "p6_v4/data/test_v4/built/test_v4.jsonl"]
BUILD_SCRIPTS = ["build_pref_ef.py", "build_pref_pairs.py"]


def chunks_of(dirs):
    names, texts, cids = {}, set(), {}
    for d in dirs:
        for p in sorted(d.glob("*.md")):
            names[p.stem] = str(d.relative_to(REPO))
            for c in chunk_document(p, d.name):
                texts.add(c.text)
                cids[c.cid] = str(d.relative_to(REPO))
    return names, texts, cids


def main() -> int:
    tn, tt, tc = chunks_of(TRAIN_DIRS)
    en, et, ec = chunks_of(EVAL_DIRS)
    pairs = read_jsonl(OUT / "dpo_pairs.jsonl")
    nat = read_jsonl(OUT / "natural_ef.jsonl")
    used_cids = {e["cid"] for p in pairs for e in p["evidence"]} | {e["cid"] for r in nat for e in r["evidence"]}
    eval_q = [r["question"] for p in EVAL_Q if p.exists() for r in read_jsonl(p)]
    train_q = sorted({p["question"] for p in pairs} | {r["question"] for r in nat})
    sims = sorted(((max(bd.jaccard(q, e) for e in eval_q), q) for q in train_q), reverse=True)
    near_internal = 0
    for i, a in enumerate(train_q):
        for b in train_q[i + 1:]:
            if bd.jaccard(a, b) > 0.8:
                near_internal += 1
    fz = json.loads((V5 / "configs/freeze.json").read_text(encoding="utf-8"))
    frozen = {}
    for name in ("hard_dev", "test_v5"):
        h = hashlib.sha256((V5 / "data/frozen" / f"{name}.jsonl").read_bytes()).hexdigest()
        man = json.loads((V5 / "data/frozen" / f"{name}.manifest.json").read_text(encoding="utf-8"))["frozen_sha256"]
        frozen[name] = {"current": h, "manifest": man, "freeze_json": fz[name]["frozen_sha256"], "unchanged": h == man == fz[name]["frozen_sha256"]}
    static = {}
    for s in BUILD_SCRIPTS:
        src = (V5 / "scripts" / s).read_text(encoding="utf-8")
        static[s] = {"reads_eval_reference_answers_or_facts": bool(re.search(r"(hard_dev|test_v5).{0,80}(reference_answer|\[\"facts\"\]|fact_cids)", src, flags=re.S)),
                     "reads_oracle_outputs": "oracle_ablation" in src or "diagnostics" in src,
                     "reads_eval_questions_for_similarity_only": "eval_docs" in src}
    rep = {"1_doc_isolation": {"train_docs": len(tn), "eval_docs": len(en), "name_overlap": sorted(set(tn) & set(en)),
                               "chunk_text_overlap": len(tt & et)},
           "2_evidence_from_training_corpus": {"evidence_chunks_used": len(used_cids), "not_in_training_corpus": sorted(used_cids - set(tc))[:20],
                                               "in_eval_corpus": sorted(used_cids & set(ec))[:20]},
           "3_question_similarity": {"n_train_questions": len(train_q), "n_eval_questions": len(eval_q), "max_jaccard_vs_eval": round(sims[0][0], 3) if sims else None,
                                     "n_over_0.5": sum(1 for s, _ in sims if s > 0.5), "top5": [[round(s, 3), q] for s, q in sims[:5]],
                                     "internal_near_duplicate_pairs(>0.8)": near_internal,
                                     "pairs_per_question_max": max(sum(1 for p in pairs if p["question"] == q) for q in train_q) if pairs else 0},
           "4_frozen_eval_hashes": frozen, "5_static_code_check": static}
    ok = (not rep["1_doc_isolation"]["name_overlap"] and rep["1_doc_isolation"]["chunk_text_overlap"] == 0
          and not rep["2_evidence_from_training_corpus"]["not_in_training_corpus"] and not rep["2_evidence_from_training_corpus"]["in_eval_corpus"]
          and rep["3_question_similarity"]["n_over_0.5"] == 0 and all(v["unchanged"] for v in frozen.values())
          and not any(v["reads_eval_reference_answers_or_facts"] or v["reads_oracle_outputs"] for v in static.values()))
    rep["all_checks_passed"] = ok
    (V5 / "results" / "pref_leakage_check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
