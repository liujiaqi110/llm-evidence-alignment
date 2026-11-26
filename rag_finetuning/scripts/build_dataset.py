"""Build the P6 RAG SFT dataset: QA specs (data/qa/*.yaml) + corpus → data/built/{train,val,test}.jsonl.

    python -m rag_finetuning.scripts.build_dataset            # build + validate + leakage report
    python -m rag_finetuning.scripts.build_dataset --check    # validate only (non-zero exit on any problem)

Deterministic (evidence order / distractors are seeded by item id). Every record keeps provenance:
``generation_method`` (claude_authored | derived_gold_removed), ``review_status`` (provisional until a human marks it
reviewed in reports/test_review_sheet.csv), ``source_docs`` and ``dataset_version``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import yaml

from rag_finetuning.common import (BUILT, CANONICAL_REFUSAL, DATASET_VERSION, EXPLICIT_ONLY, QA_DIR, ROOT, SPLITS,
                                   Chunk, load_corpus, messages_for, write_jsonl)
from rag_finetuning.evaluation.scoring import score_item

REVIEW_SHEET = ROOT / "reports" / "test_review_sheet.csv"


def _rng(item_id: str) -> random.Random:
    return random.Random(int(hashlib.sha256(item_id.encode()).hexdigest()[:16], 16))


def render_answer(template: str, gold: List[Chunk]) -> str:
    text = template.replace("\\n", "\n").strip()
    used = []
    for i, c in enumerate(gold, 1):
        if "{%d}" % i in text:
            used.append(c.cid)
        text = text.replace("{%d}" % i, c.cid)
    left = re.findall(r"\{\d+\}", text)
    if left:
        raise ValueError(f"unresolved placeholders {left}")
    return text + "\n\n引用：" + ", ".join(used), used


def pick_evidence(item: dict, gold: List[Chunk], chunks: Dict[str, Chunk], split: str) -> List[Chunk]:
    rng = _rng(item["id"])
    gold_keys = {f"{c.doc}#{c.section}" for c in gold}
    extra = [chunks[k] for k in item.get("extra") or []]
    target = max(4, len(gold) + len(extra) + (2 if len(gold) >= 3 else 1))
    chosen = list(gold) + extra
    taken = gold_keys | {f"{c.doc}#{c.section}" for c in extra}
    # 1) hard negatives: other sections of the gold / topic documents
    topic_docs = [c.doc for c in gold] + list(item.get("topic") or [])
    same = [c for k, c in chunks.items() if c.split == split and c.doc in topic_docs and k not in taken]
    rng.shuffle(same)
    n_same = (target - len(chosen)) if not gold else min(2, target - len(chosen))
    for c in same[:max(0, n_same)]:
        chosen.append(c)
        taken.add(f"{c.doc}#{c.section}")
    # 2) fill with chunks from other documents of the same split (never conflict / injection documents)
    pool = [c for k, c in chunks.items() if c.split == split and k not in taken and c.doc not in EXPLICIT_ONLY
            and c.doc not in topic_docs]
    rng.shuffle(pool)
    chosen += pool[:max(0, target - len(chosen))]
    rng.shuffle(chosen)
    return chosen


def build_record(item: dict, split: str, chunks: Dict[str, Chunk]) -> dict:
    for k in list(item.get("gold") or []) + list(item.get("extra") or []) + [f"{d}#1" for d in item.get("topic") or []]:
        if k not in chunks:
            raise ValueError(f"{item['id']}: unknown chunk {k}")
        if chunks[k].split != split:
            raise ValueError(f"{item['id']}: {k} belongs to split {chunks[k].split}, not {split}")
    gold = [chunks[k] for k in item.get("gold") or []]
    beh = item["behavior"]
    if beh == "refuse":
        if gold:
            raise ValueError(f"{item['id']}: refuse items must not have gold evidence")
        ref, used = CANONICAL_REFUSAL, []
    else:
        if not gold or not item.get("answer"):
            raise ValueError(f"{item['id']}: {beh} items need gold + answer")
        ref, used = render_answer(item["answer"], gold)
        if set(used) != {c.cid for c in gold}:
            raise ValueError(f"{item['id']}: every gold chunk must be cited by the reference answer")
    ev = pick_evidence(item, gold, chunks, split)
    return {
        "id": item["id"], "split": split, "type": item["type"], "subtype": item.get("subtype", ""),
        "behavior": beh, "question": item["question"],
        "evidence": [{"cid": c.cid, "source": c.source, "doc": c.doc, "section": c.section, "text": c.text} for c in ev],
        "gold_cids": [c.cid for c in gold], "reference_answer": ref,
        "facts": [str(f) for f in item.get("facts") or []], "forbid": item.get("forbid") or [],
        "format": item.get("format"), "source_docs": sorted({c.doc for c in ev}),
        "generation_method": "claude_authored", "derived_from": None,
        "review_status": "provisional", "dataset_version": DATASET_VERSION,
    }


def derive_refusal(rec: dict, chunks: Dict[str, Chunk], split: str) -> dict:
    """Same question, gold evidence removed (replaced by further same-document sections) → correct behaviour is refusal."""
    gold = set(rec["gold_cids"])
    keep = [e for e in rec["evidence"] if e["cid"] not in gold]
    have = {e["cid"] for e in keep}
    docs = {chunks_by_cid(chunks)[g].doc for g in gold}
    rng = _rng(rec["id"] + "_nogold")
    cand = [c for c in chunks.values() if c.split == split and c.doc in docs and c.cid not in gold and c.cid not in have]
    rng.shuffle(cand)
    for c in cand[:max(0, 4 - len(keep))]:
        keep.append({"cid": c.cid, "source": c.source, "doc": c.doc, "section": c.section, "text": c.text})
    rng.shuffle(keep)
    out = dict(rec, id=rec["id"] + "_nogold", type="C", subtype="derived_gold_removed", behavior="refuse",
               evidence=keep, gold_cids=[], reference_answer=CANONICAL_REFUSAL, facts=[], format=None,
               forbid=list(rec.get("forbid") or []), source_docs=sorted({e["doc"] for e in keep}),
               generation_method="derived_gold_removed", derived_from=rec["id"])
    # the original answer's distinctive facts must not appear in a refusal → reuse them as forbidden patterns
    out["forbid"] += [re.escape(f.split("|")[0]) for f in rec["facts"] if re.search(r"\d", f)]
    return out


_CID_CACHE: Dict[int, Dict[str, Chunk]] = {}


def chunks_by_cid(chunks: Dict[str, Chunk]) -> Dict[str, Chunk]:
    key = id(chunks)
    if key not in _CID_CACHE:
        _CID_CACHE[key] = {c.cid: c for c in chunks.values()}
    return _CID_CACHE[key]


def bigrams(s: str) -> set:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard(a: str, b: str) -> float:
    x, y = bigrams(a), bigrams(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def leakage_report(built: Dict[str, List[dict]], chunks: Dict[str, Chunk]) -> dict:
    docs = {s: {c.doc for c in chunks.values() if c.split == s} for s in SPLITS}
    texts = {s: {c.text for c in chunks.values() if c.split == s} for s in SPLITS}
    ev_docs = {s: {d for r in built[s] for d in r["source_docs"]} for s in SPLITS}
    rep = {"doc_overlap": {f"{a}-{b}": sorted(docs[a] & docs[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
           "chunk_text_overlap": {f"{a}-{b}": len(texts[a] & texts[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
           "evidence_doc_overlap_train_test": sorted(ev_docs["train"] & ev_docs["test"]),
           "test_reference_in_train_targets": 0, "question_similarity": {}}
    train_targets = {r["reference_answer"] for r in built["train"]}
    rep["test_reference_in_train_targets"] = sum(1 for r in built["test"] if r["reference_answer"] in train_targets
                                                 and r["behavior"] != "refuse")
    trq = [r["question"] for r in built["train"] if r["generation_method"] == "claude_authored"]
    for s in ("val", "test"):
        sims = []
        for r in built[s]:
            if r["generation_method"] != "claude_authored":
                continue
            best = max(((jaccard(r["question"], q), q) for q in trq), key=lambda t: t[0])
            sims.append({"id": r["id"], "question": r["question"], "max_jaccard": round(best[0], 3), "nearest_train": best[1]})
        sims.sort(key=lambda d: -d["max_jaccard"])
        rep["question_similarity"][s] = {"max": sims[0]["max_jaccard"] if sims else None,
                                         "n_over_0.5": sum(1 for d in sims if d["max_jaccard"] > 0.5), "top5": sims[:5]}
    # within-train near-duplicates (sample-inflation check)
    dup = []
    for i in range(len(trq)):
        for j in range(i + 1, len(trq)):
            if jaccard(trq[i], trq[j]) > 0.6:
                dup.append([trq[i], trq[j]])
    rep["train_near_duplicate_questions"] = dup
    rep["ok"] = (not any(rep["doc_overlap"].values()) and not any(rep["chunk_text_overlap"].values())
                 and not rep["evidence_doc_overlap_train_test"] and rep["test_reference_in_train_targets"] == 0)
    return rep


def self_check(built: Dict[str, List[dict]]) -> List[str]:
    """Each reference answer must pass the scorer it will be graded with (validates both data and rules)."""
    problems = []
    for s, rows in built.items():
        ids = Counter(r["id"] for r in rows)
        problems += [f"{s}: duplicate id {i}" for i, n in ids.items() if n > 1]
        for r in rows:
            sc = score_item(r, r["reference_answer"])
            if not sc["pass"]:
                problems.append(f"{s}/{r['id']}: reference fails own scoring: errors={sc['errors']} "
                                f"pred={sc['pred_behavior']} unsupported={sc['unsupported_numbers']} forbid={sc['forbidden_hits']}")
    return problems


def token_lengths(built: Dict[str, List[dict]]) -> dict:
    """Prompt character lengths (tokenizer-free proxy; the trainer re-checks real token counts)."""
    out = {}
    for s, rows in built.items():
        L = sorted(sum(len(m["content"]) for m in messages_for(r)) + len(r["reference_answer"]) for r in rows)
        out[s] = {"chars_min": L[0], "chars_median": L[len(L) // 2], "chars_max": L[-1]} if L else {}
    return out


def write_review_sheet(test_rows: List[dict]) -> None:
    """Human-review sheet for the test set. Existing human decisions are preserved across rebuilds."""
    prev = {}
    if REVIEW_SHEET.exists():
        with open(REVIEW_SHEET, encoding="utf-8-sig") as f:
            prev = {r["id"]: r for r in csv.DictReader(f)}
    REVIEW_SHEET.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "type", "behavior", "question", "reference_answer", "facts", "gold_sources", "review_status",
            "reviewer", "review_note"]
    with open(REVIEW_SHEET, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in test_rows:
            p = prev.get(r["id"], {})
            gold_src = "; ".join(f"{e['doc']}#{e['section']}" for e in r["evidence"] if e["cid"] in r["gold_cids"])
            w.writerow({"id": r["id"], "type": r["type"], "behavior": r["behavior"], "question": r["question"],
                        "reference_answer": r["reference_answer"], "facts": " / ".join(r["facts"]),
                        "gold_sources": gold_src, "review_status": p.get("review_status") or "provisional",
                        "reviewer": p.get("reviewer", ""), "review_note": p.get("review_note", "")})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate only, do not write files")
    a = ap.parse_args(argv)
    chunks = load_corpus()
    built: Dict[str, List[dict]] = {}
    for s in SPLITS:
        items = yaml.safe_load((QA_DIR / f"{s}.yaml").read_text(encoding="utf-8"))
        rows = [build_record(it, s, chunks) for it in items]
        rows += [derive_refusal(r, chunks, s) for r, it in zip(list(rows), items) if it.get("derive_refusal")]
        built[s] = rows
    problems = self_check(built)
    leak = leakage_report(built, chunks)
    if not leak["ok"]:
        problems.append("leakage check failed: " + json.dumps({k: leak[k] for k in leak if k != 'question_similarity'},
                                                              ensure_ascii=False))
    stats = {s: {"n": len(rows), "unique_questions": len({r["question"] for r in rows}),
                 "by_type": dict(Counter(r["type"] for r in rows)), "by_behavior": dict(Counter(r["behavior"] for r in rows)),
                 "by_generation": dict(Counter(r["generation_method"] for r in rows)),
                 "docs": len({c.doc for c in chunks.values() if c.split == s})} for s, rows in built.items()}
    manifest = {"dataset_version": DATASET_VERSION, "stats": stats, "lengths": token_lengths(built),
                "leakage": leak, "problems": problems,
                "files": {}}
    if not a.check:
        for s, rows in built.items():
            p = BUILT / f"{s}.jsonl"
            write_jsonl(p, rows)
            manifest["files"][s] = {"path": str(p.relative_to(ROOT)).replace("\\", "/"),
                                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        (BUILT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_review_sheet(built["test"])
    print(json.dumps({"stats": stats, "lengths": manifest["lengths"], "leak_ok": leak["ok"],
                      "q_sim": {s: {k: v for k, v in d.items() if k != "top5"} for s, d in leak["question_similarity"].items()},
                      "train_dups": leak["train_near_duplicate_questions"]}, ensure_ascii=False, indent=1))
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
